"""
Figure 6: Per-task MAML vs SetONet scatter plots across 3 OCP environments.

Each point = one held-out task.
x-axis = Model relative L2, y-axis = MAML relative L2.
Points above y=x => MAML performs worse.

Usage (matches Makefile):
    python src/plotting/plot_maml_scatter.py \
        --config configs \
        --data data \
        --checkpoints checkpoints \
        --output outputs/figures/figure6.pdf

This script both evaluates and plots (single script for Figure 6).
"""

import argparse
import copy
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

from src.setonet import SetONet, NormalizedSetONet
from src.normalization import GridEnvironmentNormalizer
from src.training.maml import MAMLMLP
from src.envs.create_p2p_cost import generate_dataset


def _inner(m):
    """The underlying SetONet (handles the NormalizedSetONet wrapper)."""
    return m.model if isinstance(m, NormalizedSetONet) else m


# Trunk-time convention differs by env: VaryingDynamicsData (P2P-Dynamics) trains
# with RAW integer timesteps (0,1,...,H-1); DoubleIntegratorLQRData (P2P-Cost) and
# QuadrotorDataLoader use NORMALIZED time t/H. Querying with the wrong convention
# leaves t=0 correct but corrupts every later step (silent closed-loop drift).
def _qtime(t, H, raw_time):
    return float(t) if raw_time else t / H


def _times_arr(H, raw_time, lib=np):
    return lib.arange(H, dtype=float) if raw_time else lib.linspace(0, 1, H, endpoint=False)


def setonet_freeze_filter(ft, freeze_mode):
    """eqx filter selecting trainable leaves for a (possibly Normalized) SetONet.

    freeze_mode: 'trunk' (branch only), 'last_branch', 'last_both'.
    """
    norm = isinstance(ft, NormalizedSetONet)
    trunk = (lambda m: m.model.trunk) if norm else (lambda m: m.trunk)
    rho_last = (lambda m: m.model.rho.layers[-1]) if norm else (lambda m: m.rho.layers[-1])
    trunk_last = (lambda m: m.model.trunk.layers[-1]) if norm else (lambda m: m.trunk.layers[-1])
    if freeze_mode == "trunk":  # branch only (freeze trunk)
        filt = jax.tree_util.tree_map(lambda x: eqx.is_array(x), ft)
        filt = eqx.tree_at(trunk, filt, replace=jax.tree_util.tree_map(lambda _: False, trunk(ft)))
    elif freeze_mode == "last_branch":
        filt = jax.tree_util.tree_map(lambda _: False, ft)
        filt = eqx.tree_at(rho_last, filt,
                           replace=jax.tree_util.tree_map(lambda x: eqx.is_array(x), rho_last(ft)))
    elif freeze_mode == "last_both":
        filt = jax.tree_util.tree_map(lambda _: False, ft)
        filt = eqx.tree_at(lambda m: (trunk_last(m), rho_last(m)), filt,
                           replace=(jax.tree_util.tree_map(lambda x: eqx.is_array(x), trunk_last(ft)),
                                    jax.tree_util.tree_map(lambda x: eqx.is_array(x), rho_last(ft))))
    else:
        raise ValueError(freeze_mode)
    return filt


def build_dynamics_normalized(base_setonet, cfg):
    """Wrap a base SetONet in NormalizedSetONet using the env config's ranges
    (leaf values are overwritten on deserialize, so max_acceleration is a placeholder)."""
    dc = cfg.get("data", {})
    xr = dc.get("workspace", {}).get("x_range", [-5.0, 5.0])
    mv = dc.get("dynamics_ranges", {}).get("max_velocity_range", [10.0, 15.0])[1]
    return NormalizedSetONet(base_setonet,
                             GridEnvironmentNormalizer(position_range=(xr[0], xr[1]),
                                                       max_velocity=mv, max_acceleration=1.0))

DEFAULT_SEED = 42
GRAD_STEPS = 25       # SetONet adaptation steps
MAML_GRAD_STEPS = 25  # MAML inner-loop steps
NUM_DEMOS = 10        # expert trajectories for adaptation (matches Figure 6 caption)
NUM_TASKS = 20
K = 64
N_EVAL = 32
NUM_EVAL_BATCHES = 5
FT_LR = 1e-4

COLORS = {
    "blue":   "#4477AA",
    "orange": "#EE7733",
    "green":  "#228833",
    "cyan":   "#66CCEE",
    "grey":   "#BBBBBB",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def relative_l2(pred_phys, target_phys):
    diff = pred_phys - target_phys
    return float(np.sqrt(np.sum(diff**2)) / (np.sqrt(np.sum(target_phys**2)) + 1e-12))


def predict_setonet(model, src_input, src_values, tgt_states):
    pred = jax.vmap(
        jax.vmap(model, in_axes=(None, None, 0)),
        in_axes=(None, None, 0),
    )(src_input, src_values, jnp.array(tgt_states))
    return np.array(pred)


def predict_maml(model, tgt_states):
    pred = jax.vmap(
        jax.vmap(model, in_axes=(0,)),
        in_axes=(0,),
    )(jnp.array(tgt_states))
    return np.array(pred)


class TransferTaskDataLoader:
    """Splits each task's data into train (adaptation) and holdout (eval)."""

    def __init__(self, dataset, train_holdout_split=0.25, normalize=True):
        self.dataset = dataset
        self.normalize = normalize
        self.num_goals = int(dataset['num_goals'])
        self.state_dim = dataset['states'].shape[-1]
        self.control_dim = dataset['actions'].shape[-1]
        self.norm_stats = dataset.get('norm_stats')
        self.max_action = float(np.abs(dataset['actions']).max())

        self.task_data = {}
        for goal_idx in range(self.num_goals):
            goal_mask = dataset['goal_indices'] == goal_idx
            traj_indices = np.where(goal_mask)[0]
            np.random.shuffle(traj_indices)
            n_train = int(len(traj_indices) * train_holdout_split)
            self.task_data[goal_idx] = {
                'train_indices': traj_indices[:n_train],
                'holdout_indices': traj_indices[n_train:],
            }

    def get_task_ids(self):
        return list(self.task_data.keys())

    def sample_random_context(self, task_id, K, goal_state,
                               state_range=(-10, 10), vel_range=(-10, 10),
                               control_range=(-5, 5)):
        """Sample RANDOM (state, control) -> cost context matching P2P-Cost training.

        The pretrained model was trained with random context, not trajectory-based.
        Using trajectory context causes distribution mismatch and poor predictions.
        """
        Q = np.eye(4) * 1.0
        R = np.eye(2) * 0.1
        ctx_pos = np.random.uniform(state_range[0], state_range[1], size=(K, 2))
        ctx_vel = np.random.uniform(vel_range[0], vel_range[1], size=(K, 2))
        ctx_s = np.concatenate([ctx_pos, ctx_vel], axis=-1)
        ctx_c = np.random.uniform(control_range[0], control_range[1], size=(K, 2))
        state_err = ctx_s - goal_state[None, :]
        ctx_cost = (np.sum(state_err @ Q * state_err, axis=-1) +
                    np.sum(ctx_c @ R * ctx_c, axis=-1)).reshape(-1, 1)

        if self.normalize and self.norm_stats is not None:
            ctx_s = (ctx_s - self.norm_stats['state_mean']) / self.norm_stats['state_std']
            ctx_cost = (ctx_cost - self.norm_stats['cost_mean']) / self.norm_stats['cost_std']
            ctx_c = ctx_c / self.max_action
        return ctx_s, ctx_c, ctx_cost

    def _sample(self, traj_indices, K, N):
        all_trans = []
        for idx in traj_indices:
            states = self.dataset['states'][idx]
            actions = self.dataset['actions'][idx]
            costs = self.dataset['costs'][idx]
            if costs.ndim > 1:
                costs = costs.squeeze()
            horizon = len(actions)
            for t in range(horizon):
                all_trans.append({
                    'state': states[t], 'action': actions[t],
                    'cost': costs[t], 'time': t / horizon,
                })

        si = np.random.choice(len(all_trans), size=min(K, len(all_trans)),
                              replace=K > len(all_trans))
        src_s = np.array([all_trans[i]['state'] for i in si])
        src_c = np.array([all_trans[i]['action'] for i in si])
        src_co = np.array([all_trans[i]['cost'] for i in si]).reshape(-1, 1)

        ti = np.random.choice(traj_indices, size=min(N, len(traj_indices)),
                              replace=N > len(traj_indices))
        tgt_s = self.dataset['states'][ti, :-1, :]
        tgt_c = self.dataset['actions'][ti]

        if self.normalize and self.norm_stats is not None:
            src_s = (src_s - self.norm_stats['state_mean']) / self.norm_stats['state_std']
            tgt_s = (tgt_s - self.norm_stats['state_mean']) / self.norm_stats['state_std']
            src_co = (src_co - self.norm_stats['cost_mean']) / self.norm_stats['cost_std']
            src_c = src_c / self.max_action
            tgt_c = tgt_c / self.max_action

        horizon = tgt_s.shape[1]
        time_norm = np.linspace(0.0, 1.0, horizon, endpoint=False)[:, None]
        time_bc = np.repeat(time_norm[None, :, :], tgt_s.shape[0], axis=0)
        tgt_s = np.concatenate([tgt_s, time_bc], axis=-1)

        return src_s, src_c, src_co, None, tgt_s, tgt_c

    def sample_train(self, tid, K, N):
        return self._sample(self.task_data[tid]['train_indices'], K, N)

    def sample_holdout(self, tid, K, N):
        return self._sample(self.task_data[tid]['holdout_indices'], K, N)


# ── Per-task evaluation functions ───────────────────────────────────────────

def _get_context(dl, tid, dataset):
    """On-policy branch context for P2P-Cost: K (state, control) -> immediate-cost
    transitions sampled from the task's stored optimal trajectories.

    This matches how the aligned operator is trained (DoubleIntegratorLQRData, which
    draws on-policy transitions). The operator only ever sees on-policy context, so
    feeding uniform-random context (the old online-training convention) is a
    distribution shift that yields poor branch encodings and closed-loop drift.
    Context is drawn from the task's TRAIN split, leaving the holdout split for eval.
    """
    src_s, src_c, src_co, _, _, _ = dl._sample(dl.task_data[tid]['train_indices'], K, N=1)
    si = jnp.array(np.concatenate([src_s, src_c], axis=-1))
    sv = jnp.array(src_co)
    return si, sv


def eval_pretrained_per_task(model, dl, task_ids, max_action, dataset):
    task_errors = []
    for tid in task_ids:
        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            si, sv = _get_context(dl, tid, dataset)
            eb = dl.sample_holdout(tid, K, N=N_EVAL)
            pred = predict_setonet(model, si, sv, eb[4])
            errors.append(relative_l2(pred * max_action, eb[5] * max_action))
        task_errors.append(np.mean(errors))
    return np.array(task_errors)


def _eval_partial_ft_per_task(model, dl, task_ids, num_demos, grad_steps,
                               max_action, dataset, update_trunk_last=False):
    task_errors = []
    for tid in task_ids:
        m = copy.deepcopy(model)
        if update_trunk_last:
            filt = jax.tree_util.tree_map(lambda _: False, m)
            filt = eqx.tree_at(
                lambda m: (m.trunk.layers[-1], m.rho.layers[-1]), filt,
                replace=(
                    jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.trunk.layers[-1]),
                    jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.rho.layers[-1]),
                ))
        else:
            filt = jax.tree_util.tree_map(lambda _: False, m)
            filt = eqx.tree_at(
                lambda m: m.rho.layers[-1], filt,
                replace=jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.rho.layers[-1]))

        trainable, frozen = eqx.partition(m, filt)
        opt = optax.adam(FT_LR)
        opt_state = opt.init(trainable)

        for _ in range(grad_steps):
            si, sv = _get_context(dl, tid, dataset)
            b = dl.sample_train(tid, K, N=num_demos)
            ts, tc = jnp.array(b[4]), jnp.array(b[5])
            def loss_fn(tr):
                mm = eqx.combine(tr, frozen)
                p = jax.vmap(jax.vmap(mm, in_axes=(None, None, 0)),
                             in_axes=(None, None, 0))(si, sv, ts)
                return jnp.mean(jnp.square(p - tc))
            _, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
            updates, opt_state = opt.update(grads, opt_state)
            trainable = eqx.apply_updates(trainable, updates)

        final = eqx.combine(trainable, frozen)
        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            si, sv = _get_context(dl, tid, dataset)
            eb = dl.sample_holdout(tid, K, N=N_EVAL)
            pred = predict_setonet(final, si, sv, eb[4])
            errors.append(relative_l2(pred * max_action, eb[5] * max_action))
        task_errors.append(np.mean(errors))
    return np.array(task_errors)


def eval_setonet_ft_per_task(model, dl, task_ids, num_demos, grad_steps, max_action, dataset):
    """Fine-tune branch (freeze trunk). Uses random context."""
    task_errors = []
    for tid in task_ids:
        ft = copy.deepcopy(model)
        filt = jax.tree_util.tree_map(lambda x: eqx.is_array(x), ft)
        filt = eqx.tree_at(lambda m: m.trunk, filt,
                           replace=jax.tree_util.tree_map(lambda _: False, ft.trunk))
        trainable, frozen = eqx.partition(ft, filt)
        opt = optax.adam(FT_LR)
        opt_state = opt.init(trainable)

        for _ in range(grad_steps):
            si, sv = _get_context(dl, tid, dataset)
            b = dl.sample_train(tid, K, N=num_demos)
            ts, tc = jnp.array(b[4]), jnp.array(b[5])
            def loss_fn(tr):
                mm = eqx.combine(tr, frozen)
                p = jax.vmap(jax.vmap(mm, in_axes=(None, None, 0)),
                             in_axes=(None, None, 0))(si, sv, ts)
                return jnp.mean(jnp.square(p - tc))
            _, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
            updates, opt_state = opt.update(grads, opt_state)
            trainable = eqx.apply_updates(trainable, updates)

        ft = eqx.combine(trainable, frozen)
        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            si, sv = _get_context(dl, tid, dataset)
            eb = dl.sample_holdout(tid, K, N=N_EVAL)
            pred = predict_setonet(ft, si, sv, eb[4])
            errors.append(relative_l2(pred * max_action, eb[5] * max_action))
        task_errors.append(np.mean(errors))
    return np.array(task_errors)


def eval_maml_per_task(maml_model, dl, task_ids, num_demos, grad_steps,
                       inner_lr, max_action):
    task_errors = []
    for tid in task_ids:
        b = dl.sample_train(tid, K, N=num_demos)
        sup_s = jnp.array(b[4].reshape(-1, b[4].shape[-1]))
        sup_c = jnp.array(b[5].reshape(-1, b[5].shape[-1]))

        adapted = maml_model
        for _ in range(grad_steps):
            def loss_fn(m):
                pred = jax.vmap(m)(sup_s)
                return jnp.mean(jnp.square(pred - sup_c))
            _, grads = eqx.filter_value_and_grad(loss_fn)(adapted)
            updates = jax.tree_util.tree_map(lambda g: -inner_lr * g, grads)
            adapted = eqx.apply_updates(adapted, updates)

        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            eb = dl.sample_holdout(tid, K, N=N_EVAL)
            pred = predict_maml(adapted, eb[4])
            errors.append(relative_l2(pred * max_action, eb[5] * max_action))
        task_errors.append(np.mean(errors))
    return np.array(task_errors)


# ── Single environment runner ───────────────────────────────────────────────

def run_p2p_cost(config_dir, data_dir, checkpoint_dir, seed):
    """Run per-task evaluation for P2P-Cost."""
    print("\n=== P2P-Cost ===")

    cfg_path = Path(config_dir) / "p2p_cost.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    key = jr.PRNGKey(seed)
    keys = jr.split(key, 3)
    model_cfg = cfg["model"]

    # Load pretrained
    pretrained = SetONet(
        input_size_src=6, output_size_src=1,
        input_size_tgt=5, output_size_tgt=2,
        **{k: model_cfg[k] for k in ['p', 'phi_hidden_size', 'phi_output_size',
           'rho_hidden_size', 'trunk_hidden_size', 'n_phi_layers', 'n_rho_layers',
           'n_trunk_layers', 'aggregation_type', 'attention_n_heads',
           'attention_n_tokens', 'use_bias']},
        key=keys[0],
    )
    pretrained = eqx.tree_deserialise_leaves(
        str(Path(checkpoint_dir) / "p2p_cost" / "pretrained" / "setonet.eqx"), pretrained)

    # Load MAML
    maml_cfg = cfg.get("maml_model", {})
    maml_model = MAMLMLP(input_dim=5, output_dim=2,
                         hidden_size=maml_cfg.get("hidden_size", 128),
                         num_layers=maml_cfg.get("num_layers", 16), key=keys[1])
    maml_model = eqx.tree_deserialise_leaves(
        str(Path(checkpoint_dir) / "p2p_cost" / "maml" / "maml.eqx"), maml_model)

    inner_lr = cfg.get("maml", {}).get("inner_lr", 0.01)

    # Load norm stats from TRAINING HISTORY (must match what model was trained with)
    # P2P-Cost uses online normalization — dataset norms are different and will
    # cause prediction errors if used instead.
    hist = np.load(str(Path(checkpoint_dir) / "p2p_cost" / "pretrained" / "training_history.npz"))
    norm_stats = {
        'state_mean': np.array(hist['state_mean']),
        'state_std': np.array(hist['state_std']),
        'cost_mean': float(hist['cost_mean']),
        'cost_std': float(hist['cost_std']),
    }
    max_action = float(hist['max_action'])

    # Generate transfer tasks
    np.random.seed(seed)
    transfer_ds = generate_dataset(
        num_goals=NUM_TASKS, trajectories_per_goal=50,
        horizon=50, dt=0.1,
        Q_weight=1.0, R_weight=0.1, Qf_weight=10.0,
        goal_range=(-5.0, 5.0), state_range=(-10.0, 10.0),
        vel_range=(-5.0, 5.0), zero_velocity_goal=True,
        seed=seed + 1000,
    )
    transfer_ds['norm_stats'] = norm_stats

    np.random.seed(seed)
    dl = TransferTaskDataLoader(transfer_ds, train_holdout_split=0.25, normalize=True)
    dl.max_action = max_action
    task_ids = dl.get_task_ids()

    print(f"Evaluating {len(task_ids)} tasks...")

    print("  Pre-trained...")
    pretrained_errors = eval_pretrained_per_task(pretrained, dl, task_ids, max_action, transfer_ds)
    print(f"    Mean: {np.mean(pretrained_errors):.4f}")

    print(f"  SetONet-FT ({GRAD_STEPS} steps)...")
    ft_errors = eval_setonet_ft_per_task(pretrained, dl, task_ids, NUM_DEMOS, GRAD_STEPS, max_action, transfer_ds)
    print(f"    Mean: {np.mean(ft_errors):.4f}")

    print(f"  Last-Branch ({GRAD_STEPS} steps)...")
    lb_errors = _eval_partial_ft_per_task(pretrained, dl, task_ids, NUM_DEMOS, GRAD_STEPS, max_action, transfer_ds, False)
    print(f"    Mean: {np.mean(lb_errors):.4f}")

    print(f"  Last-Both ({GRAD_STEPS} steps)...")
    lboth_errors = _eval_partial_ft_per_task(pretrained, dl, task_ids, NUM_DEMOS, GRAD_STEPS, max_action, transfer_ds, True)
    print(f"    Mean: {np.mean(lboth_errors):.4f}")

    print(f"  MAML ({MAML_GRAD_STEPS} steps)...")
    maml_errors = eval_maml_per_task(maml_model, dl, task_ids, NUM_DEMOS, MAML_GRAD_STEPS, inner_lr, max_action)
    print(f"    Mean: {np.mean(maml_errors):.4f}")

    return {
        'env': 'P2P-Cost',
        'pretrained': pretrained_errors,
        'setonet_ft': ft_errors,
        'last_branch': lb_errors,
        'last_both': lboth_errors,
        'maml': maml_errors,
    }


# ── Dynamics-based per-task eval helpers ─────────────────────────────────────

def _collect_transitions(states_arr, actions_arr, traj_indices):
    all_t = []
    for idx in traj_indices:
        s = states_arr[idx]
        a = actions_arr[idx]
        for t in range(len(a)):
            all_t.append((s[t], a[t], s[t + 1]))
    return all_t


def _dynamics_eval_pretrained_per_task(model, dataset, task_configs, traj_index_fn,
                                       K, action_denorm_fn=None, raw_time=False):
    """Evaluate pretrained dynamics model per task (dynamics config)."""
    task_errors = []
    for config_idx in task_configs:
        traj_indices = traj_index_fn(config_idx)
        if len(traj_indices) == 0:
            continue
        transitions = _collect_transitions(dataset['states'], dataset['actions'], traj_indices)
        ctx_idx = np.random.choice(len(transitions), size=min(K, len(transitions)), replace=False)
        ctx_s = np.array([transitions[i][0] for i in ctx_idx])
        ctx_a = np.array([transitions[i][1] for i in ctx_idx])
        ctx_ns = np.array([transitions[i][2] for i in ctx_idx])
        src_inputs = jnp.concatenate([jnp.array(ctx_s), jnp.array(ctx_a)], axis=-1)
        src_outputs = jnp.array(ctx_ns)

        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            ei = np.random.choice(traj_indices)
            expert_s = dataset['states'][ei]
            expert_a = dataset['actions'][ei]
            H = len(expert_a)
            # Build target with time
            pred_list = []
            for t in range(H):
                tgt = jnp.concatenate([jnp.array(expert_s[t]), jnp.array([_qtime(t, H, raw_time)])])
                pred_list.append(model(src_inputs, src_outputs, tgt))
            pred = np.array(jnp.stack(pred_list))
            if action_denorm_fn:
                pred = action_denorm_fn(pred)
            errors.append(relative_l2(pred, expert_a))
        task_errors.append(np.mean(errors))
    return np.array(task_errors)


def _dynamics_eval_ft_per_task(model, dataset, task_configs, traj_index_fn,
                                K, grad_steps, num_demos, action_denorm_fn=None,
                                freeze_mode='trunk', action_mean=None, action_std=None,
                                raw_time=False):
    """Fine-tune dynamics model per task."""
    task_errors = []
    for config_idx in task_configs:
        traj_indices = traj_index_fn(config_idx)
        if len(traj_indices) == 0:
            continue

        ft = copy.deepcopy(model)
        filt = setonet_freeze_filter(ft, freeze_mode)
        trainable, frozen = eqx.partition(ft, filt)
        opt = optax.adam(FT_LR)
        opt_state = opt.init(trainable)

        transitions = _collect_transitions(dataset['states'], dataset['actions'], traj_indices)

        for _ in range(grad_steps):
            ctx_idx = np.random.choice(len(transitions), size=min(K, len(transitions)), replace=False)
            ctx_s = np.array([transitions[i][0] for i in ctx_idx])
            ctx_a = np.array([transitions[i][1] for i in ctx_idx])
            ctx_ns = np.array([transitions[i][2] for i in ctx_idx])
            si = jnp.concatenate([jnp.array(ctx_s), jnp.array(ctx_a)], axis=-1)
            so = jnp.array(ctx_ns)

            demo_idx = np.random.choice(traj_indices, size=min(num_demos, len(traj_indices)), replace=False)
            demo_s = jnp.array(dataset['states'][demo_idx, :-1, :])
            demo_a_raw = jnp.array(dataset['actions'][demo_idx])
            # Normalize demo actions to match model output space
            if action_mean is not None and action_std is not None:
                demo_a_norm = (demo_a_raw - jnp.array(action_mean)) / jnp.array(action_std)
            else:
                demo_a_norm = demo_a_raw
            H = demo_a_norm.shape[1]
            time_norm = _times_arr(H, raw_time, jnp)
            time_bc = jnp.broadcast_to(time_norm[None, :, None], (demo_s.shape[0], H, 1))
            demo_st = jnp.concatenate([demo_s, time_bc], axis=-1)

            def loss_fn(tr):
                mm = eqx.combine(tr, frozen)
                p = jax.vmap(jax.vmap(mm, in_axes=(None, None, 0)),
                             in_axes=(None, None, 0))(si, so, demo_st)
                return jnp.mean(jnp.square(p - demo_a_norm))

            _, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
            updates, opt_state = opt.update(grads, opt_state)
            trainable = eqx.apply_updates(trainable, updates)

        final = eqx.combine(trainable, frozen)

        # Eval
        ctx_idx = np.random.choice(len(transitions), size=min(K, len(transitions)), replace=False)
        ctx_s = np.array([transitions[i][0] for i in ctx_idx])
        ctx_a = np.array([transitions[i][1] for i in ctx_idx])
        ctx_ns = np.array([transitions[i][2] for i in ctx_idx])
        src_inputs = jnp.concatenate([jnp.array(ctx_s), jnp.array(ctx_a)], axis=-1)
        src_outputs = jnp.array(ctx_ns)

        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            ei = np.random.choice(traj_indices)
            expert_s = dataset['states'][ei]
            expert_a = dataset['actions'][ei]
            H = len(expert_a)
            pred_list = []
            for t in range(H):
                tgt = jnp.concatenate([jnp.array(expert_s[t]), jnp.array([_qtime(t, H, raw_time)])])
                pred_list.append(final(src_inputs, src_outputs, tgt))
            pred = np.array(jnp.stack(pred_list))
            if action_denorm_fn:
                pred = action_denorm_fn(pred)
            errors.append(relative_l2(pred, expert_a))
        task_errors.append(np.mean(errors))
    return np.array(task_errors)


def _dynamics_eval_maml_per_task(maml_model, dataset, task_configs, traj_index_fn,
                                  num_demos, grad_steps, inner_lr, action_denorm_fn=None,
                                  raw_time=False):
    """Evaluate MAML per task for dynamics environments."""
    task_errors = []
    for config_idx in task_configs:
        traj_indices = traj_index_fn(config_idx)
        if len(traj_indices) == 0:
            continue

        # Support set
        demo_idx = np.random.choice(traj_indices, size=min(num_demos, len(traj_indices)), replace=False)
        demo_s = dataset['states'][demo_idx, :-1, :]
        demo_a = dataset['actions'][demo_idx]
        H = demo_a.shape[1]
        time_norm = _times_arr(H, raw_time)[:, None]
        time_bc = np.repeat(time_norm[None, :, :], demo_s.shape[0], axis=0)
        demo_st = np.concatenate([demo_s, time_bc], axis=-1)
        sup_s = jnp.array(demo_st.reshape(-1, demo_st.shape[-1]))
        sup_c = jnp.array(demo_a.reshape(-1, demo_a.shape[-1]))

        adapted = maml_model
        for _ in range(grad_steps):
            def loss_fn(m):
                pred = jax.vmap(m)(sup_s)
                return jnp.mean(jnp.square(pred - sup_c))
            _, grads = eqx.filter_value_and_grad(loss_fn)(adapted)
            updates = jax.tree_util.tree_map(lambda g: -inner_lr * g, grads)
            adapted = eqx.apply_updates(adapted, updates)

        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            ei = np.random.choice(traj_indices)
            expert_s = dataset['states'][ei, :-1, :]
            expert_a = dataset['actions'][ei]
            H = len(expert_a)
            time_norm_e = _times_arr(H, raw_time)[:, None]
            tgt_s = jnp.array(np.concatenate([expert_s, time_norm_e], axis=-1))
            pred = np.array(jax.vmap(adapted)(tgt_s))
            if action_denorm_fn:
                pred = action_denorm_fn(pred)
            errors.append(relative_l2(pred, expert_a))
        task_errors.append(np.mean(errors))
    return np.array(task_errors)


# ── P2P-Dynamics runner ─────────────────────────────────────────────────────

def run_p2p_dynamics(config_dir, data_dir, checkpoint_dir, seed):
    """Run per-task evaluation for P2P-Dynamics."""
    print("\n=== P2P-Dynamics ===")

    cfg_path = Path(config_dir) / "p2p_dynamics.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    key = jr.PRNGKey(seed)
    keys = jr.split(key, 3)
    model_cfg = cfg["model"]

    base = SetONet(
        input_size_src=6, output_size_src=4,
        input_size_tgt=5, output_size_tgt=2,
        **{k: model_cfg[k] for k in ['p', 'phi_hidden_size', 'phi_output_size',
           'rho_hidden_size', 'trunk_hidden_size', 'n_phi_layers', 'n_rho_layers',
           'n_trunk_layers', 'aggregation_type', 'attention_n_heads',
           'attention_n_tokens', 'use_bias']},
        key=keys[0],
    )
    # P2P-Dynamics is aligned to a NormalizedSetONet (min-max state/action scaling).
    pretrained = build_dynamics_normalized(base, cfg) if model_cfg.get("use_normalization", True) else base
    pretrained = eqx.tree_deserialise_leaves(
        str(Path(checkpoint_dir) / "p2p_dynamics" / "pretrained" / "setonet.eqx"), pretrained)

    maml_cfg = cfg.get("maml_model", {})
    maml_model = MAMLMLP(input_dim=5, output_dim=2,
                         hidden_size=maml_cfg.get("hidden_size", 128),
                         num_layers=maml_cfg.get("num_layers", 16), key=keys[1])
    maml_model = eqx.tree_deserialise_leaves(
        str(Path(checkpoint_dir) / "p2p_dynamics" / "maml" / "maml.eqx"), maml_model)

    inner_lr = cfg.get("maml", {}).get("inner_lr", 0.01)

    ds = np.load(str(Path(data_dir) / "p2p_dynamics" / "trajectories.npz"), allow_pickle=True)
    dataset = {k: ds[k] for k in ds.files}

    # Test dynamics configs (last 20%)
    unique_dyn = np.unique(dataset['dynamics_indices'])
    np.random.seed(seed)
    np.random.shuffle(unique_dyn)
    n_train = int(len(unique_dyn) * 0.8)
    test_configs = list(unique_dyn[n_train:])[:NUM_TASKS]

    def traj_fn(ci):
        return np.where(dataset['dynamics_indices'] == ci)[0]

    print(f"Evaluating {len(test_configs)} tasks...")

    print("  Pre-trained...")
    pretrained_errors = _dynamics_eval_pretrained_per_task(
        pretrained, dataset, test_configs, traj_fn, K, raw_time=True)
    print(f"    Mean: {np.mean(pretrained_errors):.4f}")

    print(f"  SetONet-FT ({GRAD_STEPS} steps)...")
    ft_errors = _dynamics_eval_ft_per_task(
        pretrained, dataset, test_configs, traj_fn, K, GRAD_STEPS, NUM_DEMOS, freeze_mode='trunk', raw_time=True)
    print(f"    Mean: {np.mean(ft_errors):.4f}")

    print(f"  Last-Branch ({GRAD_STEPS} steps)...")
    lb_errors = _dynamics_eval_ft_per_task(
        pretrained, dataset, test_configs, traj_fn, K, GRAD_STEPS, NUM_DEMOS, freeze_mode='last_branch', raw_time=True)
    print(f"    Mean: {np.mean(lb_errors):.4f}")

    print(f"  Last-Both ({GRAD_STEPS} steps)...")
    lboth_errors = _dynamics_eval_ft_per_task(
        pretrained, dataset, test_configs, traj_fn, K, GRAD_STEPS, NUM_DEMOS, freeze_mode='last_both', raw_time=True)
    print(f"    Mean: {np.mean(lboth_errors):.4f}")

    print(f"  MAML ({MAML_GRAD_STEPS} steps)...")
    maml_errors = _dynamics_eval_maml_per_task(
        maml_model, dataset, test_configs, traj_fn, NUM_DEMOS, MAML_GRAD_STEPS, inner_lr, raw_time=True)
    print(f"    Mean: {np.mean(maml_errors):.4f}")

    return {
        'env': 'P2P-Dynamics',
        'pretrained': pretrained_errors,
        'setonet_ft': ft_errors,
        'last_branch': lb_errors,
        'last_both': lboth_errors,
        'maml': maml_errors,
    }


# ── Quadrotor runner ────────────────────────────────────────────────────────

def run_quadrotor(config_dir, data_dir, checkpoint_dir, seed):
    """Run per-task evaluation for Quadrotor."""
    print("\n=== Quadrotor ===")

    cfg_path = Path(config_dir) / "quadrotor.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    key = jr.PRNGKey(seed)
    keys = jr.split(key, 3)
    model_cfg = cfg["model"]

    pretrained = SetONet(
        input_size_src=8, output_size_src=6,
        input_size_tgt=7, output_size_tgt=2,
        **{k: model_cfg[k] for k in ['p', 'phi_hidden_size', 'phi_output_size',
           'rho_hidden_size', 'trunk_hidden_size', 'n_phi_layers', 'n_rho_layers',
           'n_trunk_layers', 'aggregation_type', 'attention_n_heads',
           'attention_n_tokens', 'use_bias']},
        key=keys[0],
    )
    pretrained = eqx.tree_deserialise_leaves(
        str(Path(checkpoint_dir) / "quadrotor" / "pretrained" / "setonet.eqx"), pretrained)

    maml_cfg = cfg.get("maml_model", {})
    maml_model = MAMLMLP(input_dim=7, output_dim=2,
                         hidden_size=maml_cfg.get("hidden_size", 128),
                         num_layers=maml_cfg.get("num_layers", 16), key=keys[1])
    maml_model = eqx.tree_deserialise_leaves(
        str(Path(checkpoint_dir) / "quadrotor" / "maml" / "maml.eqx"), maml_model)

    inner_lr = cfg.get("maml", {}).get("inner_lr", 0.01)

    ds = np.load(str(Path(data_dir) / "quadrotor" / "trajectories.npz"), allow_pickle=True)
    dataset = {k: ds[k] for k in ds.files}

    # Action normalization from training
    hist = np.load(str(Path(checkpoint_dir) / "quadrotor" / "pretrained" / "training_history.npz"))
    action_mean = np.array(hist['action_mean'])
    action_std = np.array(hist['action_std'])
    denorm = lambda pred: pred * action_std + action_mean

    # Test dynamics configs
    unique_dyn = np.unique(dataset['dynamics_indices'])
    split_rng = np.random.default_rng(DEFAULT_SEED)
    split_rng.shuffle(unique_dyn)
    n_train = int(len(unique_dyn) * 0.8)
    test_configs = list(unique_dyn[n_train:])[:NUM_TASKS]

    def traj_fn(ci):
        return np.where(dataset['dynamics_indices'] == ci)[0]

    print(f"Evaluating {len(test_configs)} tasks...")

    print("  Pre-trained...")
    pretrained_errors = _dynamics_eval_pretrained_per_task(
        pretrained, dataset, test_configs, traj_fn, K, action_denorm_fn=denorm)
    print(f"    Mean: {np.mean(pretrained_errors):.4f}")

    print(f"  SetONet-FT ({GRAD_STEPS} steps)...")
    ft_errors = _dynamics_eval_ft_per_task(
        pretrained, dataset, test_configs, traj_fn, K, GRAD_STEPS, NUM_DEMOS,
        action_denorm_fn=denorm, freeze_mode='trunk',
        action_mean=action_mean, action_std=action_std)
    print(f"    Mean: {np.mean(ft_errors):.4f}")

    print(f"  Last-Branch ({GRAD_STEPS} steps)...")
    lb_errors = _dynamics_eval_ft_per_task(
        pretrained, dataset, test_configs, traj_fn, K, GRAD_STEPS, NUM_DEMOS,
        action_denorm_fn=denorm, freeze_mode='last_branch',
        action_mean=action_mean, action_std=action_std)
    print(f"    Mean: {np.mean(lb_errors):.4f}")

    print(f"  Last-Both ({GRAD_STEPS} steps)...")
    lboth_errors = _dynamics_eval_ft_per_task(
        pretrained, dataset, test_configs, traj_fn, K, GRAD_STEPS, NUM_DEMOS,
        action_denorm_fn=denorm, freeze_mode='last_both',
        action_mean=action_mean, action_std=action_std)
    print(f"    Mean: {np.mean(lboth_errors):.4f}")

    print(f"  MAML ({MAML_GRAD_STEPS} steps)...")
    maml_errors = _dynamics_eval_maml_per_task(
        maml_model, dataset, test_configs, traj_fn, NUM_DEMOS, MAML_GRAD_STEPS, inner_lr,
        action_denorm_fn=denorm)
    print(f"    Mean: {np.mean(maml_errors):.4f}")

    return {
        'env': 'Quadrotor',
        'pretrained': pretrained_errors,
        'setonet_ft': ft_errors,
        'last_branch': lb_errors,
        'last_both': lboth_errors,
        'maml': maml_errors,
    }


# ── Plotting ────────────────────────────────────────────────────────────────

def make_figure(all_env_results, output_path):
    """Create 1xN scatter plot figure."""
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['axes.labelweight'] = 'bold'
    matplotlib.rcParams['axes.titleweight'] = 'bold'
    matplotlib.rcParams['font.size'] = 10

    n_envs = len(all_env_results)
    fig, axes = plt.subplots(1, n_envs, figsize=(3.5 * n_envs, 3.0))
    fig.patch.set_facecolor('white')
    if n_envs == 1:
        axes = [axes]

    methods = [
        ('Pre-trained', 'pretrained', COLORS["blue"], 's'),
        ('SetONet-FT', 'setonet_ft', COLORS["orange"], 'o'),
        ('Last-Branch', 'last_branch', COLORS["cyan"], '^'),
        ('Last-Both', 'last_both', COLORS["green"], 'D'),
    ]

    for ax, result in zip(axes, all_env_results):
        maml_errors = result['maml']
        for label, key, color, marker in methods:
            model_errors = result[key]
            ax.scatter(model_errors, maml_errors,
                       alpha=0.7, color=color, edgecolors='white', linewidth=0.5,
                       s=30, marker=marker, label=label, zorder=3)

        model_vals = np.concatenate([result[k] for _, k, _, _ in methods])
        data_x_max = np.max(model_vals) * 1.15
        y_max = np.max(maml_errors) * 1.05
        # Extend x-axis to show diagonal, but cap for envs with tight model errors
        if data_x_max < y_max * 0.15:
            # Model errors much smaller than MAML — use fixed margin (e.g. Quadrotor)
            x_max = max(data_x_max * 3, 1.5)
        else:
            x_max = max(data_x_max, y_max * 0.6)
        diag_max = max(x_max, y_max)
        ax.plot([0, diag_max], [0, diag_max], '--', color=COLORS["grey"],
                linewidth=1.0, label='$y = x$', zorder=1)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

        ax.set_xlabel('Model Relative L2', fontweight='bold')
        ax.set_title(result['env'], fontweight='bold')
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight('bold')

    axes[0].set_ylabel('MAML Relative L2', fontweight='bold')

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
               ncol=len(labels) + 1, frameon=False, prop={'weight': 'bold', 'size': 10},
               columnspacing=1.5)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Figure 6: MAML vs SetONet scatter plots")
    parser.add_argument("--config", default="configs")
    parser.add_argument("--data", default="data")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--output", default="outputs/figures/figure6.pdf")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print("=" * 60)
    print("Figure 6: MAML vs SetONet Scatter Plots")
    print("=" * 60)

    all_results = []
    for runner in [run_p2p_cost, run_p2p_dynamics, run_quadrotor]:
        result = runner(args.config, args.data, args.checkpoints, args.seed)
        all_results.append(result)

    # Save intermediate results
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in all_results:
        env_name = r['env'].lower().replace(' ', '_').replace('-', '_')
        np.savez(out_dir / f"scatter_{env_name}.npz", **r)

    make_figure(all_results, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
