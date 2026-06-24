"""
Figure 8: OOD goal evaluation for Quadrotor with expert-demo adaptation.

Training goal: [0, 1.5, 0, 0, 0, 0] (hover at z=1.5)
OOD goal:      [0.5, 2.0, 0, 0, 0, 0] (shifted hover)

Evaluates 7 adaptation methods on 5 dynamics configs x 20 trajectories each.
Produces relative L2 bar plot data + rollout trajectories for the two
trajectory panels (SetONet-FT and SetONet-Meta-Full).

Usage (matches Makefile):
    python src/evaluation/evaluate_ood.py \
        --config configs/quadrotor.yaml \
        --data data/quadrotor \
        --checkpoints checkpoints/quadrotor \
        --output outputs/results/quadrotor_ood
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
from pathlib import Path

from src.setonet import SetONet
from src.training.maml import MAMLMLP
from src.envs.dynamics_models import PlanarQuadrotor, PlanarQuadrotorParams
from src.envs.create_quadrotor import generate_dataset as generate_quad_dataset

# ── Parameters ──
GRAD_STEPS = 25
FT_LR = 1e-4
META_INNER_LR = 0.01
NUM_TASKS = 5
NUM_TRAJ_PER_TASK = 20
K = 64
NUM_ADAPT = 4
NUM_EVAL = 4
NUM_EVAL_BATCHES = 5
SEED = 42

OOD_GOAL = np.array([[0.5, 2.0, 0.0, 0.0, 0.0, 0.0]])
HORIZON = 100
DT_SIM = 0.02


# ── Data loader ──

class QuadrotorTransferDataLoader:
    """Splits each task's OOD data into train (adapt) and holdout (eval)."""

    def __init__(self, dataset, test_dynamics_indices, train_holdout_split,
                 action_mean, action_std):
        self.dataset = dataset
        self.action_mean = action_mean
        self.action_std = action_std

        self.task_train_indices = {}
        self.task_holdout_indices = {}

        for dyn_idx in test_dynamics_indices:
            mask = dataset['dynamics_indices'] == dyn_idx
            traj_indices = np.where(mask)[0]
            np.random.shuffle(traj_indices)
            n_train = int(len(traj_indices) * train_holdout_split)
            self.task_train_indices[dyn_idx] = traj_indices[:n_train]
            self.task_holdout_indices[dyn_idx] = traj_indices[n_train:]

    def get_task_ids(self):
        return list(self.task_train_indices.keys())

    def _collect_transitions(self, traj_indices):
        states = self.dataset['states']
        actions = self.dataset['actions']
        transitions = []
        for idx in traj_indices:
            s, a = states[idx], actions[idx]
            for t in range(len(a)):
                transitions.append((s[t], a[t], s[t + 1]))
        return transitions

    def _sample(self, traj_indices, K, N):
        transitions = self._collect_transitions(traj_indices)
        total = len(transitions)
        si = np.random.choice(total, size=K, replace=K > total)
        src_s = np.array([transitions[i][0] for i in si])
        src_a = np.array([transitions[i][1] for i in si])
        src_ns = np.array([transitions[i][2] for i in si])

        sampled = np.random.choice(traj_indices, size=N, replace=N > len(traj_indices))
        states = self.dataset['states']
        actions = self.dataset['actions']

        expert_traj = states[sampled, :-1, :]  # (N, horizon, 6) — drop terminal
        horizon = expert_traj.shape[1]
        time_norm = np.linspace(0.0, 1.0, horizon)[:, None]
        time_bc = np.repeat(time_norm[None, :, :], N, axis=0)
        tgt_states = np.concatenate([expert_traj, time_bc], axis=-1)

        tgt_controls = (actions[sampled] - self.action_mean) / self.action_std

        return src_s, src_a, src_ns, tgt_states, tgt_controls

    def sample_train(self, task_id, K, N):
        return self._sample(self.task_train_indices[task_id], K, N)

    def sample_holdout(self, task_id, K, N):
        return self._sample(self.task_holdout_indices[task_id], K, N)


# ── Model loading ──

def load_setonet(checkpoint_path, cfg, key):
    model_cfg = cfg.get("model", {})
    model = SetONet(
        input_size_src=8, output_size_src=6,
        input_size_tgt=7, output_size_tgt=2,
        p=model_cfg.get("p", 128),
        phi_hidden_size=model_cfg.get("phi_hidden_size", 128),
        phi_output_size=model_cfg.get("phi_output_size", 128),
        rho_hidden_size=model_cfg.get("rho_hidden_size", 128),
        trunk_hidden_size=model_cfg.get("trunk_hidden_size", 128),
        n_phi_layers=model_cfg.get("n_phi_layers", 2),
        n_rho_layers=model_cfg.get("n_rho_layers", 2),
        n_trunk_layers=model_cfg.get("n_trunk_layers", 2),
        aggregation_type=model_cfg.get("aggregation_type", "attention"),
        attention_n_heads=model_cfg.get("attention_n_heads", 4),
        attention_n_tokens=model_cfg.get("attention_n_tokens", 4),
        use_bias=model_cfg.get("use_bias", True),
        key=key,
    )
    return eqx.tree_deserialise_leaves(str(checkpoint_path), model)


def load_maml(checkpoint_path, cfg, key):
    model_cfg = cfg.get("maml_model", {})
    model = MAMLMLP(
        input_dim=7, output_dim=2,
        hidden_size=model_cfg.get("hidden_size", 128),
        num_layers=model_cfg.get("num_layers", 16),
        key=key,
    )
    return eqx.tree_deserialise_leaves(str(checkpoint_path), model)


# ── Prediction / loss helpers ──

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


def setonet_loss(model, si, sv, ts, tc):
    pred = jax.vmap(
        jax.vmap(model, in_axes=(None, None, 0)),
        in_axes=(None, None, 0),
    )(si, sv, ts)
    return jnp.mean(jnp.square(pred - tc))


def relative_l2(pred, target):
    diff = pred - target
    return float(np.sqrt(np.sum(diff**2)) / (np.sqrt(np.sum(target**2)) + 1e-12))


def clip_grads(grads, max_norm=1.0):
    leaves = jax.tree_util.tree_leaves(grads)
    total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-8))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


# ── Adaptation methods ──

def _make_batch(dl, task_id, N):
    """Prepare JAX arrays from data loader sample."""
    src_s, src_a, src_ns, tgt_s, tgt_c = dl.sample_train(task_id, K, N=N)
    si = jnp.array(np.concatenate([src_s, src_a], axis=-1))
    sv = jnp.array(src_ns)
    ts = jnp.array(tgt_s)
    tc = jnp.array(tgt_c)
    return si, sv, ts, tc


def adapt_setonet_ft(model, dl, task_id, grad_steps, lr):
    """Fine-tune branch params (trunk frozen) with Adam."""
    ft = copy.deepcopy(model)
    if grad_steps == 0:
        return ft
    filt = jax.tree_util.tree_map(lambda x: eqx.is_array(x), ft)
    filt = eqx.tree_at(lambda m: m.trunk, filt,
                       replace=jax.tree_util.tree_map(lambda _: False, ft.trunk))
    trainable, frozen = eqx.partition(ft, filt)
    opt = optax.adam(lr)
    opt_state = opt.init(trainable)
    for _ in range(grad_steps):
        si, sv, ts, tc = _make_batch(dl, task_id, NUM_ADAPT)
        def loss_fn(tr):
            m = eqx.combine(tr, frozen)
            return setonet_loss(m, si, sv, ts, tc)
        _, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
        updates, opt_state = opt.update(grads, opt_state)
        trainable = eqx.apply_updates(trainable, updates)
    return eqx.combine(trainable, frozen)


def adapt_setonet_ft_all(model, dl, task_id, grad_steps, lr):
    """Fine-tune ALL parameters with Adam."""
    ft = copy.deepcopy(model)
    if grad_steps == 0:
        return ft
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(ft, eqx.is_array))
    for _ in range(grad_steps):
        si, sv, ts, tc = _make_batch(dl, task_id, NUM_ADAPT)
        def loss_fn(m):
            return setonet_loss(m, si, sv, ts, tc)
        _, grads = eqx.filter_value_and_grad(loss_fn)(ft)
        updates, opt_state = opt.update(grads, opt_state)
        ft = eqx.apply_updates(ft, updates)
    return ft


def adapt_last_branch(model, dl, task_id, grad_steps, lr):
    """Fine-tune only rho.layers[-1] with Adam."""
    m = copy.deepcopy(model)
    if grad_steps == 0:
        return m
    filt = jax.tree_util.tree_map(lambda _: False, m)
    filt = eqx.tree_at(lambda m: m.rho.layers[-1], filt,
                       replace=jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.rho.layers[-1]))
    trainable, frozen = eqx.partition(m, filt)
    opt = optax.adam(lr)
    opt_state = opt.init(trainable)
    for _ in range(grad_steps):
        si, sv, ts, tc = _make_batch(dl, task_id, NUM_ADAPT)
        def loss_fn(tr):
            mm = eqx.combine(tr, frozen)
            return setonet_loss(mm, si, sv, ts, tc)
        _, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
        updates, opt_state = opt.update(grads, opt_state)
        trainable = eqx.apply_updates(trainable, updates)
    return eqx.combine(trainable, frozen)


def adapt_last_both(model, dl, task_id, grad_steps, lr):
    """Fine-tune trunk.layers[-1] + rho.layers[-1] with Adam."""
    m = copy.deepcopy(model)
    if grad_steps == 0:
        return m
    filt = jax.tree_util.tree_map(lambda _: False, m)
    filt = eqx.tree_at(
        lambda m: (m.trunk.layers[-1], m.rho.layers[-1]), filt,
        replace=(
            jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.trunk.layers[-1]),
            jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.rho.layers[-1]),
        ))
    trainable, frozen = eqx.partition(m, filt)
    opt = optax.adam(lr)
    opt_state = opt.init(trainable)
    for _ in range(grad_steps):
        si, sv, ts, tc = _make_batch(dl, task_id, NUM_ADAPT)
        def loss_fn(tr):
            mm = eqx.combine(tr, frozen)
            return setonet_loss(mm, si, sv, ts, tc)
        _, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
        updates, opt_state = opt.update(grads, opt_state)
        trainable = eqx.apply_updates(trainable, updates)
    return eqx.combine(trainable, frozen)


def adapt_maml(model, dl, task_id, grad_steps, inner_lr):
    """MAML SGD adaptation with gradient clipping."""
    adapted = model
    if grad_steps == 0:
        return adapted
    batch = dl.sample_train(task_id, K, N=NUM_ADAPT)
    _, _, _, tgt_s, tgt_c = batch
    support_inputs = jnp.array(tgt_s.reshape(-1, tgt_s.shape[-1]))
    support_actions = jnp.array(tgt_c.reshape(-1, tgt_c.shape[-1]))
    for _ in range(grad_steps):
        def loss_fn(m):
            pred = jax.vmap(m)(support_inputs)
            return jnp.mean(jnp.square(pred - support_actions))
        _, grads = eqx.filter_value_and_grad(loss_fn)(adapted)
        grads = clip_grads(grads, max_norm=1.0)
        updates = jax.tree_util.tree_map(lambda g: -inner_lr * g, grads)
        adapted = eqx.apply_updates(adapted, updates)
    return adapted


def adapt_setonet_meta(model, dl, task_id, grad_steps, inner_lr, freeze_trunk=True):
    """SetONet-Meta SGD adaptation with gradient clipping."""
    adapted = model
    if grad_steps == 0:
        return adapted
    si, sv, ts, tc = _make_batch(dl, task_id, NUM_ADAPT)

    if freeze_trunk:
        filt = jax.tree_util.tree_map(lambda x: eqx.is_array(x), adapted)
        filt = eqx.tree_at(lambda m: m.trunk, filt,
                           replace=jax.tree_util.tree_map(lambda _: False, adapted.trunk))
        diff, static = eqx.partition(adapted, filt)
        for _ in range(grad_steps):
            def loss_fn(dp):
                m = eqx.combine(dp, static)
                return setonet_loss(m, si, sv, ts, tc)
            _, grads = eqx.filter_value_and_grad(loss_fn)(diff)
            grads = clip_grads(grads, max_norm=1.0)
            updates = jax.tree_util.tree_map(lambda g: -inner_lr * g, grads)
            diff = eqx.apply_updates(diff, updates)
        return eqx.combine(diff, static)
    else:
        for _ in range(grad_steps):
            def loss_fn(m):
                return setonet_loss(m, si, sv, ts, tc)
            _, grads = eqx.filter_value_and_grad(loss_fn)(adapted)
            grads = clip_grads(grads, max_norm=1.0)
            updates = jax.tree_util.tree_map(lambda g: -inner_lr * g, grads)
            adapted = eqx.apply_updates(adapted, updates)
        return adapted


# ── Rollout ──

def rollout_quad(model, src_input, src_values, start_state, dynamics,
                 action_mean, action_std, is_maml=False):
    """Roll out policy through quadrotor dynamics."""
    states = np.zeros((HORIZON + 1, 6))
    states[0] = start_state
    state = jnp.array(start_state)
    for t in range(HORIZON):
        time_norm = t / HORIZON
        query = jnp.concatenate([state, jnp.array([time_norm])])
        if is_maml:
            action_norm = model(query)
        else:
            action_norm = model(src_input, src_values, query)
        action = np.array(action_norm) * action_std + action_mean
        state = dynamics.step(state, jnp.array(action), DT_SIM)
        states[t + 1] = np.array(state)
    return states


# ── Evaluation ──

def eval_on_holdout(model, dl, task_id, action_mean, action_std, is_maml=False):
    """Evaluate adapted model on holdout data, return mean relative L2."""
    errors = []
    for _ in range(NUM_EVAL_BATCHES):
        if not is_maml:
            train_batch = dl.sample_train(task_id, K, N=1)
            src_s, src_a, src_ns, _, _ = train_batch
            src_input = jnp.array(np.concatenate([src_s, src_a], axis=-1))
            src_values = jnp.array(src_ns)

        eval_batch = dl.sample_holdout(task_id, K, N=NUM_EVAL)
        _, _, _, tgt_s, tgt_c = eval_batch

        if is_maml:
            pred_norm = predict_maml(model, tgt_s)
        else:
            pred_norm = predict_setonet(model, src_input, src_values, tgt_s)

        pred_phys = pred_norm * action_std + action_mean
        tgt_phys = tgt_c * action_std + action_mean
        errors.append(relative_l2(pred_phys, tgt_phys))
    return np.mean(errors)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="OOD quadrotor evaluation (Figure 8)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt = Path(args.checkpoints)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load action normalization from training history
    hist = np.load(str(ckpt / "pretrained" / "training_history.npz"))
    action_mean = hist['action_mean']
    action_std = hist['action_std']
    print(f"Action norm: mean={action_mean}, std={action_std}")

    # Load models
    print("Loading models...")
    key = jr.PRNGKey(0)
    keys = jr.split(key, 5)

    pretrained = load_setonet(ckpt / "pretrained" / "setonet.eqx", cfg, keys[0])
    maml_model = load_maml(ckpt / "maml" / "maml.eqx", cfg, keys[1])
    meta_branch = load_setonet(ckpt / "meta_branch" / "setonet.eqx", cfg, keys[2])
    meta_full = load_setonet(ckpt / "meta_full" / "setonet.eqx", cfg, keys[3])
    print("All models loaded.")

    # Generate OOD transfer dataset
    print(f"\nGenerating OOD dataset with goal={OOD_GOAL[0].tolist()}")
    print(f"  (Training goal was [0, 1.5, 0, 0, 0, 0])")
    np.random.seed(args.seed)
    gen_config = {
        'num_trajectories_per_task': NUM_TRAJ_PER_TASK,
        'horizon': HORIZON,
        'dt': DT_SIM,
        'max_iterations': cfg['data'].get('max_iterations', 150),
        'dynamics_ranges': cfg['data']['dynamics_ranges'],
        'start_state': cfg['data']['start_state'],
        'cost_weights': cfg['data']['cost_weights'],
        'workspace': cfg['data']['workspace'],
    }
    transfer_dataset = generate_quad_dataset(
        num_dynamics_configs=NUM_TASKS,
        goal_states=jnp.array(OOD_GOAL),
        config=gen_config,
        save_path=None,
        seed=args.seed + 2000,
    )

    unique_dynamics = np.unique(transfer_dataset['dynamics_indices'])
    np.random.seed(args.seed)
    dl = QuadrotorTransferDataLoader(
        transfer_dataset,
        test_dynamics_indices=list(unique_dynamics),
        train_holdout_split=0.5,
        action_mean=action_mean,
        action_std=action_std,
    )
    task_ids = dl.get_task_ids()
    print(f"  {len(task_ids)} tasks, {NUM_TRAJ_PER_TASK} trajs/task")

    # ── Evaluate all methods ──
    all_methods = ['SetONet', 'SetONet-FT', 'Full-Branch', 'Last-Branch',
                   'Last-Both', 'SetONet-Meta', 'SetONet-Meta-Full']

    bar_results = {m: [] for m in all_methods}

    # Store rollout trajectories for SetONet-FT and SetONet-Meta-Full
    rollout_data = {
        'SetONet-FT': {'trajectories': [], 'start_states': []},
        'SetONet-Meta-Full': {'trajectories': [], 'start_states': []},
    }

    print(f"\nEvaluating {len(task_ids)} tasks, {GRAD_STEPS} gradient steps")
    for task_id in task_ids:
        # SetONet (no adaptation)
        err = eval_on_holdout(pretrained, dl, task_id, action_mean, action_std)
        bar_results['SetONet'].append(err)

        # SetONet-FT (all params, Adam)
        ft_all = adapt_setonet_ft_all(pretrained, dl, task_id, GRAD_STEPS, FT_LR)
        err = eval_on_holdout(ft_all, dl, task_id, action_mean, action_std)
        bar_results['SetONet-FT'].append(err)

        # Full-Branch (trunk frozen, Adam)
        fb = adapt_setonet_ft(pretrained, dl, task_id, GRAD_STEPS, FT_LR)
        err = eval_on_holdout(fb, dl, task_id, action_mean, action_std)
        bar_results['Full-Branch'].append(err)

        # Last-Branch
        lb = adapt_last_branch(pretrained, dl, task_id, GRAD_STEPS, FT_LR)
        err = eval_on_holdout(lb, dl, task_id, action_mean, action_std)
        bar_results['Last-Branch'].append(err)

        # Last-Both
        lboth = adapt_last_both(pretrained, dl, task_id, GRAD_STEPS, FT_LR)
        err = eval_on_holdout(lboth, dl, task_id, action_mean, action_std)
        bar_results['Last-Both'].append(err)

        # SetONet-Meta (branch SGD, trunk frozen)
        meta_adapted = adapt_setonet_meta(meta_branch, dl, task_id, GRAD_STEPS,
                                          META_INNER_LR, freeze_trunk=True)
        err = eval_on_holdout(meta_adapted, dl, task_id, action_mean, action_std)
        bar_results['SetONet-Meta'].append(err)

        # SetONet-Meta-Full (all params SGD)
        mf_adapted = adapt_setonet_meta(meta_full, dl, task_id, GRAD_STEPS,
                                        META_INNER_LR, freeze_trunk=False)
        err = eval_on_holdout(mf_adapted, dl, task_id, action_mean, action_std)
        bar_results['SetONet-Meta-Full'].append(err)

        # ── Rollout trajectories for trajectory panels ──
        task_params_arr = transfer_dataset['dynamics_params'][task_id]
        quad_params = PlanarQuadrotorParams(
            mass=float(task_params_arr[0]), inertia=float(task_params_arr[1]),
            arm_length=float(task_params_arr[2]), gravity=float(task_params_arr[3]),
            max_thrust=float(task_params_arr[4]), max_torque=float(task_params_arr[5]),
        )
        task_dynamics = PlanarQuadrotor(quad_params)

        # Get context for rollout
        src_batch = dl.sample_train(task_id, K, N=1)
        src_s, src_a, src_ns, _, _ = src_batch
        src_input = jnp.array(np.concatenate([src_s, src_a], axis=-1))
        src_values = jnp.array(src_ns)

        holdout_indices = dl.task_holdout_indices[task_id]
        num_rollouts = min(4, len(holdout_indices))

        for ri in range(num_rollouts):
            start = transfer_dataset['states'][holdout_indices[ri], 0]
            for method_name, adapted_model in [('SetONet-FT', ft_all),
                                                ('SetONet-Meta-Full', mf_adapted)]:
                traj = rollout_quad(adapted_model, src_input, src_values,
                                   start, task_dynamics, action_mean, action_std)
                rollout_data[method_name]['trajectories'].append(traj)
                rollout_data[method_name]['start_states'].append(start.copy())

        print(f"  Task {task_id}: SetONet={bar_results['SetONet'][-1]:.4f}"
              f"  FT={bar_results['SetONet-FT'][-1]:.4f}"
              f"  MetaFull={bar_results['SetONet-Meta-Full'][-1]:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"OOD Goal Quadrotor Results (goal={OOD_GOAL[0].tolist()})")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Rel L2 (mean)':>14} {'± std':>10}")
    print(f"{'-'*44}")
    for m in all_methods:
        d = np.array(bar_results[m])
        print(f"{m:<20} {np.mean(d):>14.4f} {np.std(d):>10.4f}")

    # ── Save ──
    save_dict = {
        'methods': np.array(all_methods),
        'ood_goal': OOD_GOAL,
        'training_goal': np.array([[0.0, 1.5, 0.0, 0.0, 0.0, 0.0]]),
        'grad_steps': GRAD_STEPS,
    }
    for m in all_methods:
        save_dict[m.replace('-', '_')] = np.array(bar_results[m])

    for method_name in ['SetONet-FT', 'SetONet-Meta-Full']:
        key_name = method_name.replace('-', '_')
        save_dict[f'{key_name}_trajectories'] = np.array(
            rollout_data[method_name]['trajectories'])
        save_dict[f'{key_name}_start_states'] = np.array(
            rollout_data[method_name]['start_states'])

    np.savez(output_dir / "results.npz", **save_dict)
    print(f"\nResults saved to {output_dir / 'results.npz'}")


if __name__ == "__main__":
    main()
