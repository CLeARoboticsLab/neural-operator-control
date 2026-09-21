"""Unified adaptation evaluation script.

Evaluates all adaptation methods across a grid of (num_demos, gradient_steps, seeds)
and saves results for table generation.

Usage (matches Makefile):
    python src/evaluation/evaluate.py \
        --config configs/p2p_cost.yaml \
        --data data/p2p_cost \
        --checkpoint checkpoints/p2p_cost \
        --method setonet_ft \
        --steps 25 \
        --output outputs/results/p2p_cost_setonet_ft_25.json \
        --seed 42

Or run the full grid:
    python src/evaluation/evaluate.py \
        --config configs/p2p_cost.yaml \
        --data data/p2p_cost \
        --checkpoint checkpoints/p2p_cost \
        --output outputs/results/p2p_cost \
        --grid
"""

import argparse
import json
import copy
import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import yaml
from pathlib import Path
from itertools import product


# ── Grid parameters ──
NUM_DEMOS_LIST = [1, 5, 25]
GRAD_STEPS_LIST = [0, 1, 5, 25]
NUM_SEEDS = 5
NUM_TASKS = 20
K = 64           # source samples (context)
N_EVAL = 32      # target trajectories for evaluation
NUM_EVAL_BATCHES = 5
FT_LR = 1e-4    # Adam LR for fine-tuning methods
META_LR = 0.01   # SGD LR for MAML / SetONet-Meta

METHODS = ['setonet_ft', 'last_branch', 'last_both', 'maml', 'setonet_meta', 'setonet_meta_full']


# ── Metrics ──

def relative_l2(pred_phys, target_phys):
    """||pred - target|| / ||target|| in physical space."""
    diff = pred_phys - target_phys
    return float(np.sqrt(np.sum(diff**2)) / (np.sqrt(np.sum(target_phys**2)) + 1e-12))


# ── Prediction helpers ──

def predict_setonet(model, src_input, src_values, tgt_states):
    """SetONet prediction: branch encodes context, trunk predicts at query points."""
    pred = jax.vmap(
        jax.vmap(model, in_axes=(None, None, 0)),
        in_axes=(None, None, 0),
    )(src_input, src_values, jnp.array(tgt_states))
    return np.array(pred)


def predict_maml(model, tgt_states):
    """MAML prediction: direct state -> action."""
    pred = jax.vmap(
        jax.vmap(model, in_axes=(0,)),
        in_axes=(0,),
    )(jnp.array(tgt_states))
    return np.array(pred)


# ── Transfer Task Data Loader (P2P-Cost) ──

class TransferTaskDataLoader:
    """Splits each task's data into train (for adaptation) and holdout (for evaluation)."""

    def __init__(self, dataset, train_holdout_split=0.25, normalize=True):
        self.dataset = dataset
        self.normalize = normalize

        self.num_goals = int(dataset['num_goals'])
        self.state_dim = dataset['states'].shape[-1]
        self.control_dim = dataset['actions'].shape[-1]

        if normalize and 'norm_stats' in dataset:
            self.norm_stats = dataset['norm_stats']
        else:
            self.norm_stats = None

        self.max_action = float(np.abs(dataset['actions']).max())

        # Split by goal
        self.task_data = {}
        for goal_idx in range(self.num_goals):
            goal_mask = dataset['goal_indices'] == goal_idx
            traj_indices = np.where(goal_mask)[0]
            np.random.shuffle(traj_indices)
            n_train = int(len(traj_indices) * train_holdout_split)
            self.task_data[goal_idx] = {
                'goal_state': dataset['goal_states'][goal_idx],
                'train_indices': traj_indices[:n_train],
                'holdout_indices': traj_indices[n_train:],
            }

    def get_task_ids(self):
        return list(self.task_data.keys())

    def _sample_from_indices(self, traj_indices, K, N):
        """Sample K source transitions and N target trajectories."""
        K, N = int(K), int(N)

        # Collect transitions
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

        # Sample K source
        replace = K > len(all_trans)
        si = np.random.choice(len(all_trans), size=K, replace=replace)
        src_states = np.array([all_trans[i]['state'] for i in si])
        src_controls = np.array([all_trans[i]['action'] for i in si])
        src_costs = np.array([all_trans[i]['cost'] for i in si]).reshape(-1, 1)
        src_times = np.array([all_trans[i]['time'] for i in si]).reshape(-1, 1)

        # Sample N target trajectories
        replace_n = N > len(traj_indices)
        tgt_idx = np.random.choice(traj_indices, size=N, replace=replace_n)
        tgt_states = self.dataset['states'][tgt_idx, :-1, :]
        tgt_controls = self.dataset['actions'][tgt_idx]

        # Normalize
        if self.normalize and self.norm_stats is not None:
            src_states = (src_states - self.norm_stats['state_mean']) / self.norm_stats['state_std']
            tgt_states = (tgt_states - self.norm_stats['state_mean']) / self.norm_stats['state_std']
            src_costs = (src_costs - self.norm_stats['cost_mean']) / self.norm_stats['cost_std']
            src_controls = src_controls / self.max_action
            tgt_controls = tgt_controls / self.max_action

        # Append time to target states
        horizon = tgt_states.shape[1]
        time_norm = np.linspace(0.0, 1.0, horizon, endpoint=False)[:, None]
        time_bc = np.repeat(time_norm[None, :, :], tgt_states.shape[0], axis=0)
        tgt_states = np.concatenate([tgt_states, time_bc], axis=-1)

        return src_states, src_controls, src_costs, src_times, tgt_states, tgt_controls

    def sample_train(self, task_id, K, N):
        return self._sample_from_indices(self.task_data[task_id]['train_indices'], K, N)

    def sample_holdout(self, task_id, K, N):
        return self._sample_from_indices(self.task_data[task_id]['holdout_indices'], K, N)


# ── Adaptation methods (P2P-Cost) ──

def _eval_setonet_holdout(model, dl, task_ids, max_action, src_override=None, sv_override=None):
    """Evaluate SetONet on holdout data."""
    task_errors = []
    for tid in task_ids:
        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            if src_override is None:
                batch = dl.sample_train(tid, K, N=1)
                si = jnp.array(np.concatenate([batch[0], batch[1]], axis=-1))
                sv = jnp.array(batch[2])
            else:
                si, sv = src_override, sv_override

            eb = dl.sample_holdout(tid, K, N=N_EVAL)
            pred = predict_setonet(model, si, sv, eb[4])
            errors.append(relative_l2(pred * max_action, eb[5] * max_action))
        task_errors.append(np.mean(errors))
    return np.mean(task_errors)


def eval_setonet_ft(pretrained, dl, task_ids, num_demos, grad_steps, max_action):
    """Fine-tune all except trunk (Adam)."""
    if grad_steps == 0:
        return _eval_setonet_holdout(pretrained, dl, task_ids, max_action)

    task_errors = []
    for tid in task_ids:
        ft = copy.deepcopy(pretrained)
        filt = jax.tree_util.tree_map(lambda x: eqx.is_array(x), ft)
        filt = eqx.tree_at(lambda m: m.trunk, filt,
                           replace=jax.tree_util.tree_map(lambda _: False, ft.trunk))
        trainable, frozen = eqx.partition(ft, filt)
        opt = optax.adam(FT_LR)
        opt_state = opt.init(trainable)

        for _ in range(grad_steps):
            b = dl.sample_train(tid, K, N=num_demos)
            si = jnp.array(np.concatenate([b[0], b[1]], axis=-1))
            sv, ts, tc = jnp.array(b[2]), jnp.array(b[4]), jnp.array(b[5])

            def loss_fn(tr):
                m = eqx.combine(tr, frozen)
                p = jax.vmap(jax.vmap(m, in_axes=(None, None, 0)),
                             in_axes=(None, None, 0))(si, sv, ts)
                return jnp.mean(jnp.square(p - tc))

            _, grads = eqx.filter_value_and_grad(loss_fn)(trainable)
            updates, opt_state = opt.update(grads, opt_state)
            trainable = eqx.apply_updates(trainable, updates)

        ft = eqx.combine(trainable, frozen)
        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            sb = dl.sample_train(tid, K, N=1)
            si = jnp.array(np.concatenate([sb[0], sb[1]], axis=-1))
            sv = jnp.array(sb[2])
            eb = dl.sample_holdout(tid, K, N=N_EVAL)
            pred = predict_setonet(ft, si, sv, eb[4])
            errors.append(relative_l2(pred * max_action, eb[5] * max_action))
        task_errors.append(np.mean(errors))
    return np.mean(task_errors)


def _eval_partial_ft(pretrained, dl, task_ids, num_demos, grad_steps, max_action,
                     update_trunk_last=False):
    """Fine-tune selected last layers (Adam)."""
    if grad_steps == 0:
        return _eval_setonet_holdout(pretrained, dl, task_ids, max_action)

    task_errors = []
    for tid in task_ids:
        m = copy.deepcopy(pretrained)
        filt = jax.tree_util.tree_map(lambda _: False, m)

        if update_trunk_last:
            filt = eqx.tree_at(
                lambda m: (m.trunk.layers[-1], m.rho.layers[-1]), filt,
                replace=(
                    jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.trunk.layers[-1]),
                    jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.rho.layers[-1]),
                ))
        else:
            filt = eqx.tree_at(
                lambda m: m.rho.layers[-1], filt,
                replace=jax.tree_util.tree_map(lambda x: eqx.is_array(x), m.rho.layers[-1]))

        trainable, frozen = eqx.partition(m, filt)
        opt = optax.adam(FT_LR)
        opt_state = opt.init(trainable)

        for _ in range(grad_steps):
            b = dl.sample_train(tid, K, N=num_demos)
            si = jnp.array(np.concatenate([b[0], b[1]], axis=-1))
            sv, ts, tc = jnp.array(b[2]), jnp.array(b[4]), jnp.array(b[5])

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
            sb = dl.sample_train(tid, K, N=1)
            si = jnp.array(np.concatenate([sb[0], sb[1]], axis=-1))
            sv = jnp.array(sb[2])
            eb = dl.sample_holdout(tid, K, N=N_EVAL)
            pred = predict_setonet(final, si, sv, eb[4])
            errors.append(relative_l2(pred * max_action, eb[5] * max_action))
        task_errors.append(np.mean(errors))
    return np.mean(task_errors)


def eval_last_branch(pretrained, dl, task_ids, num_demos, grad_steps, max_action):
    return _eval_partial_ft(pretrained, dl, task_ids, num_demos, grad_steps, max_action, False)


def eval_last_both(pretrained, dl, task_ids, num_demos, grad_steps, max_action):
    return _eval_partial_ft(pretrained, dl, task_ids, num_demos, grad_steps, max_action, True)


def _maml_inner_update(model, states, controls, alpha):
    """MAML SGD inner step."""
    def loss_fn(m):
        pred = jax.vmap(m)(states)
        return jnp.mean(jnp.square(pred - controls))
    _, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates = jax.tree_util.tree_map(lambda g: -alpha * g, grads)
    return eqx.apply_updates(model, updates)


def eval_maml(maml_model, dl, task_ids, num_demos, grad_steps, inner_lr, max_action):
    """MAML with SGD inner-loop steps."""
    task_errors = []
    for tid in task_ids:
        adapted = maml_model
        if grad_steps > 0:
            b = dl.sample_train(tid, K, N=num_demos)
            sup_s = b[4].reshape(-1, b[4].shape[-1])  # flatten trajectories
            sup_c = b[5].reshape(-1, b[5].shape[-1])
            for _ in range(grad_steps):
                adapted = _maml_inner_update(adapted, jnp.array(sup_s), jnp.array(sup_c), inner_lr)

        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            eb = dl.sample_holdout(tid, K, N=N_EVAL)
            pred = predict_maml(adapted, eb[4])
            errors.append(relative_l2(pred * max_action, eb[5] * max_action))
        task_errors.append(np.mean(errors))
    return np.mean(task_errors)


def _setonet_meta_inner_update(model, batch, alpha, freeze_trunk=True):
    """SetONet-Meta SGD inner step."""
    src_s, src_c, src_co, tgt_s, tgt_c = batch
    si = jnp.concatenate([src_s, src_c], axis=-1)
    sv = src_co

    if freeze_trunk:
        filt = jax.tree_util.tree_map(lambda x: eqx.is_array(x), model)
        filt = eqx.tree_at(lambda m: m.trunk, filt,
                           replace=jax.tree_util.tree_map(lambda _: False, model.trunk))
        diff, static = eqx.partition(model, filt)

        def loss_fn(d):
            m = eqx.combine(d, static)
            if tgt_s.ndim == 3:
                ts_flat = tgt_s.reshape(-1, tgt_s.shape[-1])
                tc_flat = tgt_c.reshape(-1, tgt_c.shape[-1])
            else:
                ts_flat, tc_flat = tgt_s, tgt_c
            pred = jax.vmap(lambda q: m(si, sv, q))(ts_flat)
            return jnp.mean(jnp.square(pred - tc_flat))

        _, grads = eqx.filter_value_and_grad(loss_fn)(diff)
        updates = jax.tree_util.tree_map(lambda g: -alpha * g, grads)
        return eqx.combine(eqx.apply_updates(diff, updates), static)
    else:
        def loss_fn(m):
            if tgt_s.ndim == 3:
                ts_flat = tgt_s.reshape(-1, tgt_s.shape[-1])
                tc_flat = tgt_c.reshape(-1, tgt_c.shape[-1])
            else:
                ts_flat, tc_flat = tgt_s, tgt_c
            pred = jax.vmap(lambda q: m(si, sv, q))(ts_flat)
            return jnp.mean(jnp.square(pred - tc_flat))

        _, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates = jax.tree_util.tree_map(lambda g: -alpha * g, grads)
        return eqx.apply_updates(model, updates)


def eval_setonet_meta(meta_model, dl, task_ids, num_demos, grad_steps, inner_lr,
                      max_action, freeze_trunk=True):
    """SetONet-Meta: adapt with SGD."""
    task_errors = []
    for tid in task_ids:
        adapted = meta_model
        if grad_steps > 0:
            b = dl.sample_train(tid, K, N=num_demos)
            inner_batch = (jnp.array(b[0]), jnp.array(b[1]), jnp.array(b[2]),
                           jnp.array(b[4]), jnp.array(b[5]))
            for _ in range(grad_steps):
                adapted = _setonet_meta_inner_update(adapted, inner_batch, inner_lr, freeze_trunk)

        sb = dl.sample_train(tid, K, N=1)
        si = jnp.array(np.concatenate([sb[0], sb[1]], axis=-1))
        sv = jnp.array(sb[2])

        errors = []
        for _ in range(NUM_EVAL_BATCHES):
            eb = dl.sample_holdout(tid, K, N=N_EVAL)
            pred = predict_setonet(adapted, si, sv, eb[4])
            errors.append(relative_l2(pred * max_action, eb[5] * max_action))
        task_errors.append(np.mean(errors))
    return np.mean(task_errors)


# ── Model loading ──

def load_setonet(checkpoint_dir, cfg, key):
    """Load a SetONet model from checkpoint."""
    from src.setonet import SetONet
    model_cfg = cfg.get("model", {})
    state_dim = 4  # default for p2p_cost
    control_dim = 2

    model = SetONet(
        input_size_src=state_dim + control_dim,
        output_size_src=1,
        input_size_tgt=state_dim + 1,
        output_size_tgt=control_dim,
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
    return eqx.tree_deserialise_leaves(checkpoint_dir, model)


def load_maml(checkpoint_dir, cfg, key):
    """Load a MAML MLP model from checkpoint."""
    from src.training.maml import MAMLMLP
    model_cfg = cfg.get("maml_model", {})
    model = MAMLMLP(
        input_dim=5,  # state_dim + 1 (time)
        output_dim=2,
        hidden_size=model_cfg.get("hidden_size", 128),
        num_layers=model_cfg.get("num_layers", 16),
        key=key,
    )
    return eqx.tree_deserialise_leaves(checkpoint_dir, model)


# ── Main ──

def run_grid_p2p_cost(cfg, data_dir, checkpoint_dir, output_dir, seed=42):
    """Run full grid evaluation for P2P-Cost."""
    from src.envs.create_p2p_cost import generate_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ckpt = Path(checkpoint_dir)
    key = jr.PRNGKey(seed)
    keys = jr.split(key, 5)

    # Load norm stats from TRAINING HISTORY (must match what model was trained with)
    # P2P-Cost uses online normalization — dataset norms differ and cause errors.
    hist = np.load(str(ckpt / "pretrained" / "training_history.npz"))
    norm_stats = {
        'state_mean': np.array(hist['state_mean']),
        'state_std': np.array(hist['state_std']),
        'cost_mean': float(hist['cost_mean']),
        'cost_std': float(hist['cost_std']),
    }
    max_action = float(hist['max_action'])

    # Load models
    print("Loading models...")
    pretrained = load_setonet(ckpt / "pretrained" / "setonet.eqx", cfg, keys[0])
    maml_model = load_maml(ckpt / "maml" / "maml.eqx", cfg, keys[1])
    meta_branch = load_setonet(ckpt / "meta_branch" / "setonet.eqx", cfg, keys[2])
    meta_full = load_setonet(ckpt / "meta_full" / "setonet.eqx", cfg, keys[3])
    print("All models loaded.")

    maml_inner_lr = cfg.get("maml", {}).get("inner_lr", META_LR)
    meta_inner_lr = cfg.get("setonet_meta", {}).get("inner_lr", META_LR)

    results = {m: {nd: {gs: [] for gs in GRAD_STEPS_LIST} for nd in NUM_DEMOS_LIST} for m in METHODS}

    total = len(METHODS) * len(NUM_DEMOS_LIST) * len(GRAD_STEPS_LIST) * NUM_SEEDS
    done = 0

    print(f"\nGrid: {len(METHODS)} methods × {len(NUM_DEMOS_LIST)} demos × "
          f"{len(GRAD_STEPS_LIST)} steps × {NUM_SEEDS} seeds = {total} combos")

    for seed_i in range(NUM_SEEDS):
        s = 42 + seed_i * 100
        np.random.seed(s)
        print(f"\n{'='*60}\nSeed {s} ({seed_i+1}/{NUM_SEEDS})\n{'='*60}")

        # Generate fresh transfer tasks
        transfer_ds = generate_dataset(
            num_goals=NUM_TASKS, trajectories_per_goal=50,
            horizon=50, dt=0.1,
            Q_weight=1.0, R_weight=0.1, Qf_weight=10.0,
            goal_range=(-5.0, 5.0), state_range=(-10.0, 10.0),
            vel_range=(-5.0, 5.0), zero_velocity_goal=True,
            seed=s + 1000,
        )
        transfer_ds['norm_stats'] = norm_stats

        np.random.seed(s)
        dl = TransferTaskDataLoader(transfer_ds, train_holdout_split=0.25, normalize=True)
        dl.max_action = max_action
        task_ids = dl.get_task_ids()

        for nd, gs in product(NUM_DEMOS_LIST, GRAD_STEPS_LIST):
            for method, fn, args in [
                ('setonet_ft', eval_setonet_ft,
                 (pretrained, dl, task_ids, nd, gs, max_action)),
                ('last_branch', eval_last_branch,
                 (pretrained, dl, task_ids, nd, gs, max_action)),
                ('last_both', eval_last_both,
                 (pretrained, dl, task_ids, nd, gs, max_action)),
                ('maml', eval_maml,
                 (maml_model, dl, task_ids, nd, gs, maml_inner_lr, max_action)),
                ('setonet_meta', eval_setonet_meta,
                 (meta_branch, dl, task_ids, nd, gs, meta_inner_lr, max_action, True)),
                ('setonet_meta_full', eval_setonet_meta,
                 (meta_full, dl, task_ids, nd, gs, meta_inner_lr, max_action, False)),
            ]:
                val = fn(*args)
                results[method][nd][gs].append(val)
                done += 1
                print(f"  [{done:3d}/{total}] {method:20s} demos={nd:2d} steps={gs:2d} => {val:.6f}")

    # Save results
    save_dict = {
        'methods': np.array(METHODS),
        'num_demos_list': np.array(NUM_DEMOS_LIST),
        'grad_steps_list': np.array(GRAD_STEPS_LIST),
        'num_seeds': NUM_SEEDS,
        'num_tasks': NUM_TASKS,
        'K': K, 'N_eval': N_EVAL,
        'metric': 'relative_l2',
    }

    for method in METHODS:
        arr = np.zeros((len(NUM_DEMOS_LIST), len(GRAD_STEPS_LIST), NUM_SEEDS))
        for i, nd in enumerate(NUM_DEMOS_LIST):
            for j, gs in enumerate(GRAD_STEPS_LIST):
                arr[i, j, :] = results[method][nd][gs]
        save_dict[method] = arr

    out_file = output_path / "grid_results.npz"
    np.savez(out_file, **save_dict)
    print(f"\nResults saved to {out_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary (mean ± std)")
    print("=" * 70)
    for method in METHODS:
        arr = save_dict[method]
        print(f"\n  {method}:")
        for i, nd in enumerate(NUM_DEMOS_LIST):
            row = f"    demos={nd:2d}: "
            row += " | ".join(f"steps={gs:2d}: {np.mean(arr[i,j,:]):.4f}±{np.std(arr[i,j,:]):.4f}"
                              for j, gs in enumerate(GRAD_STEPS_LIST))
            print(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate adaptation methods")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default=None, help="Single method to evaluate")
    parser.add_argument("--steps", type=int, default=None, help="Single step count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", action="store_true", help="Run full grid evaluation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generator = cfg.get("generator")

    if args.grid:
        if generator == "p2p_cost":
            run_grid_p2p_cost(cfg, args.data, args.checkpoint, args.output, args.seed)
        elif generator == "halfcheetah":
            from src.evaluation.evaluate_halfcheetah import run_grid
            run_grid(cfg, args.data, args.checkpoint, args.output, args.seed)
        else:
            print(f"Grid evaluation for '{generator}' not yet implemented")
            sys.exit(1)
    else:
        print("Single-method evaluation not yet implemented. Use --grid for now.")
        sys.exit(1)


if __name__ == "__main__":
    main()
