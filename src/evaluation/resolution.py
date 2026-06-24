"""
Figure 5: Task resolution invariance across all environments.

Evaluates how model performance varies with the number of context samples (K)
or number of obstacles. Generates a 2x2 figure with one subplot per environment.

Usage (matches Makefile):
    python src/evaluation/resolution.py \
        --config configs \
        --data data \
        --checkpoints checkpoints \
        --output outputs/results/resolution

Then plot with:
    python src/plotting/plot_resolution.py \
        --results outputs/results/resolution \
        --output outputs/figures/figure5.pdf
"""

import argparse
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from pathlib import Path
from tqdm import tqdm

from src.setonet import SetONet
from src.envs.dataloader import (
    DoubleIntegratorLQRData,
    VaryingDynamicsData,
    ObstacleAvoidanceImitation,
)

DEFAULT_SEED = 42

# K values to sweep for context-based environments
K_SAMPLES_LIST = [1, 4, 8, 16, 32, 64, 128, 256]

# Obstacle counts to sweep (training used {2, 4, 6}; 3 and 5 are unseen)
OBS_COUNT_LIST = [2, 3, 4, 5, 6]

NUM_EVAL_TASKS = 50


# ── Shared helpers ──────────────────────────────────────────────────────────

def relative_l2(pred, target):
    """Relative L2 error: ||pred - target|| / ||target||, averaged over tasks."""
    diff = pred - target
    return float(np.sqrt(np.sum(diff**2)) / (np.sqrt(np.sum(target**2)) + 1e-12))


def load_setonet(checkpoint_path, model_cfg, input_size_src, output_size_src,
                 input_size_tgt, output_size_tgt, key):
    """Load a SetONet from checkpoint."""
    model = SetONet(
        input_size_src=input_size_src,
        output_size_src=output_size_src,
        input_size_tgt=input_size_tgt,
        output_size_tgt=output_size_tgt,
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


# ── P2P-Cost ────────────────────────────────────────────────────────────────

def eval_p2p_cost(config_dir, data_dir, checkpoint_dir, seed):
    """Evaluate P2P-Cost across different K values.

    Uses sample_task_batch (same as training) to generate fresh LQR tasks,
    varying only the number of source samples K.
    """
    from src.training.train_p2p_cost import sample_task_batch
    print("\n=== P2P-Cost ===")

    cfg_path = Path(config_dir) / "p2p_cost.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    key = jr.PRNGKey(seed)
    model = load_setonet(
        Path(checkpoint_dir) / "p2p_cost" / "pretrained" / "setonet.eqx",
        cfg["model"], 6, 1, 5, 2, key,
    )

    # Load norm stats from training history (matches what model was trained with)
    hist = np.load(str(Path(checkpoint_dir) / "p2p_cost" / "pretrained" / "training_history.npz"))
    norm_stats = {
        'state_mean': jnp.array(hist['state_mean']),
        'state_std': jnp.array(hist['state_std']),
        'cost_mean': float(hist['cost_mean']),
        'cost_std': float(hist['cost_std']),
        'max_action': float(hist['max_action']),
    }
    print(f"\nNorm stats (from training history):")
    print(f"  state_mean: {norm_stats['state_mean']}")
    print(f"  state_std:  {norm_stats['state_std']}")
    print(f"  max_action: {norm_stats['max_action']:.4f}")
    max_action = norm_stats['max_action']

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    dt = data_cfg["dt"]
    horizon = data_cfg["horizon"]
    Q_weight = data_cfg["cost_weights"]["Q_weight"]
    R_weight = data_cfg["cost_weights"]["R_weight"]
    Qf_weight = data_cfg["cost_weights"]["Qf_weight"]
    goal_range = tuple(train_cfg["goal_range"])
    state_range = tuple(train_cfg["state_range"])
    src_vel_range = tuple(train_cfg["src_vel_range"])

    A_jnp = jnp.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    B_jnp = jnp.array([[0,0],[0,0],[dt,0],[0,dt]])

    results = {}
    for K in tqdm(K_SAMPLES_LIST, desc="P2P-Cost K sweep"):
        errors = []
        for i in range(NUM_EVAL_TASKS):
            key, batch_key = jr.split(key)
            batch = sample_task_batch(
                batch_key, 1, K, horizon, A_jnp, B_jnp, 4, 2,
                Q_weight, R_weight, Qf_weight,
                norm_stats=norm_stats,
                goal_range=goal_range, state_range=state_range,
                src_vel_range=src_vel_range,
            )
            src_s, src_c, src_co, tgt_s, tgt_c = batch

            si = jnp.concatenate([src_s[0], src_c[0]], axis=-1)
            so = src_co[0]
            ts = tgt_s[0]
            tc = tgt_c[0]

            pred = jax.vmap(lambda t: model(si, so, t))(ts)
            pred_phys = np.array(pred) * max_action
            tgt_phys = np.array(tc) * max_action
            diff = pred_phys - tgt_phys
            rl2 = float(np.sqrt(np.sum(diff**2)) / (np.sqrt(np.sum(tgt_phys**2)) + 1e-12))
            errors.append(rl2)
        results[K] = errors

    return {
        'env': 'P2P-Cost',
        'x_values': K_SAMPLES_LIST,
        'x_label': 'K Samples',
        'results': results,
        'train_values': [32, 64, 128],
    }


# ── P2P-Dynamics ────────────────────────────────────────────────────────────

def eval_p2p_dynamics(config_dir, data_dir, checkpoint_dir, seed):
    """Evaluate P2P-Dynamics across different K values."""
    print("\n=== P2P-Dynamics ===")

    cfg_path = Path(config_dir) / "p2p_dynamics.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    key = jr.PRNGKey(seed)
    model = load_setonet(
        Path(checkpoint_dir) / "p2p_dynamics" / "pretrained" / "setonet.eqx",
        cfg["model"], 6, 4, 5, 2, key,
    )

    ds = np.load(str(Path(data_dir) / "p2p_dynamics" / "trajectories.npz"), allow_pickle=True)
    dataset = {k: ds[k] for k in ds.files}

    np.random.seed(seed)
    data_loader = VaryingDynamicsData(dataset, train_perc=0.8)

    results = {}
    for K in tqdm(K_SAMPLES_LIST, desc="P2P-Dynamics K sweep"):
        errors = []
        for _ in range(NUM_EVAL_TASKS):
            task_data = data_loader.get_task("test", K=K, N=1)
            (rand_s, rand_a, rand_ns, _, expert_traj, expert_act, _, _) = task_data

            src_inputs = jnp.concatenate([jnp.array(rand_s), jnp.array(rand_a)], axis=-1)
            src_outputs = jnp.array(rand_ns)

            # Expert trajectory: (1, H+1, state_dim+1), actions: (1, H, action_dim)
            tgt_states = jnp.array(expert_traj[0, :-1, :])  # (H, state_dim+1)
            tgt_controls = jnp.array(expert_act[0])

            pred = jax.vmap(lambda t: model(src_inputs, src_outputs, t))(tgt_states)
            errors.append(relative_l2(np.array(pred), np.array(tgt_controls)))
        results[K] = errors

    return {
        'env': 'P2P-Dynamics',
        'x_values': K_SAMPLES_LIST,
        'x_label': 'K Samples',
        'results': results,
        'train_values': [32, 64, 128],
    }


# ── Quadrotor ───────────────────────────────────────────────────────────────

def eval_quadrotor(config_dir, data_dir, checkpoint_dir, seed):
    """Evaluate Quadrotor across different K values."""
    print("\n=== Quadrotor ===")

    cfg_path = Path(config_dir) / "quadrotor.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    key = jr.PRNGKey(seed)
    model = load_setonet(
        Path(checkpoint_dir) / "quadrotor" / "pretrained" / "setonet.eqx",
        cfg["model"], 8, 6, 7, 2, key,
    )

    ds = np.load(str(Path(data_dir) / "quadrotor" / "trajectories.npz"), allow_pickle=True)
    dataset = {k: ds[k] for k in ds.files}

    # Load action normalization
    hist = np.load(str(Path(checkpoint_dir) / "quadrotor" / "pretrained" / "training_history.npz"))
    action_mean = np.array(hist['action_mean'])
    action_std = np.array(hist['action_std'])

    # Reconstruct train/test split
    unique_dyn = np.unique(dataset['dynamics_indices'])
    split_rng = np.random.default_rng(DEFAULT_SEED)
    split_rng.shuffle(unique_dyn)
    n_train = int(len(unique_dyn) * 0.8)
    train_dynamics = set(int(d) for d in unique_dyn[:n_train])
    test_dynamics = set(int(d) for d in unique_dyn[n_train:])

    test_mask = np.array([d in test_dynamics for d in dataset['dynamics_indices']])

    def _collect_transitions(traj_indices):
        all_t = []
        for idx in traj_indices:
            s = dataset['states'][idx]
            a = dataset['actions'][idx]
            for t in range(len(a)):
                all_t.append((s[t], a[t], s[t + 1]))
        return all_t

    np.random.seed(seed)
    available_test = list(test_dynamics)

    results = {}
    for K in tqdm(K_SAMPLES_LIST, desc="Quadrotor K sweep"):
        errors = []
        for _ in range(NUM_EVAL_TASKS):
            config_idx = np.random.choice(available_test)
            dyn_mask = (dataset['dynamics_indices'] == config_idx) & test_mask
            traj_indices = np.where(dyn_mask)[0]
            if len(traj_indices) == 0:
                continue

            transitions = _collect_transitions(traj_indices)
            n_ctx = min(K, len(transitions))
            ctx_idx = np.random.choice(len(transitions), size=n_ctx, replace=n_ctx > len(transitions))
            ctx_s = np.array([transitions[i][0] for i in ctx_idx])
            ctx_a = np.array([transitions[i][1] for i in ctx_idx])
            ctx_ns = np.array([transitions[i][2] for i in ctx_idx])

            src_inputs = jnp.concatenate([jnp.array(ctx_s), jnp.array(ctx_a)], axis=-1)
            src_outputs = jnp.array(ctx_ns)

            expert_idx = np.random.choice(traj_indices)
            expert_states = dataset['states'][expert_idx]
            expert_actions = dataset['actions'][expert_idx]
            H = len(expert_actions)
            horizon = H

            # Predict with normalized time
            pred_list = []
            for t in range(H):
                tgt = jnp.concatenate([jnp.array(expert_states[t]), jnp.array([t / horizon])])
                p = model(src_inputs, src_outputs, tgt)
                pred_list.append(p)
            pred_norm = jnp.stack(pred_list)

            # Denormalize
            pred_phys = np.array(pred_norm) * action_std + action_mean
            errors.append(relative_l2(pred_phys, expert_actions))
        results[K] = errors

    return {
        'env': 'Quadrotor',
        'x_values': K_SAMPLES_LIST,
        'x_label': 'K Samples',
        'results': results,
        'train_values': [32, 64, 128],
    }


# ── Obstacle Avoidance ──────────────────────────────────────────────────────

def eval_obstacle(config_dir, data_dir, checkpoint_dir, seed):
    """Evaluate Obstacle across different obstacle counts."""
    print("\n=== Obstacle Avoidance ===")

    cfg_path = Path(config_dir) / "obstacle.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    key = jr.PRNGKey(seed)
    model = load_setonet(
        Path(checkpoint_dir) / "obstacle" / "pretrained" / "setonet.eqx",
        cfg["model"], 2, 1, 5, 2, key,
    )

    expert_data = np.load(
        str(Path(data_dir) / "obstacle" / "trajectories.npy"), allow_pickle=True
    ).item()

    np.random.seed(seed)
    data_loader = ObstacleAvoidanceImitation(
        expert_data, train_perc=0.8,
        normalize=cfg.get("training", {}).get("normalize", True),
        seed=seed,
    )

    # Group test tasks by obstacle count (data has 2, 4, 6)
    test_by_count = {}
    for t in data_loader.test_data:
        n = t['obstacles'].shape[0]
        test_by_count.setdefault(n, []).append(t)

    results = {}
    for n_obs in tqdm(OBS_COUNT_LIST, desc="Obstacle count sweep"):
        errors = []

        # For unseen counts (3, 5): subsample from tasks with more obstacles
        if n_obs in test_by_count:
            # Exact match — use tasks directly
            task_pool = test_by_count[n_obs]
            subsample_obs = False
        else:
            # Subsample: pick from tasks with more obstacles, use first n_obs
            donor_counts = [c for c in sorted(test_by_count.keys()) if c > n_obs]
            if not donor_counts:
                donor_counts = [c for c in sorted(test_by_count.keys()) if c >= n_obs]
            if not donor_counts:
                results[n_obs] = []
                continue
            task_pool = test_by_count[donor_counts[0]]
            subsample_obs = True

        for _ in range(NUM_EVAL_TASKS):
            task = task_pool[np.random.choice(len(task_pool))]
            K_traj = min(10, task['states'].shape[0])

            # Sample trajectories
            indices = np.random.choice(task['states'].shape[0], size=K_traj, replace=False)
            states = jnp.array(task['states'][indices])
            states = jnp.transpose(states, (0, 2, 1))  # (K, T, state_dim)
            states = states.at[:, :, :2].set(states[:, :, :2] / data_loader.max_pos)
            states = states.at[:, :, 2:].set(states[:, :, 2:] / data_loader.max_vel)

            actions = jnp.array(task['actions'][indices])
            actions = jnp.transpose(actions, (0, 2, 1))
            actions = actions / data_loader.max_action

            obs_raw = task['obstacles']
            if subsample_obs and obs_raw.shape[0] > n_obs:
                # Randomly select n_obs obstacles from the task
                obs_idx = np.random.choice(obs_raw.shape[0], size=n_obs, replace=False)
                obs_raw = obs_raw[obs_idx]
            obstacles = jnp.array(obs_raw) / data_loader.max_pos
            obs_positions = obstacles[:, :2]
            obs_values = obstacles[:, 2:3]

            T = states.shape[1] - 1
            time_arr = jnp.arange(T + 1) / T

            # Predict at expert states
            all_pred = []
            for k in range(K_traj):
                pred_k = []
                for t in range(T):
                    tgt = jnp.concatenate([states[k, t, :], jnp.array([time_arr[t]])])
                    p = model(obs_positions, obs_values, tgt)
                    pred_k.append(p)
                all_pred.append(jnp.stack(pred_k))
            pred = jnp.stack(all_pred)

            errors.append(relative_l2(np.array(pred), np.array(actions[:, :T, :])))
        results[n_obs] = errors

    return {
        'env': 'Obstacle Avoidance',
        'x_values': OBS_COUNT_LIST,
        'x_label': 'Num Obstacles',
        'results': results,
        'train_values': [2, 4, 6],
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate task resolution invariance")
    parser.add_argument("--config", default="configs")
    parser.add_argument("--data", default="data")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--output", default="outputs/results/resolution")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for eval_fn in [eval_p2p_cost, eval_p2p_dynamics, eval_quadrotor, eval_obstacle]:
        result = eval_fn(args.config, args.data, args.checkpoints, args.seed)
        all_results.append(result)

        # Save per-environment
        env_name = result['env'].lower().replace(' ', '_').replace('-', '_')
        save_dict = {
            'x_values': np.array(result['x_values']),
            'train_values': np.array(result['train_values']),
            'x_label': result['x_label'],
            'env': result['env'],
        }
        for x_val, errors in result['results'].items():
            save_dict[f'errors_{x_val}'] = np.array(errors)
        np.savez(output_dir / f"{env_name}.npz", **save_dict)
        print(f"  Saved to {output_dir / env_name}.npz")

    print("\nAll environments evaluated.")


if __name__ == "__main__":
    main()
