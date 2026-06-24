"""
Figure 4: Operator Fitting — Expert vs SetONet predictions + rollouts.

Generates a (2*N_envs) x 3 figure:
  Columns 0-1: Control signals for Task 1 and Task 2
  Column 2:    2D state-space trajectories (merged, spans 2 rows)

Each environment occupies 2 rows (one per control dimension).

Usage (matches Makefile):
    python src/plotting/plot_fitting.py \
        --config configs \
        --data data \
        --checkpoints checkpoints \
        --output outputs/figures/figure4.pdf
"""

import argparse
import json
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.setonet import SetONet
from src.envs.dynamics_models import (
    LinearModel, LinearModelParams,
    PlanarQuadrotor, PlanarQuadrotorParams,
)

DEFAULT_SEED = 42

# ── Colors ──────────────────────────────────────────────────────────────────

EXPERT_COLOR = '#2ca02c'
PRED_COLOR = '#1f77b4'
ROLLOUT_COLOR = '#d62728'

# Per-task colors for state-space plots
TASK_EXPERT_COLORS = ['#1b7837', '#2ca02c', '#1b7837', '#2ca02c', '#1b7837']
TASK_ROLLOUT_COLORS = ['#b2182b', '#d62728', '#b2182b', '#d62728', '#b2182b']
TASK_START_COLORS = ['black', '#222222', 'black', '#222222', 'black']


# ── Prediction / rollout helpers ────────────────────────────────────────────

def predict_at_states_normalized_time(model, src_inputs, src_outputs,
                                      expert_states, horizon):
    """Predict at expert states using (state, t/horizon) target input."""
    actions = []
    for t in range(horizon):
        time_norm = t / horizon
        tgt = jnp.concatenate([expert_states[t], jnp.array([time_norm])])[None, :]
        pred = jax.vmap(model, in_axes=(None, None, 0))(src_inputs, src_outputs, tgt)
        actions.append(pred[0])
    return jnp.array(actions)


def predict_at_states_with_time(model, src_inputs, src_outputs,
                                expert_states, horizon):
    """Predict at expert states using (state, integer t) target input."""
    actions = []
    for t in range(horizon):
        tgt = jnp.concatenate([expert_states[t], jnp.array([float(t)])])[None, :]
        pred = jax.vmap(model, in_axes=(None, None, 0))(src_inputs, src_outputs, tgt)
        actions.append(pred[0])
    return jnp.array(actions)


def rollout_discrete(model, A, B, start_state, horizon,
                     src_inputs, src_outputs, make_query_fn, denorm_fn=None):
    """Rollout with discrete linear dynamics x_{t+1} = A x_t + B u_t."""
    states = [start_state]
    actions = []
    state = start_state
    for t in range(horizon):
        query = make_query_fn(state, t)
        action = model(src_inputs, src_outputs, query)
        if denorm_fn is not None:
            action = denorm_fn(action)
        actions.append(action)
        state = A @ state + B @ action
        states.append(state)
    return jnp.stack(states), jnp.stack(actions)


def rollout_continuous(model, vehicle, start_state, horizon, dt,
                       src_inputs, src_outputs, make_target_fn, denorm_fn=None):
    """Rollout with continuous dynamics (vehicle.step)."""
    states = [start_state]
    actions = []
    state = start_state
    for t in range(horizon):
        tgt = make_target_fn(state, t)
        action = model(src_inputs, src_outputs, tgt)
        if denorm_fn is not None:
            action = denorm_fn(action)
        actions.append(action)
        state = vehicle.step(state, action, dt)
        states.append(state)
    return jnp.stack(states), jnp.stack(actions)


def _collect_transitions(states_arr, actions_arr, traj_indices):
    """Collect (state, action, next_state) transitions from trajectories."""
    all_t = []
    for idx in traj_indices:
        s = states_arr[idx]
        a = actions_arr[idx]
        for t in range(len(a)):
            all_t.append((s[t], a[t], s[t + 1]))
    return all_t


def _sample_transitions(transitions, K, rng):
    """Sample K transitions returning (states, actions, next_states)."""
    idx = rng.choice(len(transitions), size=min(K, len(transitions)), replace=False)
    s = np.array([transitions[i][0] for i in idx])
    a = np.array([transitions[i][1] for i in idx])
    ns = np.array([transitions[i][2] for i in idx])
    return s, a, ns


# ── Model loading ───────────────────────────────────────────────────────────

def load_setonet(checkpoint_path, cfg, key):
    """Load a SetONet model from checkpoint using config for architecture."""
    model_cfg = cfg.get("model", {})
    # Determine dimensions from generator type
    gen = cfg.get("generator", "p2p_cost")
    if gen == "quadrotor":
        state_dim, action_dim = 6, 2
        src_out = state_dim  # dynamics-based: next_state
    elif gen == "p2p_dynamics":
        state_dim, action_dim = 4, 2
        src_out = state_dim
    else:  # p2p_cost
        state_dim, action_dim = 4, 2
        src_out = 1  # cost

    model = SetONet(
        input_size_src=state_dim + action_dim,
        output_size_src=src_out,
        input_size_tgt=state_dim + 1,
        output_size_tgt=action_dim,
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


# ── Environment loaders ─────────────────────────────────────────────────────
# Each returns: (env_label, ctrl_labels, task_results_list, env_meta)
# task_results_list is a list of dicts with keys:
#   time, expert_actions, pred_actions, rollout_actions,
#   expert_states, rollout_states, [goal_state]

def load_p2p_cost(rng, config_dir, data_dir, checkpoint_dir):
    """Load P2P-Cost (double integrator) and generate task results."""
    from trajax import tvlqr as tvlqr_lib
    print("Loading P2P-Cost...")

    cfg_path = Path(config_dir) / "p2p_cost.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    key = jr.PRNGKey(rng.integers(2**31))
    model = load_setonet(
        Path(checkpoint_dir) / "p2p_cost" / "pretrained" / "setonet.eqx",
        cfg, key,
    )

    # Load dataset
    ds = np.load(str(Path(data_dir) / "p2p_cost" / "trajectories.npz"), allow_pickle=True)
    dataset = {k: ds[k] for k in ds.files}
    if "norm_stats" in dataset:
        dataset["norm_stats"] = dataset["norm_stats"].item()

    A = dataset['A']
    B = dataset['B']
    dt = float(dataset['dt'])
    horizon = int(dataset['horizon'])
    state_dim = 4
    control_dim = 2

    Q_weight = float(dataset.get('Q_weight', 1.0))
    R_weight = float(dataset.get('R_weight', 0.1))
    Qf_weight = float(dataset.get('Qf_weight', 10.0))

    # Normalization from training history (must match what model was trained with)
    hist = np.load(str(Path(checkpoint_dir) / "p2p_cost" / "pretrained" / "training_history.npz"))
    state_mean = jnp.array(hist['state_mean'])
    state_std = jnp.array(hist['state_std'])
    cost_mean = float(hist['cost_mean'])
    cost_std = float(hist['cost_std'])
    max_action = float(hist['max_action'])

    A_jnp = jnp.array(A)
    B_jnp = jnp.array(B)
    Q_jnp = jnp.eye(state_dim) * Q_weight
    R_jnp = jnp.eye(control_dim) * R_weight
    Qf_jnp = jnp.eye(state_dim) * Qf_weight
    Q_np = np.eye(state_dim) * Q_weight
    R_np = np.eye(control_dim) * R_weight

    # tvLQR setup
    A_seq = jnp.tile(A_jnp[None, :, :], (horizon, 1, 1))
    B_seq = jnp.tile(B_jnp[None, :, :], (horizon, 1, 1))
    R_seq = jnp.tile((jnp.eye(control_dim) * R_weight)[None, :, :], (horizon, 1, 1))
    M_seq = jnp.zeros((horizon, state_dim, control_dim))
    c_seq = jnp.zeros((horizon, state_dim))
    r_seq = jnp.zeros((horizon, control_dim))

    train_cfg = cfg.get("training", {})
    state_range = tuple(train_cfg.get("state_range", [-10.0, 10.0]))
    vel_range = tuple(train_cfg.get("src_vel_range", [-5.0, 5.0]))
    K = max(train_cfg.get("K", [128]))

    def _make_query(state, t):
        state_norm = (state - state_mean) / state_std
        return jnp.concatenate([state_norm, jnp.array([t / horizon])])

    # Pick goals
    goal_states = dataset['goal_states']
    chosen_goals = rng.choice(len(goal_states), size=min(5, len(goal_states)), replace=False)

    results = []
    for goal_idx in chosen_goals:
        goal_state = goal_states[goal_idx]
        goal_jnp = jnp.array(goal_state)

        # Random context: (state, control) -> cost
        ctx_pos = rng.uniform(state_range[0], state_range[1], size=(K, 2))
        ctx_vel = rng.uniform(vel_range[0], vel_range[1], size=(K, 2))
        ctx_s = np.concatenate([ctx_pos, ctx_vel], axis=-1)
        ctx_c = rng.uniform(-5.0, 5.0, size=(K, control_dim))

        state_err = ctx_s - goal_state[None, :]
        ctx_cost = (np.sum(state_err @ Q_np * state_err, axis=-1) +
                    np.sum(ctx_c @ R_np * ctx_c, axis=-1)).reshape(-1, 1)

        ctx_s_norm = (ctx_s - np.array(state_mean)) / np.array(state_std)
        ctx_cost_norm = (ctx_cost - cost_mean) / cost_std
        ctx_c_norm = ctx_c / max_action

        src_inputs = jnp.concatenate([jnp.array(ctx_s_norm), jnp.array(ctx_c_norm)], axis=-1)
        src_outputs = jnp.array(ctx_cost_norm)

        # Solve tvLQR
        q_seq = jnp.tile((-Q_jnp @ goal_jnp)[None, :], (horizon, 1))
        qf = -Qf_jnp @ goal_jnp
        Q_seq_full = jnp.concatenate([
            jnp.tile(Q_jnp[None, :, :], (horizon, 1, 1)), Qf_jnp[None, :, :]
        ], axis=0)
        q_seq_full = jnp.concatenate([q_seq, qf[None, :]], axis=0)

        K_gains, k_gains, _, _ = tvlqr_lib.tvlqr(
            Q_seq_full, q_seq_full, R_seq, r_seq, M_seq, A_seq, B_seq, c_seq
        )

        # Expert trajectory
        start_pos = rng.uniform(state_range[0], state_range[1], size=(2,))
        start_vel = rng.uniform(vel_range[0], vel_range[1], size=(2,))
        start_state = jnp.array(np.concatenate([start_pos, start_vel]))

        expert_states, expert_actions = tvlqr_lib.rollout(
            K_gains, k_gains, start_state, A_seq, B_seq, c_seq
        )

        # Predict at expert states
        expert_states_norm = (np.array(expert_states) - np.array(state_mean)) / np.array(state_std)
        pred_actions = predict_at_states_normalized_time(
            model, src_inputs, src_outputs,
            jnp.array(expert_states_norm[:horizon]), horizon,
        )
        pred_actions = pred_actions * max_action

        # Rollout
        rollout_states, rollout_actions = rollout_discrete(
            model, A_jnp, B_jnp, start_state, horizon,
            src_inputs, src_outputs, _make_query,
            denorm_fn=lambda a: a * max_action,
        )

        results.append({
            'time': np.arange(horizon) * dt,
            'expert_actions': np.array(expert_actions),
            'pred_actions': np.array(pred_actions),
            'rollout_actions': np.array(rollout_actions),
            'expert_states': np.array(expert_states),
            'rollout_states': np.array(rollout_states),
            'goal_state': goal_state,
        })

    env_meta = {'pos_labels': ('$x$', '$y$')}
    return "P2P-Cost", ["$u_x$", "$u_y$"], results, env_meta


def load_p2p_dynamics(rng, config_dir, data_dir, checkpoint_dir):
    """Load P2P-Dynamics and generate task results."""
    print("Loading P2P-Dynamics...")

    cfg_path = Path(config_dir) / "p2p_dynamics.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    key = jr.PRNGKey(rng.integers(2**31))
    model = load_setonet(
        Path(checkpoint_dir) / "p2p_dynamics" / "pretrained" / "setonet.eqx",
        cfg, key,
    )

    ds = np.load(str(Path(data_dir) / "p2p_dynamics" / "trajectories.npz"), allow_pickle=True)
    dataset = {k: ds[k] for k in ds.files}
    config = dataset['config'].item()
    dt = config['dt']
    state_dim = dataset['states'].shape[-1]
    action_dim = dataset['actions'].shape[-1]
    K = 64

    # Split by dynamics config (use first 80% for train, rest for test)
    unique_dyn = np.unique(dataset['dynamics_indices'])
    n_train = int(len(unique_dyn) * 0.8)
    test_configs = unique_dyn[n_train:]

    chosen = rng.choice(test_configs, size=min(5, len(test_configs)), replace=False)

    results = []
    for config_idx in chosen:
        config_idx = int(config_idx)
        dyn_params = LinearModelParams(*dataset['dynamics_params'][config_idx])
        vehicle = LinearModel(dyn_params)

        task_mask = dataset['dynamics_indices'] == config_idx
        traj_indices = np.where(task_mask)[0]
        transitions = _collect_transitions(dataset['states'], dataset['actions'], traj_indices)
        ctx_s, ctx_a, ctx_ns = _sample_transitions(transitions, K, rng)
        src_inputs = jnp.concatenate([jnp.array(ctx_s), jnp.array(ctx_a)], axis=-1)
        src_outputs = jnp.array(ctx_ns)

        expert_idx = rng.choice(traj_indices)
        expert_states = dataset['states'][expert_idx]
        expert_actions = dataset['actions'][expert_idx]
        H = len(expert_actions)

        pred_actions = predict_at_states_with_time(
            model, src_inputs, src_outputs, jnp.array(expert_states[:H]), H,
        )

        def _make_tgt(state, t):
            return jnp.concatenate([state, jnp.array([float(t)])])

        rollout_states, rollout_actions = rollout_continuous(
            model, vehicle, jnp.array(expert_states[0]),
            H, dt, src_inputs, src_outputs, _make_tgt,
        )

        results.append({
            'time': np.arange(H) * dt,
            'expert_actions': np.array(expert_actions),
            'pred_actions': np.array(pred_actions),
            'rollout_actions': np.array(rollout_actions),
            'expert_states': np.array(expert_states),
            'rollout_states': np.array(rollout_states),
        })

    goal_state = dataset['goal_states'][0]
    env_meta = {'goal_state': goal_state, 'pos_labels': ('$x$', '$y$')}
    return "P2P-Dynamics", ["$u_x$", "$u_y$"], results, env_meta


def load_quadrotor(rng, config_dir, data_dir, checkpoint_dir):
    """Load Quadrotor and generate task results."""
    print("Loading Quadrotor...")

    cfg_path = Path(config_dir) / "quadrotor.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    key = jr.PRNGKey(rng.integers(2**31))
    model = load_setonet(
        Path(checkpoint_dir) / "quadrotor" / "pretrained" / "setonet.eqx",
        cfg, key,
    )

    ds = np.load(str(Path(data_dir) / "quadrotor" / "trajectories.npz"), allow_pickle=True)
    dataset = {k: ds[k] for k in ds.files}
    config = dataset['config'].item()
    dt = config['dt']
    state_dim = dataset['states'].shape[-1]
    action_dim = dataset['actions'].shape[-1]
    horizon = dataset['actions'].shape[1]
    K = 64

    # Load action normalization from training history
    hist = np.load(str(Path(checkpoint_dir) / "quadrotor" / "pretrained" / "training_history.npz"))
    action_mean = jnp.array(hist['action_mean'])
    action_std = jnp.array(hist['action_std'])

    # Reconstruct train/test split (80/20 by dynamics config)
    # Uses a fixed seed to match the split done during training
    unique_dyn = np.unique(dataset['dynamics_indices'])
    split_rng = np.random.default_rng(DEFAULT_SEED)
    split_rng.shuffle(unique_dyn)
    n_train = int(len(unique_dyn) * 0.8)
    test_dynamics = set(int(d) for d in unique_dyn[n_train:])

    denorm = lambda a: a * action_std + action_mean

    test_configs = sorted(test_dynamics)
    chosen = rng.choice(test_configs, size=min(5, len(test_configs)), replace=False)

    results = []
    for config_idx in chosen:
        config_idx = int(config_idx)
        params_arr = dataset['dynamics_params'][config_idx]
        quad_params = PlanarQuadrotorParams(
            mass=float(params_arr[0]), inertia=float(params_arr[1]),
            arm_length=float(params_arr[2]), gravity=float(params_arr[3]),
            max_thrust=float(params_arr[4]), max_torque=float(params_arr[5]),
        )
        quad = PlanarQuadrotor(quad_params)

        dyn_mask = dataset['dynamics_indices'] == config_idx
        traj_indices = np.where(dyn_mask)[0]
        transitions = _collect_transitions(dataset['states'], dataset['actions'], traj_indices)
        ctx_s, ctx_a, ctx_ns = _sample_transitions(transitions, K, rng)
        src_inputs = jnp.concatenate([jnp.array(ctx_s), jnp.array(ctx_a)], axis=-1)
        src_outputs = jnp.array(ctx_ns)

        expert_idx = rng.choice(traj_indices)
        expert_states = dataset['states'][expert_idx]
        expert_actions = dataset['actions'][expert_idx]
        H = len(expert_actions)

        pred_actions_norm = predict_at_states_normalized_time(
            model, src_inputs, src_outputs,
            jnp.array(expert_states[:H]), H,
        )
        pred_actions = jax.vmap(denorm)(pred_actions_norm)

        def _make_tgt(state, t):
            return jnp.concatenate([state, jnp.array([t / horizon])])

        rollout_states, rollout_actions = rollout_continuous(
            model, quad, jnp.array(expert_states[0]),
            H, dt, src_inputs, src_outputs, _make_tgt, denorm_fn=denorm,
        )

        results.append({
            'time': np.arange(H) * dt,
            'expert_actions': np.array(expert_actions),
            'pred_actions': np.array(pred_actions),
            'rollout_actions': np.array(rollout_actions),
            'expert_states': np.array(expert_states),
            'rollout_states': np.array(rollout_states),
        })

    goal_state = dataset['goal_states'][0]
    env_meta = {'goal_state': goal_state, 'pos_labels': ('$y$', '$z$')}
    return "Quadrotor", ["$F_z$", r"$\tau$"], results, env_meta


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_control_subplot(ax, time, expert, pred, rollout,
                         dim_idx, show_legend=False):
    """Plot one control dimension on a single axis."""
    ax.plot(time, expert[:, dim_idx], color=EXPERT_COLOR,
            linewidth=2.0, label='Expert')
    ax.plot(time[:len(pred)], np.array(pred)[:, dim_idx], color=PRED_COLOR,
            linewidth=2.0, label='SetONet')
    ax.plot(time[:len(rollout)], np.array(rollout)[:, dim_idx], color=ROLLOUT_COLOR,
            linewidth=2.0, linestyle='--', label='SetONet (rollout)')
    ax.set_facecolor('white')
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    if show_legend:
        ax.legend(loc='best', framealpha=0.9, prop={'weight': 'bold', 'size': 14})


def plot_state_space_subplot(ax, task_results, env_meta, show_legend=True):
    """Plot 2D state-space trajectories for multiple tasks."""
    for i, tr in enumerate(task_results):
        exp_s = tr['expert_states']
        roll_s = tr['rollout_states']

        exp_label = 'Expert' if i == 0 else '_nolegend_'
        roll_label = 'Rollout' if i == 0 else '_nolegend_'

        ax.plot(exp_s[:, 0], exp_s[:, 1], color=TASK_EXPERT_COLORS[i],
                linewidth=2.0, alpha=1.0, label=exp_label)
        ax.plot(roll_s[:, 0], roll_s[:, 1], color=TASK_ROLLOUT_COLORS[i],
                linewidth=2.0, linestyle='--', alpha=1.0, label=roll_label)

        # Start marker
        ax.scatter(exp_s[0, 0], exp_s[0, 1], color=TASK_START_COLORS[i],
                   s=60, zorder=5, edgecolors='white', linewidths=0.8)
        ax.annotate(f'T{i+1}', (exp_s[0, 0], exp_s[0, 1]),
                    fontsize=14, fontweight='bold', color=TASK_START_COLORS[i],
                    textcoords='offset points', xytext=(6, 6), zorder=10,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.85))

        # Per-task goal marker
        if 'goal_state' in tr:
            gs = tr['goal_state']
            ax.scatter(gs[0], gs[1], color='red', s=200, marker='*',
                       zorder=6, edgecolors='darkred', linewidths=1.0)

    # Shared goal marker (dynamics/quadrotor)
    if 'goal_state' in env_meta:
        gs = env_meta['goal_state']
        ax.scatter(gs[0], gs[1], color='red', s=200, marker='*',
                   zorder=6, edgecolors='darkred', linewidths=1.0)

    ax.set_facecolor('white')
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    if show_legend:
        ax.legend(loc='best', framealpha=0.9, prop={'weight': 'bold', 'size': 14})


def make_figure(all_env_results, output_path):
    """Assemble the multi-environment figure.

    Layout: (2*N_envs) rows x 3 columns.
    Columns 0-1: control signals for Task 1 and Task 2.
    Column 2: merged state-space subplot spanning 2 rows per env.
    """
    n_envs = len(all_env_results)
    n_rows = n_envs * 2

    fig = plt.figure(figsize=(14, 2.5 * n_rows))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(n_rows, 3, figure=fig, width_ratios=[1, 1, 1])

    for env_i, (env_label, ctrl_labels, task_results, env_meta) in enumerate(all_env_results):
        # Control subplots (columns 0-1): first 2 tasks
        for task_j, tr in enumerate(task_results[:2]):
            for dim_k in range(2):
                row = env_i * 2 + dim_k
                ax = fig.add_subplot(gs[row, task_j])
                plot_control_subplot(
                    ax, tr['time'], tr['expert_actions'],
                    tr['pred_actions'], tr['rollout_actions'],
                    dim_idx=dim_k,
                    show_legend=(env_i == 0 and task_j == 0 and dim_k == 0),
                )

        # State-space subplot (column 2, spans 2 rows)
        ax_ss = fig.add_subplot(gs[env_i * 2: env_i * 2 + 2, 2])
        plot_state_space_subplot(
            ax_ss, task_results, env_meta,
            show_legend=(env_i == 0),
        )

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate operator fitting figure (Figure 4)")
    parser.add_argument("--config", default="configs", help="Config directory")
    parser.add_argument("--data", default="data", help="Data directory")
    parser.add_argument("--checkpoints", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--output", default="outputs/figures/figure4.pdf", help="Output path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("Figure 4: Operator Fitting")
    print("=" * 60)

    all_results = []
    for loader in [load_p2p_cost, load_p2p_dynamics, load_quadrotor]:
        result = loader(rng, args.config, args.data, args.checkpoints)
        all_results.append(result)

    make_figure(all_results, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
