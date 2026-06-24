"""Generate trajectory dataset for double integrator with LQR control.

This script generates a dataset where:
- Multiple goal states define different tasks
- Each task uses LQR to compute optimal controls
- Trajectories are generated from random starting states to each goal
- Immediate costs are computed as c(x,u) = (x-x_goal)^T Q (x-x_goal) + u^T R u

The data format matches what train_setonet_lqr.py expects:
- Source samples: (state, control) pairs with immediate costs
- Target samples: states with optimal LQR controls

Output format:
- states: (total_trajectories, horizon+1, 4) - [x, y, vx, vy]
- actions: (total_trajectories, horizon, 2) - [ax, ay]
- costs: (total_trajectories, horizon) - immediate costs at each timestep
- goal_states: (num_goals, 4) - array of goal states
- goal_indices: (total_trajectories,) - which goal each trajectory targets
- lqr_gains: (num_goals, horizon, control_dim, state_dim) - LQR feedback gains K
- lqr_offsets: (num_goals, horizon, control_dim) - LQR feedforward terms k
- Q_weight, R_weight, Qf_weight: LQR cost weights
- norm_stats: normalization statistics for states and costs
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

from trajax import tvlqr


def solve_lqr(
    A: np.ndarray,
    B: np.ndarray,
    goal: np.ndarray,
    horizon: int,
    Q_weight: float,
    R_weight: float,
    Qf_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve LQR problem and return gains and value function matrices.

    Args:
        A: State transition matrix (state_dim, state_dim)
        B: Control input matrix (state_dim, control_dim)
        goal: Goal state (state_dim,)
        horizon: Planning horizon
        Q_weight: State cost weight
        R_weight: Control cost weight
        Qf_weight: Terminal state cost weight

    Returns:
        K: Feedback gains (horizon, control_dim, state_dim)
        k: Feedforward terms (horizon, control_dim)
        P: Value function matrices (horizon+1, state_dim, state_dim)
        p: Value function vectors (horizon+1, state_dim)
    """
    A_jnp = jnp.array(A)
    B_jnp = jnp.array(B)
    goal_jnp = jnp.array(goal)

    state_dim = A_jnp.shape[0]
    control_dim = B_jnp.shape[1]

    # Create cost matrices
    Q = jnp.eye(state_dim) * Q_weight
    R = jnp.eye(control_dim) * R_weight
    Qf = jnp.eye(state_dim) * Qf_weight

    # Time-varying matrices (constant in this case)
    A_seq = jnp.tile(A_jnp[None, :, :], (horizon, 1, 1))
    B_seq = jnp.tile(B_jnp[None, :, :], (horizon, 1, 1))
    Q_seq = jnp.tile(Q[None, :, :], (horizon, 1, 1))
    R_seq = jnp.tile(R[None, :, :], (horizon, 1, 1))
    M_seq = jnp.zeros((horizon, state_dim, control_dim))
    c_seq = jnp.zeros((horizon, state_dim))

    # Linear terms to encode goal seeking
    q_seq = -Q @ goal_jnp
    q_seq = jnp.tile(q_seq[None, :], (horizon, 1))
    r_seq = jnp.zeros((horizon, control_dim))
    qf = -Qf @ goal_jnp

    # Full sequences for tvlqr
    Q_seq_full = jnp.concatenate([Q_seq, Qf[None, :, :]], axis=0)
    q_seq_full = jnp.concatenate([q_seq, qf[None, :]], axis=0)

    # Solve LQR
    K, k, P, p = tvlqr.tvlqr(
        Q_seq_full, q_seq_full, R_seq, r_seq, M_seq,
        A_seq, B_seq, c_seq
    )

    return np.array(K), np.array(k), np.array(P), np.array(p)


def rollout_lqr(
    start_state: np.ndarray,
    goal_state: np.ndarray,
    K: np.ndarray,
    k: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rollout LQR controller from a starting state.

    Args:
        start_state: Initial state (state_dim,)
        goal_state: Goal state (state_dim,)
        K: Feedback gains (horizon, control_dim, state_dim)
        k: Feedforward terms (horizon, control_dim)
        A: State transition matrix
        B: Control input matrix
        horizon: Planning horizon

    Returns:
        states: Trajectory states (horizon+1, state_dim)
        actions: Trajectory actions (horizon, control_dim)
    """
    A_jnp = jnp.array(A)
    B_jnp = jnp.array(B)
    K_jnp = jnp.array(K)
    k_jnp = jnp.array(k)
    start_jnp = jnp.array(start_state)

    # Build sequences for rollout
    A_seq = jnp.tile(A_jnp[None, :, :], (horizon, 1, 1))
    B_seq = jnp.tile(B_jnp[None, :, :], (horizon, 1, 1))
    c_seq = jnp.zeros((horizon, A_jnp.shape[0]))

    # Rollout using tvlqr
    X, U = tvlqr.rollout(K_jnp, k_jnp, start_jnp, A_seq, B_seq, c_seq)

    return np.array(X), np.array(U)


def compute_immediate_cost(
    state: np.ndarray,
    control: np.ndarray,
    goal_state: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> float:
    """Compute immediate (stage) cost at a state with control.

    c(x, u) = (x - x_goal)^T Q (x - x_goal) + u^T R u

    Args:
        state: Current state (state_dim,)
        control: Control action (control_dim,)
        goal_state: Goal state (state_dim,)
        Q: State cost matrix (state_dim, state_dim)
        R: Control cost matrix (control_dim, control_dim)

    Returns:
        Immediate cost value
    """
    state_error = state - goal_state
    state_cost = state_error @ Q @ state_error
    control_cost = control @ R @ control
    return state_cost + control_cost


def compute_trajectory_costs(
    states: np.ndarray,
    actions: np.ndarray,
    goal_state: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Compute immediate costs along a trajectory.

    Args:
        states: Trajectory states (horizon+1, state_dim)
        actions: Trajectory actions (horizon, control_dim)
        goal_state: Goal state (state_dim,)
        Q: State cost matrix
        R: Control cost matrix

    Returns:
        costs: Immediate costs at each timestep (horizon,)
    """
    horizon = actions.shape[0]
    costs = np.zeros(horizon)

    for t in range(horizon):
        costs[t] = compute_immediate_cost(states[t], actions[t], goal_state, Q, R)

    return costs


def generate_goal_states(
    key: jax.Array,
    num_goals: int,
    goal_range: Tuple[float, float],
    zero_velocity: bool = True,
) -> np.ndarray:
    """Generate random goal states.

    Args:
        key: JAX random key
        num_goals: Number of goal states to generate
        goal_range: (min, max) range for goal positions
        zero_velocity: If True, goals have zero velocity

    Returns:
        goal_states: Array of goal states (num_goals, 4)
    """
    key1, key2 = jr.split(key)

    # Random positions
    positions = jr.uniform(key1, (num_goals, 2), minval=goal_range[0], maxval=goal_range[1])

    if zero_velocity:
        velocities = jnp.zeros((num_goals, 2))
    else:
        velocities = jr.uniform(key2, (num_goals, 2), minval=-1.0, maxval=1.0)

    goal_states = jnp.concatenate([positions, velocities], axis=-1)
    return np.array(goal_states)


def generate_start_states(
    key: jax.Array,
    num_starts: int,
    state_range: Tuple[float, float],
    vel_range: Tuple[float, float],
) -> np.ndarray:
    """Generate random starting states.

    Args:
        key: JAX random key
        num_starts: Number of starting states to generate
        state_range: (min, max) range for positions
        vel_range: (min, max) range for velocities

    Returns:
        start_states: Array of starting states (num_starts, 4)
    """
    key1, key2 = jr.split(key)

    positions = jr.uniform(key1, (num_starts, 2), minval=state_range[0], maxval=state_range[1])
    velocities = jr.uniform(key2, (num_starts, 2), minval=vel_range[0], maxval=vel_range[1])

    start_states = jnp.concatenate([positions, velocities], axis=-1)
    return np.array(start_states)


def compute_normalization_stats(
    states: np.ndarray,
    costs: np.ndarray,
) -> Dict:
    """Compute normalization statistics for states and costs.

    Args:
        states: All trajectory states (N, horizon+1, 4)
        costs: All trajectory costs (N, horizon)

    Returns:
        Dictionary with normalization statistics
    """
    # Flatten for statistics
    states_flat = states.reshape(-1, states.shape[-1])
    costs_flat = costs.flatten()

    return {
        'state_mean': np.mean(states_flat, axis=0),
        'state_std': np.std(states_flat, axis=0) + 1e-8,
        'cost_mean': np.mean(costs_flat),
        'cost_std': np.std(costs_flat) + 1e-8,
    }


def generate_dataset(
    num_goals: int,
    trajectories_per_goal: int,
    horizon: int,
    dt: float,
    Q_weight: float,
    R_weight: float,
    Qf_weight: float,
    goal_range: Tuple[float, float],
    state_range: Tuple[float, float],
    vel_range: Tuple[float, float],
    zero_velocity_goal: bool,
    seed: int,
    save_path: str = None,
) -> Dict:
    """Generate the complete dataset.

    Args:
        num_goals: Number of goal states (tasks)
        trajectories_per_goal: Number of trajectories per goal
        horizon: Planning horizon
        dt: Time step
        Q_weight: State cost weight
        R_weight: Control cost weight
        Qf_weight: Terminal state cost weight
        goal_range: (min, max) for goal positions
        state_range: (min, max) for starting positions
        vel_range: (min, max) for starting velocities
        zero_velocity_goal: If True, goals have zero velocity
        seed: Random seed
        save_path: Path to save the dataset (optional)

    Returns:
        Dictionary containing the dataset
    """
    key = jr.PRNGKey(seed)

    # Double integrator dynamics
    state_dim = 4
    control_dim = 2

    A = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    B = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [dt, 0.0],
        [0.0, dt],
    ])

    # Cost matrices
    Q = np.eye(state_dim) * Q_weight
    R = np.eye(control_dim) * R_weight

    total_trajectories = num_goals * trajectories_per_goal

    print(f"Generating double integrator LQR dataset:")
    print(f"  Number of goals (tasks): {num_goals}")
    print(f"  Trajectories per goal: {trajectories_per_goal}")
    print(f"  Total trajectories: {total_trajectories}")
    print(f"  Horizon: {horizon}")
    print(f"  dt: {dt}")
    print(f"  Q_weight: {Q_weight}, R_weight: {R_weight}, Qf_weight: {Qf_weight}")
    print(f"  Goal range: {goal_range}")
    print(f"  State range: {state_range}")
    print(f"  Velocity range: {vel_range}")

    # Generate goal states
    key, subkey = jr.split(key)
    goal_states = generate_goal_states(subkey, num_goals, goal_range, zero_velocity=zero_velocity_goal)
    print(f"\nGenerated {num_goals} goal states")

    # Solve LQR for each goal (compute gains once per goal)
    print("\nSolving LQR for each goal...")
    all_K = []
    all_k = []

    for goal_idx in range(num_goals):
        goal = goal_states[goal_idx]
        K, k, P, p = solve_lqr(A, B, goal, horizon, Q_weight, R_weight, Qf_weight)
        all_K.append(K)
        all_k.append(k)

    lqr_gains = np.stack(all_K, axis=0)      # (num_goals, horizon, control_dim, state_dim)
    lqr_offsets = np.stack(all_k, axis=0)    # (num_goals, horizon, control_dim)

    # Generate trajectories
    print("\nGenerating trajectories...")
    all_states = []
    all_actions = []
    all_costs = []
    all_goal_indices = []

    key, subkey = jr.split(key)
    start_keys = jr.split(subkey, total_trajectories)

    traj_idx = 0
    for goal_idx in range(num_goals):
        goal = goal_states[goal_idx]
        K = all_K[goal_idx]
        k = all_k[goal_idx]

        for _ in range(trajectories_per_goal):
            # Generate random start state
            start_state = generate_start_states(
                start_keys[traj_idx], 1, state_range, vel_range
            )[0]

            # Rollout LQR
            states, actions = rollout_lqr(start_state, goal, K, k, A, B, horizon)

            # Compute costs
            costs = compute_trajectory_costs(states, actions, goal, Q, R)

            all_states.append(states)
            all_actions.append(actions)
            all_costs.append(costs)
            all_goal_indices.append(goal_idx)

            traj_idx += 1

        if (goal_idx + 1) % 10 == 0:
            print(f"  Processed {goal_idx + 1}/{num_goals} goals")

    # Stack into arrays
    states = np.stack(all_states, axis=0)       # (total_traj, horizon+1, state_dim)
    actions = np.stack(all_actions, axis=0)     # (total_traj, horizon, control_dim)
    costs = np.stack(all_costs, axis=0)         # (total_traj, horizon)
    goal_indices = np.array(all_goal_indices)   # (total_traj,)

    print(f"\nStates shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Costs shape: {costs.shape}")

    # Compute normalization statistics
    print("\nComputing normalization statistics...")
    norm_stats = compute_normalization_stats(states, costs)
    print(f"  State mean: {norm_stats['state_mean']}")
    print(f"  State std: {norm_stats['state_std']}")
    print(f"  Cost mean: {norm_stats['cost_mean']:.4f}")
    print(f"  Cost std: {norm_stats['cost_std']:.4f}")

    # Create dataset dictionary
    dataset = {
        'states': states,
        'actions': actions,
        'costs': costs,
        'goal_states': goal_states,
        'goal_indices': goal_indices,
        'lqr_gains': lqr_gains,
        'lqr_offsets': lqr_offsets,
        'Q_weight': Q_weight,
        'R_weight': R_weight,
        'Qf_weight': Qf_weight,
        'norm_stats': norm_stats,
        'A': A,
        'B': B,
        'dt': dt,
        'horizon': horizon,
        'num_goals': num_goals,
        'trajectories_per_goal': trajectories_per_goal,
        'total_trajectories': total_trajectories,
        'goal_range': goal_range,
        'state_range': state_range,
        'vel_range': vel_range,
        'seed': seed,
        'generation_time': datetime.now().isoformat(),
    }

    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Position X: [{states[:, :, 0].min():.2f}, {states[:, :, 0].max():.2f}]")
    print(f"Position Y: [{states[:, :, 1].min():.2f}, {states[:, :, 1].max():.2f}]")
    print(f"Velocity X: [{states[:, :, 2].min():.2f}, {states[:, :, 2].max():.2f}]")
    print(f"Velocity Y: [{states[:, :, 3].min():.2f}, {states[:, :, 3].max():.2f}]")
    print(f"Action X: [{actions[:, :, 0].min():.2f}, {actions[:, :, 0].max():.2f}]")
    print(f"Action Y: [{actions[:, :, 1].min():.2f}, {actions[:, :, 1].max():.2f}]")
    print(f"Costs: [{costs.min():.4f}, {costs.max():.4f}], mean={costs.mean():.4f}")

    # Final distance to goal
    final_positions = states[:, -1, :2]
    goal_positions = goal_states[goal_indices, :2]
    final_distances = np.linalg.norm(final_positions - goal_positions, axis=1)
    print(f"\nFinal distance to goal:")
    print(f"  Mean: {final_distances.mean():.4f}")
    print(f"  Max: {final_distances.max():.4f}")
    print(f"  Min: {final_distances.min():.4f}")

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **dataset)
        print(f"\nDataset saved to {save_path}")
        print(f"File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")

    return dataset


def load_dataset(path: str) -> Dict:
    """Load a saved dataset.

    Args:
        path: Path to the .npz file

    Returns:
        Dictionary containing the dataset
    """
    data = np.load(path, allow_pickle=True)

    dataset = {
        'states': data['states'],
        'actions': data['actions'],
        'costs': data['costs'],
        'goal_states': data['goal_states'],
        'goal_indices': data['goal_indices'],
        'lqr_gains': data['lqr_gains'],
        'lqr_offsets': data['lqr_offsets'],
        'Q_weight': float(data['Q_weight']),
        'R_weight': float(data['R_weight']),
        'Qf_weight': float(data['Qf_weight']),
        'norm_stats': data['norm_stats'].item(),
        'A': data['A'],
        'B': data['B'],
        'dt': float(data['dt']),
        'horizon': int(data['horizon']),
        'num_goals': int(data['num_goals']),
        'trajectories_per_goal': int(data['trajectories_per_goal']),
        'total_trajectories': int(data['total_trajectories']),
        'goal_range': tuple(data['goal_range']),
        'state_range': tuple(data['state_range']),
        'vel_range': tuple(data['vel_range']),
        'seed': int(data['seed']),
        'generation_time': str(data['generation_time']),
    }

    return dataset


def main():
    """Standalone entry point using argparse (for direct invocation)."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Generate P2P-Cost dataset")
    parser.add_argument("--config", default="configs/p2p_cost.yaml", help="Path to config YAML")
    parser.add_argument("--output", default="data/p2p_cost/trajectories.npz", help="Output path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    generate_dataset(
        num_goals=data_cfg["num_goals"],
        trajectories_per_goal=data_cfg["trajectories_per_goal"],
        horizon=data_cfg["horizon"],
        dt=data_cfg["dt"],
        Q_weight=data_cfg["cost_weights"]["Q_weight"],
        R_weight=data_cfg["cost_weights"]["R_weight"],
        Qf_weight=data_cfg["cost_weights"]["Qf_weight"],
        goal_range=tuple(data_cfg["workspace"]["goal_range"]),
        state_range=tuple(data_cfg["workspace"]["state_range"]),
        vel_range=tuple(data_cfg["workspace"]["vel_range"]),
        zero_velocity_goal=data_cfg["goal_state"]["zero_velocity"],
        seed=args.seed,
        save_path=args.output,
    )

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
