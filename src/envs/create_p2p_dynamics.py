"""Generate trajectory dataset with multiple goal states and varying dynamics using LQR.

This script generates a dataset where:
- Multiple goal states are used across the dataset (random or grid-based)
- Multiple different vehicle dynamics configurations are defined
- For each configuration, multiple trajectories are generated from different starting states
- Trajectories are distributed across different goal states
- Uses Time-Varying Linear Quadratic Regulator (TVLQR/iLQR) for optimal control
- Linear dynamics: state is [x, y, vx, vy], control is [ax, ay]

The key structure:
1. Multiple goal states (random or grid-based)
2. N dynamics configurations (varying friction, max velocity, max acceleration)
3. For each dynamics configuration: M trajectories from different start states
4. Trajectories reach different goals with different dynamics and start positions

Output format:
- states: (total_trajectories, horizon+1, 4) - [x, y, vx, vy]
- actions: (total_trajectories, horizon, 2) - [ax, ay]
- start_states: (total_trajectories, 4) - different initial states
- goal_states: (num_goals, 4) - array of goal states
- goal_indices: (total_trajectories,) - which goal each trajectory targets
- dynamics_indices: (total_trajectories,) - which dynamics config each traj uses
- trajectories_per_config: (num_dynamics_configs,) - variable number of trajectories per config
- dynamics_params: List of LinearModelParams for each config
- metadata: dictionary with generation parameters
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from src.envs.dynamics_models import (
    LinearModel,
    LinearModelParams,
)

# Import trajax for optimization
from trajax import optimizers


def generate_dynamics_configurations(key, num_configs, config):
    """Generate multiple dynamics configurations with varying parameters.

    Args:
        key: JAX random key
        num_configs: Number of dynamics configurations to generate
        config: Configuration dictionary with parameter ranges

    Returns:
        List of LinearModelParams
    """
    dynamics_configs = []
    keys = jax.random.split(key, num_configs)

    for i, subkey in enumerate(keys):
        param_keys = jax.random.split(subkey, 5)

        # Sample dynamics parameters from ranges
        friction_coeff = jax.random.uniform(
            param_keys[0],
            minval=config['dynamics_ranges']['friction_range'][0],
            maxval=config['dynamics_ranges']['friction_range'][1]
        )

        max_velocity = jax.random.uniform(
            param_keys[1],
            minval=config['dynamics_ranges']['max_velocity_range'][0],
            maxval=config['dynamics_ranges']['max_velocity_range'][1]
        )

        max_acceleration = jax.random.uniform(
            param_keys[2],
            minval=config['dynamics_ranges']['max_acceleration_range'][0],
            maxval=config['dynamics_ranges']['max_acceleration_range'][1]
        )

        params = LinearModelParams(
            max_velocity_x=float(max_velocity),
            max_velocity_y=float(max_velocity),
            max_acceleration_x=float(max_acceleration),
            max_acceleration_y=float(max_acceleration),
            friction_coeff=float(friction_coeff)
        )

        dynamics_configs.append(params)

    return dynamics_configs


def compute_trajectories_per_config(num_goals, num_trajectories_per_task):
    """Compute number of trajectories for each dynamics configuration.

    Each dynamics config has trajectories for all goals, with a fixed number
    of trajectories per (dynamics, goal) task.

    Args:
        num_goals: Number of goal states
        num_trajectories_per_task: Number of trajectories per (dynamics, goal) pair

    Returns:
        Number of trajectories per dynamics config
    """
    return num_goals * num_trajectories_per_task


def generate_random_state(key, config, is_goal=False):
    """Generate a random state within workspace bounds.

    Args:
        key: JAX random key
        config: Configuration dictionary
        is_goal: If True, generate goal state (may have zero velocity)

    Returns:
        State array [x, y, vx, vy]
    """
    keys = jax.random.split(key, 4)

    # Random position within workspace
    x = jax.random.uniform(
        keys[0],
        minval=config['workspace']['x_range'][0],
        maxval=config['workspace']['x_range'][1]
    )

    y = jax.random.uniform(
        keys[1],
        minval=config['workspace']['y_range'][0],
        maxval=config['workspace']['y_range'][1]
    )

    if is_goal and config['goal_state']['zero_velocity']:
        # Goal state with zero velocity
        vx = 0.0
        vy = 0.0
    else:
        # Random velocity
        speed = jax.random.uniform(
            keys[2],
            minval=config['goal_state']['speed_range'][0] if is_goal else config['start_state']['speed_range'][0],
            maxval=config['goal_state']['speed_range'][1] if is_goal else config['start_state']['speed_range'][1]
        )

        # Random direction
        angle = jax.random.uniform(keys[3], minval=-jnp.pi, maxval=jnp.pi)

        vx = speed * jnp.cos(angle)
        vy = speed * jnp.sin(angle)

    return jnp.array([x, y, vx, vy])


def generate_random_goal_states(key, num_goals, config):
    """Generate random goal states within workspace bounds.

    Args:
        key: JAX random key
        num_goals: Number of goal states to generate
        config: Configuration dictionary

    Returns:
        Array of goal states (num_goals, 4) - [x, y, vx, vy]
    """
    keys = jax.random.split(key, num_goals)
    gen_goal = lambda k: generate_random_state(k, config, is_goal=True)
    goal_states = jax.vmap(gen_goal)(keys)
    return goal_states


def generate_grid_goal_states(config):
    """Generate goal states arranged in a grid within workspace bounds.

    Args:
        config: Configuration dictionary

    Returns:
        Array of goal states (num_goals, 4) - [x, y, vx, vy]
    """
    num_x = config['goal_state']['grid_num_x']
    num_y = config['goal_state']['grid_num_y']

    # Create grid of positions
    x_vals = jnp.linspace(
        config['workspace']['x_range'][0],
        config['workspace']['x_range'][1],
        num_x
    )
    y_vals = jnp.linspace(
        config['workspace']['y_range'][0],
        config['workspace']['y_range'][1],
        num_y
    )

    # Create meshgrid
    xx, yy = jnp.meshgrid(x_vals, y_vals)

    # Flatten to get list of positions
    x_positions = xx.flatten()
    y_positions = yy.flatten()

    # Set velocities based on config
    if config['goal_state']['zero_velocity']:
        vx = jnp.zeros_like(x_positions)
        vy = jnp.zeros_like(y_positions)
    else:
        # For grid, we'll use the mean of the speed range
        mean_speed = (config['goal_state']['speed_range'][0] +
                     config['goal_state']['speed_range'][1]) / 2.0
        vx = jnp.full_like(x_positions, mean_speed)
        vy = jnp.zeros_like(y_positions)

    # Stack into goal states
    goal_states = jnp.stack([x_positions, y_positions, vx, vy], axis=1)

    return goal_states


def create_goal_reaching_cost_functions(goal_state, cost_weights):
    """Create cost functions for goal-reaching task.

    Args:
        goal_state: Target goal state [x, y, vx, vy]
        cost_weights: Dictionary with cost weights

    Returns:
        Tuple of (stage_cost_func, terminal_cost_func)
    """
    goal_pos = goal_state[:2]
    goal_vel = goal_state[2:]

    def stage_cost(state, control, t):
        x, y, vx, vy = state
        ax, ay = control

        position = jnp.array([x, y])
        velocity = jnp.array([vx, vy])

        # Distance to goal position
        position_error = jnp.linalg.norm(position - goal_pos)

        # Velocity error
        velocity_error = jnp.linalg.norm(velocity - goal_vel)

        # Control effort
        control_effort = ax**2 + ay**2

        cost = (
            cost_weights['position_weight'] * position_error**2
            + cost_weights['velocity_weight'] * velocity_error**2
            + cost_weights['control_weight'] * control_effort
        )

        return cost

    def terminal_cost(state, t):
        x, y, vx, vy = state

        position = jnp.array([x, y])
        velocity = jnp.array([vx, vy])

        # Distance to goal
        position_error = jnp.linalg.norm(position - goal_pos)
        velocity_error = jnp.linalg.norm(velocity - goal_vel)

        cost = (
            cost_weights['terminal_position_weight'] * position_error**2
            + cost_weights['terminal_velocity_weight'] * velocity_error**2
        )

        return cost

    return stage_cost, terminal_cost


def solve_goal_reaching_trajectory(key, start_state, goal_state, vehicle_params, config):
    """Solve a single goal-reaching trajectory using iLQR.

    Args:
        key: JAX random key
        start_state: Initial state [x, y, vx, vy]
        goal_state: Goal state [x, y, vx, vy]
        vehicle_params: LinearModelParams for this trajectory
        config: Configuration dictionary

    Returns:
        Tuple of (states, actions)
    """
    # Create vehicle with specified params
    vehicle = LinearModel(vehicle_params)

    # Create dynamics function
    def dynamics_func(state, control, t):
        return vehicle.step(state, control, config['dt'])

    # Extract goal components and weights for direct use (avoid nested closures)
    goal_x, goal_y, goal_vx, goal_vy = goal_state
    pos_w = config['cost_weights']['position_weight']
    vel_w = config['cost_weights']['velocity_weight']
    ctrl_w = config['cost_weights']['control_weight']
    term_pos_w = config['cost_weights']['terminal_position_weight']
    term_vel_w = config['cost_weights']['terminal_velocity_weight']
    horizon = config['horizon']

    # Combined cost function - inline everything to avoid closure issues with JAX
    def cost_func(state, control, t):
        x, y, vx, vy = state
        ax, ay = control

        # Position and velocity errors
        pos_error_sq = (x - goal_x)**2 + (y - goal_y)**2
        vel_error_sq = (vx - goal_vx)**2 + (vy - goal_vy)**2

        # Control effort
        control_effort = ax**2 + ay**2

        # Stage cost
        stage_cost = pos_w * pos_error_sq + vel_w * vel_error_sq + ctrl_w * control_effort

        # Terminal cost (add at last timestep)
        terminal_cost = term_pos_w * pos_error_sq + term_vel_w * vel_error_sq

        # Use jnp.where to add terminal cost at final timestep
        total_cost = jnp.where(t >= horizon - 1, stage_cost + terminal_cost, stage_cost)

        return total_cost

    # Initial control guess: zero controls
    U_init = jnp.zeros((config['horizon'], 2))

    # Solve using iLQR (iterative LQR)
    result = optimizers.ilqr(
        cost_func,
        dynamics_func,
        start_state,
        U_init,
        maxiter=config['max_iterations'],
    )

    states = result[0]
    actions = result[1]
    costs = jax.vmap(cost_func)(states[:-1,:], actions, jnp.array(list(range(actions.shape[0])))[:,None])

    return states, actions, costs


def generate_dataset(
    num_dynamics_configs,
    goal_states,
    config,
    save_path=None,
    seed=0
):
    """Generate a dataset with multiple goal states and varying dynamics configurations.

    Args:
        num_dynamics_configs: Number of different dynamics configurations
        goal_states: Array of goal states (num_goals, 4)
        config: Configuration dictionary (includes trajectory count range per config)
        save_path: Path to save the dataset (optional)
        seed: Random seed

    Returns:
        Dictionary containing states, actions, start states, goal states, and metadata
    """
    key = jax.random.PRNGKey(seed)

    num_goals = len(goal_states)
    num_trajectories_per_task = config.get('num_trajectories_per_task', 2)
    trajectories_per_config = compute_trajectories_per_config(num_goals, num_trajectories_per_task)
    total_trajectories = num_dynamics_configs * trajectories_per_config

    print(f"Generating multi-goal dataset with varying dynamics:")
    print(f"  Number of goal states: {num_goals}")
    print(f"  Dynamics configurations: {num_dynamics_configs}")
    print(f"  Trajectories per (dynamics, goal) task: {num_trajectories_per_task}")
    print(f"  Trajectories per dynamics config: {trajectories_per_config} ({num_goals} goals × {num_trajectories_per_task} trajs)")
    print(f"  Total trajectories: {total_trajectories} ({num_dynamics_configs} dynamics × {trajectories_per_config} trajs)")
    print(f"  Using LQR-based optimization (iLQR)")
    print(f"  Workspace: X=[{config['workspace']['x_range'][0]}, {config['workspace']['x_range'][1]}], "
          f"Y=[{config['workspace']['y_range'][0]}, {config['workspace']['y_range'][1]}]")

    # Step 1: Display goal states
    print(f"\nGoal states ({num_goals} total):")
    for i, goal_state in enumerate(goal_states):
        print(f"  Goal {i}: position=({goal_state[0]:.2f}, {goal_state[1]:.2f}), "
              f"velocity=({goal_state[2]:.2f}, {goal_state[3]:.2f})")

    # Step 2: Generate dynamics configurations
    print("\nGenerating dynamics configurations...")
    key, subkey = jax.random.split(key)
    dynamics_configs = generate_dynamics_configurations(
        subkey, num_dynamics_configs, config
    )

    print(f"Generated {len(dynamics_configs)} dynamics configurations:")
    for i, params in enumerate(dynamics_configs[:5]):  # Show first 5
        print(f"  Config {i}: friction={params.friction_coeff:.3f}, "
              f"max_vel={params.max_velocity_x:.1f}, "
              f"max_accel={params.max_acceleration_x:.1f}")
    if len(dynamics_configs) > 5:
        print(f"  ... ({len(dynamics_configs) - 5} more configs)")

    # Step 3: All dynamics configs get the same number of trajectories
    print(f"\nEach dynamics config covers all {num_goals} goals with {num_trajectories_per_task} trajectories each")

    # Step 4: Generate start states for all trajectories
    print("\nGenerating start states...")
    key, subkey = jax.random.split(key)
    state_keys = jax.random.split(subkey, total_trajectories)

    gen_start_state = lambda k: generate_random_state(k, config, is_goal=False)
    start_states_all = jax.vmap(gen_start_state)(state_keys)
    print(f"Start states shape: {start_states_all.shape}")

    # Step 5: Create dynamics indices and goal indices (which config/goal each trajectory uses)
    # UNIFORM APPROACH: Each (dynamics, goal) pair gets EXACTLY num_trajectories_per_task trajectories
    dynamics_indices = []
    goal_indices = []

    for config_idx in range(num_dynamics_configs):
        # For this dynamics config, generate trajectories for all goals
        for goal_idx in range(num_goals):
            # Each (dynamics, goal) pair gets exactly num_trajectories_per_task trajectories
            for _ in range(num_trajectories_per_task):
                dynamics_indices.append(config_idx)
                goal_indices.append(goal_idx)

    dynamics_indices = jnp.array(dynamics_indices)
    goal_indices = jnp.array(goal_indices)
    print(f"Dynamics indices shape: {dynamics_indices.shape}")
    print(f"Goal indices shape: {goal_indices.shape}")

    # Step 6: Solve trajectories
    print("\nSolving trajectories (this may take a while)...")
    key, subkey = jax.random.split(key)
    solve_keys = jax.random.split(subkey, total_trajectories)

    all_states = []
    all_actions = []
    all_costs = []

    # Process in batches by dynamics config for efficiency
    traj_idx = 0
    for config_idx in range(num_dynamics_configs):
        num_traj = trajectories_per_config  # All configs have the same number

        print(f"  Solving {num_traj} trajectories for dynamics config {config_idx+1}/{num_dynamics_configs}...")

        vehicle_params = dynamics_configs[config_idx]

        # Get start states and goal indices for this config
        start_states_batch = start_states_all[traj_idx:traj_idx + num_traj]
        goal_indices_batch = goal_indices[traj_idx:traj_idx + num_traj]
        goal_states_ = goal_states[goal_indices_batch]

        # Solve for this dynamics configuration with different goals
        def solve_fn(k, start, goal):
            return solve_goal_reaching_trajectory(
                k, start, goal, vehicle_params, config
            )

        states_batch, actions_batch, costs_batch = jax.vmap(solve_fn)(
            solve_keys[traj_idx:traj_idx + num_traj],
            start_states_batch,
            goal_states_
        )

        all_states.append(states_batch)
        all_actions.append(actions_batch)
        all_costs.append(costs_batch)

        traj_idx += num_traj

    # Concatenate all batches
    states = jnp.concatenate(all_states, axis=0)
    actions = jnp.concatenate(all_actions, axis=0)
    costs = jnp.concatenate(all_costs, axis=0)

    print(f"\nStates shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Costs shape: {costs.shape}")

    # Create dataset
    dataset = {
        'states': np.array(states),
        'actions': np.array(actions),
        'costs': np.array(costs),
        'actions': np.array(actions),
        'start_states': np.array(start_states_all),
        'goal_states': np.array(goal_states),  # Array of goal states
        'goal_indices': np.array(goal_indices),  # Which goal each trajectory uses
        'dynamics_indices': np.array(dynamics_indices),
        'trajectories_per_config': trajectories_per_config,
        'num_trajectories_per_task': num_trajectories_per_task,
        'dynamics_params': dynamics_configs,  # List of LinearModelParams
        'config': config,
        'generation_time': datetime.now().isoformat(),
        'num_dynamics_configs': num_dynamics_configs,
        'num_goals': num_goals,
        'total_trajectories': total_trajectories,
    }

    print(f"\nSuccessfully generated {total_trajectories} trajectories")
    print(f"  {num_dynamics_configs} dynamics configs × {num_goals} goals × {num_trajectories_per_task} trajs/task")
    print(f"  Each (dynamics, goal) pair has exactly {num_trajectories_per_task} trajectories")
    print(f"  Perfect uniform distribution across all {num_dynamics_configs * num_goals} tasks")

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **dataset)
        print(f"\nDataset saved to {save_path}")
        print(f"File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")

    return dataset


def load_dataset(path):
    """Load a saved dataset.

    Args:
        path: Path to the .npz file

    Returns:
        Dictionary containing the dataset
    """
    data = np.load(path, allow_pickle=True)
    return {
        'states': data['states'],
        'actions': data['actions'],
        'costs': data['costs'],
        'start_states': data['start_states'],
        'goal_states': data['goal_states'],
        'goal_indices': data['goal_indices'],
        'dynamics_indices': data['dynamics_indices'],
        'trajectories_per_config': data['trajectories_per_config'],
        'dynamics_params': data['dynamics_params'].tolist(),
        'config': data['config'].item(),
        'generation_time': str(data['generation_time']),
        'num_dynamics_configs': int(data['num_dynamics_configs']),
        'num_goals': int(data['num_goals']),
        'total_trajectories': int(data['total_trajectories']),
    }


def config_from_yaml(data_cfg: dict) -> dict:
    """Convert the data section of a YAML config to the dict format expected by generate_dataset."""
    goal_state_config = {
        'type': data_cfg['goal_state']['type'],
        'speed_range': tuple(data_cfg['goal_state']['speed_range']),
        'zero_velocity': data_cfg['goal_state']['zero_velocity'],
    }

    if data_cfg['goal_state']['type'] == 'random':
        goal_state_config['num_goals'] = data_cfg['goal_state']['num_goals']
    elif data_cfg['goal_state']['type'] == 'grid':
        goal_state_config['grid_num_x'] = data_cfg['goal_state']['grid_num_x']
        goal_state_config['grid_num_y'] = data_cfg['goal_state']['grid_num_y']

    return {
        'horizon': data_cfg['horizon'],
        'dt': data_cfg['dt'],
        'max_iterations': data_cfg['max_iterations'],
        'workspace': {
            'x_range': tuple(data_cfg['workspace']['x_range']),
            'y_range': tuple(data_cfg['workspace']['y_range']),
        },
        'dynamics_ranges': {
            'friction_range': tuple(data_cfg['dynamics_ranges']['friction_range']),
            'max_velocity_range': tuple(data_cfg['dynamics_ranges']['max_velocity_range']),
            'max_acceleration_range': tuple(data_cfg['dynamics_ranges']['max_acceleration_range']),
        },
        'num_trajectories_per_task': data_cfg['num_trajectories_per_task'],
        'start_state': {
            'speed_range': tuple(data_cfg['start_state']['speed_range']),
        },
        'goal_state': goal_state_config,
        'cost_weights': {
            'position_weight': data_cfg['cost_weights']['position_weight'],
            'velocity_weight': data_cfg['cost_weights']['velocity_weight'],
            'control_weight': data_cfg['cost_weights']['control_weight'],
            'terminal_position_weight': data_cfg['cost_weights']['terminal_position_weight'],
            'terminal_velocity_weight': data_cfg['cost_weights']['terminal_velocity_weight'],
        }
    }


def run_from_yaml(cfg: dict, output_dir: str, seed: int) -> dict:
    """Entry point called by generate_data.py dispatcher.

    Args:
        cfg: Full YAML config dict (with 'data' key)
        output_dir: Output directory
        seed: Random seed

    Returns:
        Generated dataset dict
    """
    data_cfg = cfg['data']
    config = config_from_yaml(data_cfg)

    # Generate goal states
    goal_type = data_cfg['goal_state']['type']
    if goal_type == 'random':
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        goal_states = generate_random_goal_states(
            subkey, data_cfg['goal_state']['num_goals'], config
        )
    elif goal_type == 'grid':
        goal_states = generate_grid_goal_states(config)
    else:
        raise ValueError(f"Unknown goal state type: {goal_type}")

    save_path = str(Path(output_dir) / "trajectories.npz")

    return generate_dataset(
        num_dynamics_configs=data_cfg['num_dynamics_configs'],
        goal_states=goal_states,
        config=config,
        save_path=save_path,
        seed=seed,
    )


def main():
    """Standalone entry point using argparse."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Generate P2P-Dynamics dataset")
    parser.add_argument("--config", default="configs/p2p_dynamics.yaml", help="Path to config YAML")
    parser.add_argument("--output", default="data/p2p_dynamics", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_from_yaml(cfg, args.output, args.seed)
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()