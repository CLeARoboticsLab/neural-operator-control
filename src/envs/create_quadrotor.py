"""Generate trajectory dataset for planar quadrotor with varying dynamics.

This script generates expert trajectories for a planar (2D) quadrotor system
with multi-task learning in mind. Tasks vary by:
- Quadrotor mass
- Moment of inertia
- Goal positions (hover targets)

The planar quadrotor model:
- State: [y, z, φ, ẏ, ż, φ̇] (6D) - horizontal pos, altitude, roll, velocities
- Control: [f, τ] (2D) - total thrust and torque

Dynamics:
    m * ÿ = f * sin(φ)
    m * z̈ = f * cos(φ) - m * g
    Ixx * φ̈ = τ

Inspired by the multi-task quadrotor experiments in:
    Ammar et al. "Online Multi-Task Learning for Policy Gradient Methods" (ICML 2014)
    https://proceedings.mlr.press/v32/ammar14.html

Output format:
- states: (total_trajectories, horizon+1, 6) - [y, z, φ, ẏ, ż, φ̇]
- actions: (total_trajectories, horizon, 2) - [f, τ]
- costs: (total_trajectories, horizon) - stage costs
- goal_states: (num_goals, 6) - target states
- goal_indices: (total_trajectories,) - which goal each trajectory targets
- dynamics_indices: (total_trajectories,) - which dynamics config
- dynamics_params: List of PlanarQuadrotorParams
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List

from src.envs.dynamics_models import (
    PlanarQuadrotor,
    PlanarQuadrotorParams,
)

from trajax import optimizers


def generate_dynamics_configurations(key, num_configs, config):
    """Generate multiple quadrotor dynamics configurations with varying parameters.

    Args:
        key: JAX random key
        num_configs: Number of dynamics configurations to generate
        config: Configuration dictionary with parameter ranges

    Returns:
        List of PlanarQuadrotorParams
    """
    dynamics_configs = []
    keys = jax.random.split(key, num_configs)

    for i, subkey in enumerate(keys):
        param_keys = jax.random.split(subkey, 3)

        # Sample dynamics parameters from ranges
        mass = jax.random.uniform(
            param_keys[0],
            minval=config['dynamics_ranges']['mass_range'][0],
            maxval=config['dynamics_ranges']['mass_range'][1]
        )

        # Inertia scales roughly with mass (assuming similar geometry)
        # I = m * r^2 where r is characteristic length
        inertia_scale = jax.random.uniform(
            param_keys[1],
            minval=config['dynamics_ranges']['inertia_scale_range'][0],
            maxval=config['dynamics_ranges']['inertia_scale_range'][1]
        )
        # Base inertia for 0.18 kg quadrotor is ~0.00025, so scale proportionally
        base_inertia = 0.00025
        base_mass = 0.18
        inertia = base_inertia * (float(mass) / base_mass) * float(inertia_scale)

        arm_length = jax.random.uniform(
            param_keys[2],
            minval=config['dynamics_ranges']['arm_length_range'][0],
            maxval=config['dynamics_ranges']['arm_length_range'][1]
        )

        # Max thrust scales with mass (need more thrust for heavier quad)
        max_thrust = 2.5 * float(mass) * 9.81  # 2.5x hover thrust

        params = PlanarQuadrotorParams(
            mass=float(mass),
            inertia=float(inertia),
            arm_length=float(arm_length),
            gravity=9.81,
            max_thrust=float(max_thrust),
            max_torque=config['dynamics_ranges']['max_torque'],
        )

        dynamics_configs.append(params)

    return dynamics_configs


def generate_random_state(key, config, is_goal=False):
    """Generate a random state within workspace bounds.

    Args:
        key: JAX random key
        config: Configuration dictionary
        is_goal: If True, generate goal state (level flight, zero velocity)

    Returns:
        State array [y, z, φ, ẏ, ż, φ̇]
    """
    keys = jax.random.split(key, 6)

    # Random position within workspace
    y = jax.random.uniform(
        keys[0],
        minval=config['workspace']['y_range'][0],
        maxval=config['workspace']['y_range'][1]
    )

    z = jax.random.uniform(
        keys[1],
        minval=config['workspace']['z_range'][0],
        maxval=config['workspace']['z_range'][1]
    )

    if is_goal:
        # Goal state: level (φ=0) with zero velocity
        phi = 0.0
        y_dot = 0.0
        z_dot = 0.0
        phi_dot = 0.0
    else:
        # Random initial roll angle (small perturbation)
        phi = jax.random.uniform(
            keys[2],
            minval=config['start_state']['phi_range'][0],
            maxval=config['start_state']['phi_range'][1]
        )

        # Random initial velocities
        y_dot = jax.random.uniform(
            keys[3],
            minval=config['start_state']['velocity_range'][0],
            maxval=config['start_state']['velocity_range'][1]
        )

        z_dot = jax.random.uniform(
            keys[4],
            minval=config['start_state']['velocity_range'][0],
            maxval=config['start_state']['velocity_range'][1]
        )

        phi_dot = jax.random.uniform(
            keys[5],
            minval=config['start_state']['phi_dot_range'][0],
            maxval=config['start_state']['phi_dot_range'][1]
        )

    return jnp.array([y, z, phi, y_dot, z_dot, phi_dot])


def generate_random_goal_states(key, num_goals, config):
    """Generate random goal states (hover targets).

    Args:
        key: JAX random key
        num_goals: Number of goal states to generate
        config: Configuration dictionary

    Returns:
        Array of goal states (num_goals, 6)
    """
    keys = jax.random.split(key, num_goals)
    gen_goal = lambda k: generate_random_state(k, config, is_goal=True)
    goal_states = jax.vmap(gen_goal)(keys)
    return goal_states


def solve_hover_trajectory(key, start_state, goal_state, quad_params, config):
    """Solve a hover/position-reaching trajectory using iLQR.

    Args:
        key: JAX random key
        start_state: Initial state [y, z, φ, ẏ, ż, φ̇]
        goal_state: Goal state [y, z, φ, ẏ, ż, φ̇]
        quad_params: PlanarQuadrotorParams for this trajectory
        config: Configuration dictionary

    Returns:
        Tuple of (states, actions, costs)
    """
    quad = PlanarQuadrotor(quad_params)

    # Create dynamics function
    def dynamics_func(state, control, t):
        return quad.step(state, control, config['dt'])

    # Extract goal components and weights
    goal_y, goal_z, goal_phi, goal_ydot, goal_zdot, goal_phidot = goal_state

    pos_w = config['cost_weights']['position_weight']
    vel_w = config['cost_weights']['velocity_weight']
    angle_w = config['cost_weights']['angle_weight']
    angvel_w = config['cost_weights']['angular_velocity_weight']
    ctrl_w = config['cost_weights']['control_weight']
    term_pos_w = config['cost_weights']['terminal_position_weight']
    term_vel_w = config['cost_weights']['terminal_velocity_weight']
    term_angle_w = config['cost_weights']['terminal_angle_weight']
    horizon = config['horizon']

    # Hover thrust for this quadrotor
    f_hover = quad_params.mass * quad_params.gravity

    def cost_func(state, control, t):
        y, z, phi, y_dot, z_dot, phi_dot = state
        f, tau = control

        # Position errors
        pos_error_sq = (y - goal_y)**2 + (z - goal_z)**2

        # Angle error (want level flight)
        angle_error_sq = (phi - goal_phi)**2

        # Velocity errors
        vel_error_sq = (y_dot - goal_ydot)**2 + (z_dot - goal_zdot)**2

        # Angular velocity error
        angvel_error_sq = (phi_dot - goal_phidot)**2

        # Control effort (penalize deviation from hover)
        # Using (f - f_hover)^2 encourages staying near hover thrust
        control_effort = (f - f_hover)**2 + tau**2

        # Stage cost
        stage_cost = (
            pos_w * pos_error_sq +
            vel_w * vel_error_sq +
            angle_w * angle_error_sq +
            angvel_w * angvel_error_sq +
            ctrl_w * control_effort
        )

        # Terminal cost
        terminal_cost = (
            term_pos_w * pos_error_sq +
            term_vel_w * vel_error_sq +
            term_angle_w * angle_error_sq +
            term_angle_w * angvel_error_sq  # Also penalize angular velocity at terminal
        )

        # Add terminal cost at last timestep
        total_cost = jnp.where(t >= horizon - 1, stage_cost + terminal_cost, stage_cost)

        return total_cost

    # Initial control guess: hover thrust with zero torque
    U_init = jnp.tile(jnp.array([f_hover, 0.0]), (config['horizon'], 1))

    # Solve using iLQR
    result = optimizers.ilqr(
        cost_func,
        dynamics_func,
        start_state,
        U_init,
        maxiter=config['max_iterations'],
    )

    states = result[0]
    actions = result[1]

    # Compute costs along trajectory
    costs = jax.vmap(cost_func)(
        states[:-1, :],
        actions,
        jnp.arange(actions.shape[0])
    )

    return states, actions, costs


def generate_dataset(
    num_dynamics_configs,
    goal_states,
    config,
    save_path=None,
    seed=0
):
    """Generate a dataset with multiple goal states and varying dynamics.

    Args:
        num_dynamics_configs: Number of different dynamics configurations
        goal_states: Array of goal states (num_goals, 6)
        config: Configuration dictionary
        save_path: Path to save the dataset (optional)
        seed: Random seed

    Returns:
        Dictionary containing states, actions, costs, and metadata
    """
    key = jax.random.PRNGKey(seed)

    num_goals = len(goal_states)
    num_trajectories_per_task = config.get('num_trajectories_per_task', 10)
    trajectories_per_config = num_goals * num_trajectories_per_task
    total_trajectories = num_dynamics_configs * trajectories_per_config

    print(f"Generating planar quadrotor dataset with varying dynamics:")
    print(f"  Number of goal states: {num_goals}")
    print(f"  Dynamics configurations: {num_dynamics_configs}")
    print(f"  Trajectories per (dynamics, goal) task: {num_trajectories_per_task}")
    print(f"  Trajectories per dynamics config: {trajectories_per_config}")
    print(f"  Total trajectories: {total_trajectories}")
    print(f"  State dimension: 6 [y, z, φ, ẏ, ż, φ̇]")
    print(f"  Control dimension: 2 [f, τ]")

    # Generate dynamics configurations
    print("\nGenerating dynamics configurations...")
    key, subkey = jax.random.split(key)
    dynamics_configs = generate_dynamics_configurations(
        subkey, num_dynamics_configs, config
    )

    print(f"Generated {len(dynamics_configs)} dynamics configurations:")
    for i, params in enumerate(dynamics_configs[:5]):
        print(f"  Config {i}: mass={params.mass:.3f}kg, "
              f"inertia={params.inertia:.6f}kg*m², "
              f"arm={params.arm_length:.3f}m")
    if len(dynamics_configs) > 5:
        print(f"  ... ({len(dynamics_configs) - 5} more configs)")

    # Generate start states for all trajectories
    print("\nGenerating start states...")
    key, subkey = jax.random.split(key)
    state_keys = jax.random.split(subkey, total_trajectories)

    gen_start_state = lambda k: generate_random_state(k, config, is_goal=False)
    start_states_all = jax.vmap(gen_start_state)(state_keys)
    print(f"Start states shape: {start_states_all.shape}")

    # Create dynamics indices and goal indices
    dynamics_indices = []
    goal_indices = []

    for config_idx in range(num_dynamics_configs):
        for goal_idx in range(num_goals):
            for _ in range(num_trajectories_per_task):
                dynamics_indices.append(config_idx)
                goal_indices.append(goal_idx)

    dynamics_indices = jnp.array(dynamics_indices)
    goal_indices = jnp.array(goal_indices)

    # Solve trajectories
    print("\nSolving trajectories (this may take a while)...")
    key, subkey = jax.random.split(key)
    solve_keys = jax.random.split(subkey, total_trajectories)

    all_states = []
    all_actions = []
    all_costs = []

    traj_idx = 0
    for config_idx in range(num_dynamics_configs):
        num_traj = trajectories_per_config

        print(f"  Solving {num_traj} trajectories for dynamics config "
              f"{config_idx+1}/{num_dynamics_configs}...")

        quad_params = dynamics_configs[config_idx]

        # Get start states and goal indices for this config
        start_states_batch = start_states_all[traj_idx:traj_idx + num_traj]
        goal_indices_batch = goal_indices[traj_idx:traj_idx + num_traj]
        goal_states_ = goal_states[goal_indices_batch]

        # Solve for this dynamics configuration
        def solve_fn(k, start, goal):
            return solve_hover_trajectory(
                k, start, goal, quad_params, config
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

    # Create dataset dictionary
    dataset = {
        'states': np.array(states),
        'actions': np.array(actions),
        'costs': np.array(costs),
        'start_states': np.array(start_states_all),
        'goal_states': np.array(goal_states),
        'goal_indices': np.array(goal_indices),
        'dynamics_indices': np.array(dynamics_indices),
        'trajectories_per_config': trajectories_per_config,
        'num_trajectories_per_task': num_trajectories_per_task,
        'dynamics_params': dynamics_configs,
        'config': config,
        'generation_time': datetime.now().isoformat(),
        'num_dynamics_configs': num_dynamics_configs,
        'num_goals': num_goals,
        'total_trajectories': total_trajectories,
    }

    print(f"\nSuccessfully generated {total_trajectories} trajectories")

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **dataset)
        print(f"\nDataset saved to {save_path}")
        print(f"File size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")

    return dataset


def config_from_yaml(data_cfg: dict) -> dict:
    """Convert the data section of a YAML config to the dict format expected by generate_dataset."""
    return {
        'horizon': data_cfg['horizon'],
        'dt': data_cfg['dt'],
        'max_iterations': data_cfg['max_iterations'],
        'workspace': {
            'y_range': tuple(data_cfg['workspace']['y_range']),
            'z_range': tuple(data_cfg['workspace']['z_range']),
        },
        'dynamics_ranges': {
            'mass_range': tuple(data_cfg['dynamics_ranges']['mass_range']),
            'inertia_scale_range': tuple(data_cfg['dynamics_ranges']['inertia_scale_range']),
            'arm_length_range': tuple(data_cfg['dynamics_ranges']['arm_length_range']),
            'max_torque': data_cfg['dynamics_ranges']['max_torque'],
        },
        'num_trajectories_per_task': data_cfg['num_trajectories_per_task'],
        'start_state': {
            'phi_range': tuple(data_cfg['start_state']['phi_range']),
            'velocity_range': tuple(data_cfg['start_state']['velocity_range']),
            'phi_dot_range': tuple(data_cfg['start_state']['phi_dot_range']),
        },
        'goal_state': {
            'type': data_cfg['goal_state']['type'],
            'num_goals': data_cfg['goal_state'].get('num_goals', 1),
            'fixed_position': data_cfg['goal_state'].get('fixed_position'),
        },
        'cost_weights': {
            'position_weight': data_cfg['cost_weights']['position_weight'],
            'velocity_weight': data_cfg['cost_weights']['velocity_weight'],
            'angle_weight': data_cfg['cost_weights']['angle_weight'],
            'angular_velocity_weight': data_cfg['cost_weights']['angular_velocity_weight'],
            'control_weight': data_cfg['cost_weights']['control_weight'],
            'terminal_position_weight': data_cfg['cost_weights']['terminal_position_weight'],
            'terminal_velocity_weight': data_cfg['cost_weights']['terminal_velocity_weight'],
            'terminal_angle_weight': data_cfg['cost_weights']['terminal_angle_weight'],
        }
    }


def run_from_yaml(cfg: dict, output_dir: str, seed: int) -> dict:
    """Entry point called by generate_data.py dispatcher."""
    data_cfg = cfg['data']
    config = config_from_yaml(data_cfg)

    # Generate goal states
    goal_type = data_cfg['goal_state']['type']
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    if goal_type == "fixed":
        fixed_pos = jnp.array(data_cfg['goal_state']['fixed_position'])
        goal_states = fixed_pos[None, :]
    else:
        goal_states = generate_random_goal_states(
            subkey, data_cfg['goal_state'].get('num_goals', 1), config
        )

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

    parser = argparse.ArgumentParser(description="Generate quadrotor dataset")
    parser.add_argument("--config", default="configs/quadrotor.yaml", help="Path to config YAML")
    parser.add_argument("--output", default="data/quadrotor", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_from_yaml(cfg, args.output, args.seed)
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()