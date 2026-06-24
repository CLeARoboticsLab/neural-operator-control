"""Unified data generation entry point.

Dispatches to the appropriate generator based on the `generator` field
in the environment config YAML.

Usage (matches Makefile convention):
    python src/envs/generate_data.py --config configs/p2p_cost.yaml --output data/p2p_cost --seed 42
"""

import argparse
import sys
from pathlib import Path

import yaml


# Registry of generator name -> (module path, callable name)
GENERATORS = {
    "p2p_cost": "src.envs.create_p2p_cost",
}


def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_obstacle(cfg: dict, output_dir: str, seed: int) -> None:
    """Run the obstacle avoidance CasADi/IPOPT data generator."""
    from src.envs.create_obstacle import run_from_yaml

    run_from_yaml(cfg, output_dir, seed)


def generate_quadrotor(cfg: dict, output_dir: str, seed: int) -> None:
    """Run the planar quadrotor data generator."""
    from src.envs.create_quadrotor import run_from_yaml

    run_from_yaml(cfg, output_dir, seed)


def generate_p2p_dynamics(cfg: dict, output_dir: str, seed: int) -> None:
    """Run the varying-dynamics iLQR data generator."""
    from src.envs.create_p2p_dynamics import run_from_yaml

    run_from_yaml(cfg, output_dir, seed)


def generate_p2p_cost(cfg: dict, output_dir: str, seed: int) -> None:
    """Run the double-integrator LQR data generator."""
    from src.envs.create_p2p_cost import generate_dataset

    data_cfg = cfg["data"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = str(output_path / "trajectories.npz")

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
        seed=seed,
        save_path=save_path,
    )


# Dispatch table: generator name -> function
DISPATCH = {
    "p2p_cost": generate_p2p_cost,
    "p2p_dynamics": generate_p2p_dynamics,
    "quadrotor": generate_quadrotor,
    "obstacle": generate_obstacle,
}


def main():
    parser = argparse.ArgumentParser(description="Generate dataset for an environment")
    parser.add_argument("--config", required=True, help="Path to environment YAML config")
    parser.add_argument("--output", required=True, help="Output directory for generated data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = load_config(args.config)

    generator_name = cfg.get("generator")
    if generator_name is None:
        print(f"Error: config {args.config} missing 'generator' field", file=sys.stderr)
        sys.exit(1)

    if generator_name not in DISPATCH:
        print(
            f"Error: unknown generator '{generator_name}'. "
            f"Available: {list(DISPATCH.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Override seed from config if provided on CLI
    seed = args.seed

    print(f"Generator: {generator_name}")
    print(f"Config:    {args.config}")
    print(f"Output:    {args.output}")
    print(f"Seed:      {seed}")
    print("=" * 60)

    DISPATCH[generator_name](cfg, args.output, seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
