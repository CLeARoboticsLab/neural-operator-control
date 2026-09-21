"""Unified MAML training entry point.

Dispatches to the appropriate MAML trainer based on the `generator` field
in the environment config YAML.

Usage (matches Makefile convention):
    python src/training/train_maml.py \
        --config configs/p2p_cost.yaml \
        --data data/p2p_cost \
        --output checkpoints/p2p_cost/maml \
        --seed 42 --device cpu
"""

import argparse
import sys

import yaml


DISPATCH = {
    "p2p_cost": "src.training.train_maml_p2p_cost",
    "p2p_dynamics": "src.training.train_maml_p2p_dynamics",
    "quadrotor": "src.training.train_maml_quadrotor",
    "obstacle": "src.training.train_maml_obstacle",
    "halfcheetah": "src.training.train_maml_halfcheetah",
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train MAML baseline for an environment")
    parser.add_argument("--config", required=True, help="Path to environment YAML config")
    parser.add_argument("--data", required=True, help="Path to data directory")
    parser.add_argument("--output", required=True, help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
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

    print(f"MAML Trainer: {generator_name}")
    print(f"Config:       {args.config}")
    print(f"Data:         {args.data}")
    print(f"Output:       {args.output}")
    print(f"Seed:         {args.seed}")
    print(f"Device:       {args.device}")
    print("=" * 60)

    if generator_name == "p2p_cost":
        from src.training.train_maml_p2p_cost import run_training
    elif generator_name == "p2p_dynamics":
        from src.training.train_maml_p2p_dynamics import run_training
    elif generator_name == "quadrotor":
        from src.training.train_maml_quadrotor import run_training
    elif generator_name == "obstacle":
        from src.training.train_maml_obstacle import run_training
    elif generator_name == "halfcheetah":
        from src.training.train_maml_halfcheetah import run_training

    run_training(cfg, data_dir=args.data, output_dir=args.output,
                 seed=args.seed, device=args.device)

    print("\nMAML training complete.")


if __name__ == "__main__":
    main()
