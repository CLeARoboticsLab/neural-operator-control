"""Unified SetONet meta-training entry point.

Supports two variants controlled by --variant:
  meta_branch: Inner loop adapts branch only (phi/aggregator/rho), trunk frozen
  meta_full:   Inner loop adapts all parameters (branch + trunk)

Usage (matches Makefile):
    python src/training/meta_train.py \
        --config configs/p2p_cost.yaml \
        --data data/p2p_cost \
        --variant meta_branch \
        --output checkpoints/p2p_cost/meta_branch \
        --seed 42 --device cpu
"""

import argparse
import sys
import yaml


DISPATCH = {
    "p2p_cost": "src.training.meta_train_p2p_cost",
    "p2p_dynamics": "src.training.meta_train_p2p_dynamics",
    "quadrotor": "src.training.meta_train_quadrotor",
    "obstacle": "src.training.meta_train_obstacle",
}

VALID_VARIANTS = {"meta_branch", "meta_full"}


def main():
    parser = argparse.ArgumentParser(description="Meta-train SetONet for an environment")
    parser.add_argument("--config", required=True, help="Path to environment YAML config")
    parser.add_argument("--data", required=True, help="Path to data directory")
    parser.add_argument("--variant", required=True, choices=sorted(VALID_VARIANTS),
                        help="meta_branch (freeze trunk) or meta_full (adapt all)")
    parser.add_argument("--output", required=True, help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generator_name = cfg.get("generator")
    if generator_name is None:
        print(f"Error: config {args.config} missing 'generator' field", file=sys.stderr)
        sys.exit(1)

    if generator_name not in DISPATCH:
        print(f"Error: unknown generator '{generator_name}'. "
              f"Available: {list(DISPATCH.keys())}", file=sys.stderr)
        sys.exit(1)

    print(f"Meta-Trainer: {generator_name}")
    print(f"Variant:      {args.variant}")
    print(f"Config:       {args.config}")
    print(f"Data:         {args.data}")
    print(f"Output:       {args.output}")
    print(f"Seed:         {args.seed}")
    print("=" * 60)

    if generator_name == "p2p_cost":
        from src.training.meta_train_p2p_cost import run_training
    elif generator_name == "p2p_dynamics":
        from src.training.meta_train_p2p_dynamics import run_training
    elif generator_name == "quadrotor":
        from src.training.meta_train_quadrotor import run_training
    elif generator_name == "obstacle":
        from src.training.meta_train_obstacle import run_training

    run_training(cfg, data_dir=args.data, output_dir=args.output,
                 variant=args.variant, seed=args.seed, device=args.device)

    print("\nMeta-training complete.")


if __name__ == "__main__":
    main()
