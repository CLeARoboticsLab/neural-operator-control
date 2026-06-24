"""Entry point for Baseline 2 (context-conditioned MLP).

Dispatches to the per-environment trainer by the config's `generator` field.
Mirrors src/training/train_baseline.py.

Usage:
    python src/training/train_baseline2.py \
        --config configs/quadrotor.yaml \
        --data data/quadrotor \
        --output checkpoints/quadrotor/baseline2 \
        --seed 42
"""

import argparse
import sys
import yaml


DISPATCH = {
    "p2p_cost": "src.training.train_baseline2_p2p_cost",       # also p2p_cost_small
    "p2p_dynamics": "src.training.train_baseline2_p2p_dynamics",
    "quadrotor": "src.training.train_baseline2_quadrotor",
    "obstacle": "src.training.train_baseline2_obstacle",
}


def main():
    parser = argparse.ArgumentParser(description="Train Baseline 2 (context-conditioned MLP)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    generator = cfg.get("generator")
    if generator not in DISPATCH:
        print(f"Error: Baseline 2 trainer for '{generator}' not implemented. "
              f"Available: {list(DISPATCH)}", file=sys.stderr)
        sys.exit(1)

    print(f"Baseline 2 trainer: {generator} | config {args.config} | seed {args.seed}")
    print("=" * 60)
    module = __import__(DISPATCH[generator], fromlist=["run_training"])
    module.run_training(cfg, data_dir=args.data, output_dir=args.output,
                        seed=args.seed, device=args.device)
    print("\nBaseline 2 training complete.")


if __name__ == "__main__":
    main()
