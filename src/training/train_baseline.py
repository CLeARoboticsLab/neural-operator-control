"""Entry point for the task-conditioned MLP baseline.

Dispatches to the per-environment baseline trainer based on the `generator`
field in the config YAML. Mirrors ``src/training/train.py``.

Usage:
    python src/training/train_baseline.py \
        --config configs/p2p_cost.yaml \
        --data data/p2p_cost \
        --output checkpoints/p2p_cost/baseline \
        --seed 42 --device cpu
"""

import argparse
import sys
import yaml


# Per-env runners. P2P-Cost and P2P-Cost-Small share generator "p2p_cost".
DISPATCH = {
    "p2p_cost": "src.training.train_baseline_p2p_cost",
    "p2p_dynamics": "src.training.train_baseline_p2p_dynamics",
    "quadrotor": "src.training.train_baseline_quadrotor",
    "obstacle": "src.training.train_baseline_obstacle",
}


def main():
    parser = argparse.ArgumentParser(description="Train task-conditioned MLP baseline")
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
        print(f"Error: baseline trainer for '{generator}' not implemented. "
              f"Available: {list(DISPATCH)}", file=sys.stderr)
        sys.exit(1)

    print(f"Baseline trainer: {generator}")
    print(f"Config: {args.config} | Data: {args.data} | Output: {args.output} | Seed: {args.seed}")
    print("=" * 60)

    module = __import__(DISPATCH[generator], fromlist=["run_training"])
    module.run_training(cfg, data_dir=args.data, output_dir=args.output,
                        seed=args.seed, device=args.device)
    print("\nBaseline training complete.")


if __name__ == "__main__":
    main()
