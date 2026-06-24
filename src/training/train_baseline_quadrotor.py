"""Task-conditioned MLP baseline for Quadrotor (varying dynamics)."""

from src.training.baseline_data import extract_dynamics
from src.training.train_baseline_stored import fit


def run_training(cfg, data_dir, output_dir, seed=42, device="cpu"):
    split_seed = cfg.get("training", {}).get("split_seed", 42)
    train_tasks, _test_tasks, info = extract_dynamics(data_dir, split_seed=split_seed)
    return fit(cfg, train_tasks, info, output_dir, seed, split_seed, env_name="Quadrotor")
