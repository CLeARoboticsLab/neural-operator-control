"""Baseline 2 (context-conditioned MLP) for Obstacle.

The operator's branch for this env is the obstacle set itself — (x, y) locations
with radius values — so Baseline 2 here is Baseline 1 plus the radius, written
into a fixed bank of `max_obs` slots padded with -1. (Unlike the dynamics envs,
the context is the static obstacle configuration, not sampled transitions.)
"""

import numpy as np
import jax.random as jr

from src.training.baseline_data import extract_obstacle
from src.training.baseline2_common import fit_b2
from src.evaluation.evaluate_baseline import build_setonet
from src.training.baseline_mlp import count_params

N_TRAJ = 8  # expert trajectories sampled per task per iteration


def run_training(cfg, data_dir, output_dir, seed=42, device="cpu"):
    split_seed = cfg.get("training", {}).get("split_seed", 42)
    train_tasks, _test, info = extract_obstacle(data_dir, split_seed=split_seed)
    state_dim, action_dim = info["state_dim"], info["action_dim"]
    slots, elem_dim = info["max_obs"], 3  # (x, y, radius)

    np.random.seed(seed)

    def sample_task():
        t = train_tasks[np.random.randint(len(train_tasks))]
        n = t["states"].shape[0]
        k = min(n, N_TRAJ)
        idx = np.random.choice(n, size=k, replace=False)
        s, a = t["states"][idx], t["actions"][idx]
        h = s.shape[1]
        time = np.broadcast_to(np.linspace(0.0, 1.0, h, endpoint=False)[None, :, None], (k, h, 1))
        queries = np.concatenate([s, time], axis=-1)     # (k, H, state_dim+1)
        ctx = t["obstacles"]                             # (n_obs, 3) raw (x, y, radius)
        return ctx, queries, a

    target_params = count_params(build_setonet(cfg, 2, 1, state_dim + 1, action_dim, jr.PRNGKey(0)))
    return fit_b2(cfg, sample_task, state_dim, action_dim, elem_dim, slots, slots,
                  output_dir, seed, "Obstacle", target_params)
