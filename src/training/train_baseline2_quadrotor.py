"""Baseline 2 (context-conditioned MLP) for Quadrotor.

Reuses the SetONet quadrotor dataloader so the context is exactly the operator's
branch input — (state, action) locations with next-state values — restricted to
the canonical train split for a leak-free comparison.
"""

import numpy as np
import jax.random as jr

from src.envs.dataloader import VaryingDynamicsData  # noqa: F401 (kept for parity)
from src.training.train_quadrotor import QuadrotorDataLoader
from src.training.baseline_data import dynamics_split, restrict_dynamics_dataset
from src.training.baseline2_common import fit_b2
from src.evaluation.evaluate_baseline import build_setonet
from src.training.baseline_mlp import count_params

SLOTS = 128   # fixed context bank; SetONet quadrotor samples N=K (=64) -> 64 filled, 64 padded


def run_training(cfg, data_dir, output_dir, seed=42, device="cpu"):
    from pathlib import Path
    train_cfg = cfg.get("training", {})
    split_seed = train_cfg.get("split_seed", 42)

    raw = dict(np.load(Path(data_dir) / "trajectories.npz", allow_pickle=True))
    train_cfgs, _test, _g0 = dynamics_split(data_dir, split_seed=split_seed)
    raw = restrict_dynamics_dataset(raw, train_cfgs)
    g0 = int(np.unique(raw["goal_indices"])[0])

    np.random.seed(seed)
    dl = QuadrotorDataLoader.load_for_goal(g0, raw, train_perc=0.8)

    state_dim = raw["states"].shape[-1]       # 6
    action_dim = raw["actions"].shape[-1]     # 2
    elem_dim = state_dim + action_dim + state_dim  # (state, action, next_state) = 14

    K_cfg = train_cfg.get("K", 64)
    n_ctx = int(max(K_cfg)) if isinstance(K_cfg, (list, tuple)) else int(K_cfg)
    n_traj = int(train_cfg.get("N", 5))

    # baseline2_fixed_K: NO-PADDING mode — train & eval at one fixed K with slot bank == K.
    fixed_K = train_cfg.get("baseline2_fixed_K")
    b2_slots = int(fixed_K) if fixed_K else SLOTS
    if fixed_K:
        n_ctx = int(fixed_K)

    def sample_task():
        rs, ra, rns, expert_traj, expert_actions = dl.sample_task("train", K=n_ctx, N=n_traj)
        rs, ra, rns = np.asarray(rs), np.asarray(ra), np.asarray(rns)
        ctx = np.concatenate([rs, ra, rns], axis=-1)                   # (n_ctx, 14) raw
        queries = np.asarray(expert_traj)[:, :-1, :]                   # (n_traj, H, 7) [state,time]
        targets = np.asarray(expert_actions) * dl.action_std + dl.action_mean  # raw actions
        return ctx, queries, targets

    target_params = count_params(build_setonet(cfg, state_dim + action_dim, state_dim,
                                               state_dim + 1, action_dim, jr.PRNGKey(0)))

    return fit_b2(cfg, sample_task, state_dim, action_dim, elem_dim, b2_slots, n_ctx,
                  output_dir, seed, "Quadrotor", target_params)
