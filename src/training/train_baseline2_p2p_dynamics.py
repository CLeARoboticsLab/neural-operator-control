"""Baseline 2 (context-conditioned MLP) for P2P-Dynamics.

Context = SetONet's branch: (state, action) locations with next-state values,
sampled with a variable size N drawn from the env's K options each iteration and
padded to a fixed bank of slots. Restricted to the canonical train split.
"""

import numpy as np
import jax.random as jr
from pathlib import Path

from src.envs.dataloader import VaryingDynamicsData
from src.training.baseline_data import dynamics_split, restrict_dynamics_dataset
from src.training.baseline2_common import fit_b2
from src.evaluation.evaluate_baseline import build_setonet
from src.training.baseline_mlp import count_params

SLOTS = 128
EVAL_N_CTX = 64   # context size used at eval (matches SetONet eval K)


def run_training(cfg, data_dir, output_dir, seed=42, device="cpu"):
    train_cfg = cfg.get("training", {})
    split_seed = train_cfg.get("split_seed", 42)

    raw = dict(np.load(Path(data_dir) / "trajectories.npz", allow_pickle=True))
    train_cfgs, _test, _g0 = dynamics_split(data_dir, split_seed=split_seed)
    raw = restrict_dynamics_dataset(raw, train_cfgs)
    g0 = int(np.unique(raw["goal_indices"])[0])

    np.random.seed(seed)
    dl = VaryingDynamicsData.load_for_goal(g0, raw, train_perc=0.8)

    state_dim = raw["states"].shape[-1]
    action_dim = raw["actions"].shape[-1]
    elem_dim = state_dim + action_dim + state_dim

    K_cfg = train_cfg.get("K", [32, 64, 128])
    k_options = [int(k) for k in K_cfg] if isinstance(K_cfg, (list, tuple)) else [int(K_cfg)]
    n_traj = int(train_cfg.get("N", 8))

    # baseline2_fixed_K: NO-PADDING mode — train & eval at one fixed K with slot bank == K.
    fixed_K = train_cfg.get("baseline2_fixed_K")
    if fixed_K:
        fixed_K = int(fixed_K)
        k_options = [fixed_K]; b2_slots = fixed_K; eval_n_ctx = fixed_K
    else:
        b2_slots = SLOTS; eval_n_ctx = EVAL_N_CTX

    def sample_task():
        n_ctx = int(np.random.choice(k_options))
        out = dl.get_task("train", K=n_ctx, N=n_traj)
        rs, ra, rns = np.asarray(out[0]), np.asarray(out[1]), np.asarray(out[2])
        expert_traj, expert_actions = np.asarray(out[4]), np.asarray(out[5])
        ctx = np.concatenate([rs, ra, rns], axis=-1)        # (n_ctx, elem)
        queries = expert_traj[:, :-1, :]                    # (n_traj, H, state_dim+1) [state, int-time]
        return ctx, queries, expert_actions

    target_params = count_params(build_setonet(cfg, state_dim + action_dim, state_dim,
                                               state_dim + 1, action_dim, jr.PRNGKey(0)))
    return fit_b2(cfg, sample_task, state_dim, action_dim, elem_dim, b2_slots, eval_n_ctx,
                  output_dir, seed, "P2P-Dynamics", target_params)
