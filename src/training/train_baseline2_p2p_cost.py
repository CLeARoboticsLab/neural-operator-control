"""Baseline 2 (context-conditioned MLP) for P2P-Cost, on the STORED LQR dataset.

Same data as the operator: context = SetONet's branch input — on-policy
(state, control) transitions with immediate-cost values, drawn from the stored
LQR trajectories via ``DoubleIntegratorLQRData`` — consumed as a flat padded
vector instead of a set encoder.
"""

import numpy as np
import jax.random as jr
import random as _random
from pathlib import Path

from src.training.baseline2_common import fit_b2
from src.evaluation.evaluate_baseline import build_setonet
from src.training.baseline_mlp import count_params
from src.envs.dataloader import DoubleIntegratorLQRData

STATE_DIM, CONTROL_DIM = 4, 2


def run_training(cfg, data_dir, output_dir, seed=42, device="cpu"):
    train_cfg = cfg["training"]
    K_cfg = train_cfg["K"]
    op_K = max([int(k) for k in K_cfg]) if isinstance(K_cfg, (list, tuple)) else int(K_cfg)
    # B2 (concat-MLP) trains with a VARIABLE real-context count up to the operator's K
    # (so it learns padding and can be evaluated at a smaller context, like the operator's
    # size-agnostic set encoder). Slot bank = op_K; eval context = the operator's eval K.
    b2_train_K = sorted({32, 64, op_K})
    eval_n_ctx = int(train_cfg.get("baseline2_eval_K", 64))  # match operator eval context
    N = int(train_cfg.get("N", 32))
    train_perc = train_cfg.get("train_perc", 0.8)
    elem_dim = STATE_DIM + CONTROL_DIM + 1  # (state, control, cost) = 7

    raw = np.load(Path(data_dir) / "trajectories.npz", allow_pickle=True)
    dataset = {k: raw[k] for k in raw.files}
    if "norm_stats" in dataset:
        dataset["norm_stats"] = dataset["norm_stats"].item()

    np.random.seed(seed); _random.seed(seed)
    dl = DoubleIntegratorLQRData(dataset, train_perc=train_perc, normalize=True)
    ns = dl.norm_stats
    max_action = float(dl.max_action)

    def sample_task():
        n_ctx = int(np.random.choice(b2_train_K))
        out = dl.get_task("train", K=n_ctx, N=N)
        src_s, src_c, src_cost = out[0], out[1], out[2]   # normalized (n_ctx, 4/2/1)
        ctx = np.concatenate([src_s, src_c, src_cost], axis=-1)  # (n_ctx, 7), on-policy
        queries = out[4]                                   # (N, H, 5) normalized [state, time]
        targets = out[5]                                   # (N, H, 2) normalized
        return ctx, queries, targets

    target_params = count_params(build_setonet(cfg, STATE_DIM + CONTROL_DIM, 1,
                                               STATE_DIM + 1, CONTROL_DIM, jr.PRNGKey(0)))
    return fit_b2(cfg, sample_task, STATE_DIM, CONTROL_DIM, elem_dim, op_K, eval_n_ctx,
                  output_dir, seed, "P2P-Cost", target_params,
                  extra_meta={"max_action": max_action,
                              "state_mean": np.asarray(ns["state_mean"]),
                              "state_std": np.asarray(ns["state_std"]),
                              "cost_mean": float(ns["cost_mean"]),
                              "cost_std": float(ns["cost_std"])})
