"""Task-conditioned MLP baseline (B1) for P2P-Cost, on the STORED LQR dataset.

Same data as the operator now uses (``double_integrator_lqr.npz`` via
``DoubleIntegratorLQRData``). A plain MLP receives ``[state, time, goal]`` and
predicts the control along the expert LQR trajectory. States/controls are
normalized by the dataloader; the goal is z-scored with the state statistics.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import optax
import random as _random
from pathlib import Path

from src.training.baseline_mlp import BaselineMLP, train_step, count_params
from src.envs.dataloader import DoubleIntegratorLQRData

STATE_DIM, CONTROL_DIM = 4, 2
INPUT_DIM = STATE_DIM + 1 + STATE_DIM  # [state, time, goal]


def run_training(cfg: dict, data_dir: str, output_dir: str, seed: int = 42, device: str = "cpu"):
    train_cfg = cfg["training"]
    mlp_cfg = cfg.get("baseline_model", cfg.get("maml_model", {}))
    hidden_size = mlp_cfg.get("hidden_size", 256)
    num_layers = mlp_cfg.get("num_layers", 4)
    M = train_cfg["M"]
    N = int(train_cfg.get("N", 32))
    num_iterations = train_cfg["num_iterations"]
    eval_interval = train_cfg["eval_every"]
    learning_rate = train_cfg["learning_rate"]
    train_perc = train_cfg.get("train_perc", 0.8)

    raw = np.load(Path(data_dir) / "trajectories.npz", allow_pickle=True)
    dataset = {k: raw[k] for k in raw.files}
    if "norm_stats" in dataset:
        dataset["norm_stats"] = dataset["norm_stats"].item()

    np.random.seed(seed); _random.seed(seed)
    dl = DoubleIntegratorLQRData(dataset, train_perc=train_perc, normalize=True)
    ns = dl.norm_stats
    sm, ss_ = np.asarray(ns["state_mean"]), np.asarray(ns["state_std"])
    max_action = float(dl.max_action)

    print("=" * 60)
    print(f"P2P-Cost — MLP Baseline 1 (stored) | MLP({INPUT_DIM} -> {CONTROL_DIM}, "
          f"width={hidden_size}, depth={num_layers}) | M={M} N={N} iters={num_iterations}")
    print("=" * 60)

    def sample_b1(num_tasks):
        Xs, Ys = [], []
        for _ in range(num_tasks):
            out = dl.get_task("train", K=1, N=N)        # src ignored; tgt + goal used
            tgt_s, tgt_c, goal = out[4], out[5], out[6]  # tgt_s normalized (N,H,5), tgt_c norm, goal raw
            goal_n = (goal - sm) / ss_
            q = tgt_s.reshape(-1, tgt_s.shape[-1])       # (N*H, 5) = [state, time]
            gb = np.broadcast_to(goal_n[None, :], (q.shape[0], STATE_DIM))
            Xs.append(np.concatenate([q, gb], axis=-1))
            Ys.append(tgt_c.reshape(-1, CONTROL_DIM))
        return (jnp.asarray(np.concatenate(Xs).astype(np.float32)),
                jnp.asarray(np.concatenate(Ys).astype(np.float32)))

    key = jr.PRNGKey(seed)
    key, model_key = jr.split(key)
    model = BaselineMLP(INPUT_DIM, CONTROL_DIM, hidden_size, num_layers, model_key)
    print(f"Parameter count: {count_params(model):,}")
    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    train_losses, eval_losses = [], []
    print("\nStarting training...")
    for iteration in range(num_iterations):
        x, y = sample_b1(M)
        loss, model, opt_state = train_step(model, optim, opt_state, x, y)
        train_losses.append(float(loss))
        if iteration % eval_interval == 0:
            xe, ye = sample_b1(train_cfg.get("num_eval_tasks", 20))
            eval_loss = float(jnp.mean(jnp.square(jax.vmap(model)(xe) - ye)))
            eval_losses.append(eval_loss)
            print(f"  Iter {iteration:5d} | Train: {loss:.6f} | Eval: {eval_loss:.6f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(output_path / "baseline.eqx", model)
    print(f"\nModel saved to {output_path / 'baseline.eqx'}")
    np.savez(
        output_path / "training_history.npz",
        train_losses=np.array(train_losses), eval_losses=np.array(eval_losses),
        seed=seed,
        state_mean=sm, state_std=ss_,
        cost_mean=np.array(ns["cost_mean"]), cost_std=np.array(ns["cost_std"]),
        max_action=np.array(max_action),
        hidden_size=hidden_size, num_layers=num_layers,
    )
    print(f"Training history saved to {output_path / 'training_history.npz'}")
    return model, ns
