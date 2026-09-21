"""Pretrain SetONet on HalfCheetah-v3 (iMuJoCo) by behavioral cloning.

Each task is one dynamics configuration. The branch receives logged
(state, action) -> next_state transitions from K context episodes; the trunk
maps query states from K other episodes of the same task to the expert action.
Best-of-``num_runs`` selection by validation loss, as for the OCP environments.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import yaml

from src.envs.imujoco_dataloader import IMuJoCoImitation
from src.training.halfcheetah_common import (
    build_setonet, k_options, task_mse, train_step,
)


@eqx.filter_value_and_grad
def batch_loss(model, batch):
    """Mean MSE over a batch of M tasks."""
    context_sa, context_ns, query_states, query_actions = batch
    losses = jax.vmap(lambda a, b, c, d: task_mse(model, a, b, c, d))(
        context_sa, context_ns, query_states, query_actions)
    return jnp.mean(losses)


def evaluate(model, data_loader, M, K, H, num_batches):
    total = 0.0
    for _ in range(num_batches):
        batch = data_loader.sample("test", M=M, K=K, H=H)
        loss, _ = batch_loss(model, batch)
        total += float(loss)
    return total / num_batches


def run_training(cfg: dict, data_dir: str, output_dir: str, seed: int = 42, device: str = "cpu"):
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    data_loader = IMuJoCoImitation(
        data_dir=data_dir, env_name=data_cfg["env_name"],
        train_perc=data_cfg.get("train_perc", 0.8), seed=data_cfg.get("split_seed", 42),
        normalize=data_cfg.get("normalize", True),
    )
    obs_size, act_size = data_loader.obs_size, data_loader.act_size

    M = int(train_cfg.get("M", 16))
    K_options = k_options(train_cfg.get("K", [1, 3, 5]))
    H = int(train_cfg.get("H", 100))
    num_iterations = int(train_cfg.get("num_iterations", 10000))
    eval_interval = int(train_cfg.get("eval_every", 500))
    num_eval_batches = int(train_cfg.get("num_eval_batches", 10))
    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    num_runs = int(train_cfg.get("num_runs", 1))

    print("=" * 60)
    print("HalfCheetah-v3 SetONet Pretraining")
    print("=" * 60)
    print(f"obs_size={obs_size}, act_size={act_size}")
    print(f"M={M}, K={K_options} episodes, H={H}, iterations={num_iterations}, lr={learning_rate}")
    print(f"num_runs={num_runs}")
    print("=" * 60)

    best_model, best_eval = None, float("inf")
    all_train, all_eval = [], []

    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        np.random.seed(run_seed)
        key = jr.PRNGKey(run_seed)
        print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_seed}) ---")

        model = build_setonet(model_cfg, obs_size, act_size, key)
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate, decay_steps=num_iterations,
            alpha=float(train_cfg.get("cosine_decay_alpha", 0.01)),
        )
        optim = optax.adam(schedule)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        train_losses, eval_losses = [], []
        for it in range(num_iterations):
            K = int(np.random.choice(K_options))
            batch = data_loader.sample("train", M=M, K=K, H=H)
            loss, model, opt_state = train_step(batch_loss, model, optim, opt_state, batch)
            train_losses.append(float(loss))
            if it % eval_interval == 0:
                ev = evaluate(model, data_loader, M, max(K_options), H, num_eval_batches)
                eval_losses.append(ev)
                print(f"  Iter {it:5d} | Train: {float(loss):.6f} | Eval: {ev:.6f}")

        all_train.append(train_losses)
        all_eval.append(eval_losses)
        run_best = min(eval_losses) if eval_losses else float("inf")
        if run_best < best_eval:
            best_eval, best_model = run_best, model

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(out / "setonet.eqx", best_model)
    np.savez(
        out / "training_history.npz",
        train_losses=np.array(all_train), eval_losses=np.array(all_eval),
        eval_iterations=np.arange(0, num_iterations, eval_interval), seed=seed,
        state_mean=data_loader.state_mean, state_std=data_loader.state_std,
        action_mean=data_loader.action_mean, action_std=data_loader.action_std,
    )
    with open(out / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\nBest eval loss: {best_eval:.6f}\nModel saved to {out / 'setonet.eqx'}")
    return best_model, all_train, all_eval
