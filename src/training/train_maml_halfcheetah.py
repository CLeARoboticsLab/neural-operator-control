"""MAML baseline on HalfCheetah-v3 (iMuJoCo).

A monolithic MLP policy ``state -> action`` trained with the same bi-level
objective as SetONet-Meta-Full: one SGD inner step on the support episodes of a
task, Adam outer step on its query episodes. The MLP receives no context set.
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
from src.training.halfcheetah_common import build_maml, k_options
from src.training.setonet_meta import meta_train_step


def mlp_loss(model, states, actions):
    pred = jax.vmap(model)(states)
    return jnp.mean(jnp.square(pred - actions))


def maml_task_loss(model, sup_s, sup_a, qry_s, qry_a, alpha):
    _, grads = eqx.filter_value_and_grad(lambda m: mlp_loss(m, sup_s, sup_a))(model)
    adapted = eqx.apply_updates(model, jax.tree_util.tree_map(lambda g: -alpha * g, grads))
    return mlp_loss(adapted, qry_s, qry_a)


def sample_maml_batch(data_loader, type_, num_tasks, K, H):
    """Support = (states, actions) of the K context episodes; query = the K query episodes."""
    ctx_sa, _ctx_ns, q_s, q_a = data_loader.sample(type_, M=num_tasks, K=K, H=H)
    obs = data_loader.obs_size
    return ctx_sa[:, :, :obs], ctx_sa[:, :, obs:], q_s, q_a


def batch_maml_loss(model, batch, alpha):
    sup_s, sup_a, qry_s, qry_a = batch
    losses = [maml_task_loss(model, sup_s[i], sup_a[i], qry_s[i], qry_a[i], alpha)
              for i in range(sup_s.shape[0])]
    return jnp.mean(jnp.array(losses))


def evaluate(model, data_loader, num_tasks, K, H, alpha, num_batches):
    total = 0.0
    for _ in range(num_batches):
        batch = sample_maml_batch(data_loader, "test", num_tasks, K, H)
        total += float(batch_maml_loss(model, batch, alpha))
    return total / num_batches


def run_training(cfg, data_dir, output_dir, seed=42, device="cpu"):
    data_cfg = cfg["data"]
    maml_cfg = cfg.get("maml", {})

    data_loader = IMuJoCoImitation(
        data_dir=data_dir, env_name=data_cfg["env_name"],
        train_perc=data_cfg.get("train_perc", 0.8), seed=data_cfg.get("split_seed", 42),
        normalize=data_cfg.get("normalize", True),
    )
    obs_size, act_size = data_loader.obs_size, data_loader.act_size

    inner_lr = float(maml_cfg.get("inner_lr", 0.01))
    outer_lr = float(maml_cfg.get("outer_lr", 5e-4))
    K_options = k_options(maml_cfg.get("K", [1, 3, 5]))
    H = int(maml_cfg.get("H", 100))
    num_tasks = int(maml_cfg.get("num_tasks", 16))
    num_iterations = int(maml_cfg.get("num_iterations", 10000))
    eval_interval = int(maml_cfg.get("eval_every", 500))
    num_eval_batches = int(maml_cfg.get("num_eval_batches", 10))
    num_runs = int(maml_cfg.get("num_runs", 1))
    mm = cfg.get("maml_model", {})

    print("=" * 60)
    print("HalfCheetah-v3 MAML Baseline Training")
    print("=" * 60)
    print(f"Model: MLP({obs_size} -> {act_size}, hidden={mm.get('hidden_size', 256)}, "
          f"layers={mm.get('num_layers', 4)})")
    print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}, K={K_options}, H={H}")
    print(f"Tasks per batch: {num_tasks}, iterations: {num_iterations}, num_runs={num_runs}")
    print("=" * 60)

    best_model, best_eval = None, float("inf")
    all_train, all_eval = [], []

    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        np.random.seed(run_seed)
        key = jr.PRNGKey(run_seed)
        print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_seed}) ---")

        model = build_maml(cfg, obs_size, act_size, key)
        optim = optax.adam(outer_lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        train_losses, eval_losses = [], []
        for it in range(num_iterations):
            K = int(np.random.choice(K_options))
            batch = sample_maml_batch(data_loader, "train", num_tasks, K, H)
            loss, model, opt_state = meta_train_step(
                model, optim, opt_state, batch, inner_lr, batch_maml_loss)
            train_losses.append(float(loss))
            if it % eval_interval == 0:
                ev = evaluate(model, data_loader, num_tasks, max(K_options), H,
                              inner_lr, num_eval_batches)
                eval_losses.append(ev)
                print(f"  Iter {it:5d} | Train: {float(loss):.6f} | Eval: {ev:.6f}")

        all_train.append(train_losses)
        all_eval.append(eval_losses)
        run_best = min(eval_losses) if eval_losses else float("inf")
        if run_best < best_eval:
            best_eval, best_model = run_best, model

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(out / "maml.eqx", best_model)
    np.savez(
        out / "training_history.npz",
        train_losses=np.array(all_train), eval_losses=np.array(all_eval),
        eval_iterations=np.arange(0, num_iterations, eval_interval), seed=seed,
    )
    with open(out / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\nBest eval loss: {best_eval:.6f}\nModel saved to {out / 'maml.eqx'}")
