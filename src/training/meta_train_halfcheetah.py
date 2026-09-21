"""SetONet meta-training on HalfCheetah-v3 (iMuJoCo).

Variants (selected with ``--variant``):
  meta_branch: SetONet-Meta — inner loop adapts the branch only (trunk frozen)
  meta_full:   SetONet-Meta-Full — inner loop adapts all parameters

For one task the support set is the K context episodes themselves: the inner
step fits the operator to their (state -> action) pairs while conditioning on
their transitions; the outer loss scores the adapted operator on K query episodes.
"""

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import yaml

from src.envs.imujoco_dataloader import IMuJoCoImitation
from src.training.halfcheetah_common import build_setonet, k_options, setonet_task_loss
from src.training.setonet_meta import get_inner_update, meta_task_loss, meta_train_step


def sample_meta_batch(data_loader, type_, num_tasks, K, H):
    """Returns ``(support_batch, query_batch)``; each is a tuple of stacked arrays.

    support = (ctx_sa, ctx_ns, ctx_states, ctx_actions)   -> inner-loop targets
    query   = (ctx_sa, ctx_ns, qry_states, qry_actions)   -> outer-loop targets
    """
    ctx_sa, ctx_ns, q_s, q_a = data_loader.sample(type_, M=num_tasks, K=K, H=H)
    obs = data_loader.obs_size
    support = (ctx_sa, ctx_ns, ctx_sa[:, :, :obs], ctx_sa[:, :, obs:])
    query = (ctx_sa, ctx_ns, q_s, q_a)
    return support, query


def batch_meta_loss(model, batch, alpha, inner_update_fn):
    support_batch, query_batch = batch
    num_tasks = support_batch[0].shape[0]
    losses = []
    for i in range(num_tasks):
        sup = tuple(b[i] for b in support_batch)
        qry = tuple(b[i] for b in query_batch)
        losses.append(meta_task_loss(model, sup, qry, alpha, setonet_task_loss, inner_update_fn))
    return jnp.mean(jnp.array(losses))


def evaluate(model, data_loader, num_tasks, K, H, alpha, inner_update_fn, num_batches):
    total = 0.0
    for _ in range(num_batches):
        batch = sample_meta_batch(data_loader, "test", num_tasks, K, H)
        total += float(batch_meta_loss(model, batch, alpha, inner_update_fn))
    return total / num_batches


def run_training(cfg, data_dir, output_dir, variant, seed=42, device="cpu"):
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    meta_cfg = cfg.get("setonet_meta", {})
    inner_update_fn = get_inner_update(variant)

    data_loader = IMuJoCoImitation(
        data_dir=data_dir, env_name=data_cfg["env_name"],
        train_perc=data_cfg.get("train_perc", 0.8), seed=data_cfg.get("split_seed", 42),
        normalize=data_cfg.get("normalize", True),
    )
    obs_size, act_size = data_loader.obs_size, data_loader.act_size

    inner_lr = float(meta_cfg.get("inner_lr", 0.01))
    outer_lr = float(meta_cfg.get("outer_lr", 5e-4))
    K_options = k_options(meta_cfg.get("K", [1, 3, 5]))
    H = int(meta_cfg.get("H", 100))
    num_tasks = int(meta_cfg.get("num_tasks", 16))
    num_iterations = int(meta_cfg.get("num_iterations", 10000))
    eval_interval = int(meta_cfg.get("eval_every", 500))
    num_eval_batches = int(meta_cfg.get("num_eval_batches", 10))
    num_runs = int(meta_cfg.get("num_runs_meta_full", 1) if variant == "meta_full"
                   else meta_cfg.get("num_runs", 1))

    print("=" * 60)
    print(f"HalfCheetah-v3 SetONet Meta-Training ({variant})")
    print("=" * 60)
    print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}, K={K_options}, H={H}")
    print(f"Tasks per batch: {num_tasks}, iterations: {num_iterations}, num_runs={num_runs}")
    print("=" * 60)

    pretrained_path = Path(output_dir).parent / "pretrained" / "setonet.eqx"
    use_pretrained = bool(meta_cfg.get("init_from_pretrained", False)) and pretrained_path.exists()

    best_model, best_eval = None, float("inf")
    all_train, all_eval = [], []

    def batch_loss_fn(model, batch, alpha):
        return batch_meta_loss(model, batch, alpha, inner_update_fn)

    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        np.random.seed(run_seed)
        key = jr.PRNGKey(run_seed)
        print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_seed}) ---")

        model = build_setonet(model_cfg, obs_size, act_size, key)
        if use_pretrained:
            model = eqx.tree_deserialise_leaves(pretrained_path, model)
            print(f"Initialised from {pretrained_path}")

        optim = optax.adam(outer_lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        train_losses, eval_losses = [], []
        for it in range(num_iterations):
            K = int(np.random.choice(K_options))
            batch = sample_meta_batch(data_loader, "train", num_tasks, K, H)
            loss, model, opt_state = meta_train_step(
                model, optim, opt_state, batch, inner_lr, batch_loss_fn)
            train_losses.append(float(loss))
            if it % eval_interval == 0:
                ev = evaluate(model, data_loader, num_tasks, max(K_options), H,
                              inner_lr, inner_update_fn, num_eval_batches)
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
        eval_iterations=np.arange(0, num_iterations, eval_interval), variant=variant, seed=seed,
    )
    with open(out / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\nBest eval loss: {best_eval:.6f}\nModel saved to {out / 'setonet.eqx'}")
