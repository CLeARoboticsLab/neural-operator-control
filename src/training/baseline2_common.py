"""Baseline 2: context-conditioned MLP (must *infer* the task, like SetONet).

Where Baseline 1 is handed the ground-truth task parameters, Baseline 2 receives
exactly the data SetONet puts in its branch — the (location, value) context set —
but consumes it as a flat concatenated vector instead of a permutation-invariant
set encoder:

    [state, time, flatten(context_set)] -> action

The context set is sampled the same way SetONet samples its branch (variable size
``N`` drawn from the env's K options), then written into a fixed bank of ``slots``
elements: the first ``N`` slots hold the (scaled) sampled context, the remaining
slots are padded with -1. So the MLP must (a) infer the task from raw context and
(b) cope with order-sensitivity and padding — precisely the structure the operator
architecture handles for free.

This module is env-agnostic: each environment supplies a ``sample_task`` callable
returning ``(context, queries, targets)`` for one task; everything else (scaling,
batching, width sizing, the training loop, checkpoint metadata) lives here.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path

from src.training.baseline_mlp import BaselineMLP, train_step, count_params, width_for_target


def pad_flatten_context(ctx, slots, ctx_scale, pad=-1.0):
    """Scale context to ~[-1,1] per feature, write into `slots` slots, pad with -1."""
    n, elem = ctx.shape
    out = np.full((slots, elem), pad, dtype=np.float32)
    k = min(n, slots)
    out[:k] = (ctx[:k] / ctx_scale).astype(np.float32)
    return out.reshape(-1)


def estimate_stats(sample_task, n_warmup):
    """Estimate context max-abs scales and (state+time)/target z-score stats."""
    ctxs, qs, ys = [], [], []
    for _ in range(n_warmup):
        ctx, queries, targets = sample_task()
        ctxs.append(np.asarray(ctx))
        qs.append(np.asarray(queries).reshape(-1, np.asarray(queries).shape[-1]))
        ys.append(np.asarray(targets).reshape(-1, np.asarray(targets).shape[-1]))
    C = np.concatenate(ctxs, 0)
    Q = np.concatenate(qs, 0)
    Y = np.concatenate(ys, 0)
    ctx_scale = np.maximum(np.abs(C).max(0), 1e-6).astype(np.float32)
    st_mean, st_std = Q.mean(0).astype(np.float32), (Q.std(0) + 1e-6).astype(np.float32)
    y_mean, y_std = Y.mean(0).astype(np.float32), (Y.std(0) + 1e-6).astype(np.float32)
    return ctx_scale, st_mean, st_std, y_mean, y_std


def build_input(queries, flat_ctx, st_mean, st_std):
    """Standardize [state,time] queries and concat the shared flat context."""
    q = np.asarray(queries).reshape(-1, np.asarray(queries).shape[-1])
    qn = (q - st_mean) / st_std
    fb = np.broadcast_to(flat_ctx[None, :], (qn.shape[0], flat_ctx.shape[0]))
    return np.concatenate([qn, fb], axis=-1)


def _build_batch(sample_task, M, slots, ctx_scale, st_mean, st_std, y_mean, y_std):
    Xs, Ys = [], []
    for _ in range(M):
        ctx, queries, targets = sample_task()
        flat = pad_flatten_context(np.asarray(ctx), slots, ctx_scale)
        Xs.append(build_input(queries, flat, st_mean, st_std))
        t = np.asarray(targets).reshape(-1, np.asarray(targets).shape[-1])
        Ys.append((t - y_mean) / y_std)
    return (np.concatenate(Xs, 0).astype(np.float32),
            np.concatenate(Ys, 0).astype(np.float32))


def fit_b2(cfg, sample_task, state_dim, action_dim, elem_dim, slots, n_ctx,
           output_dir, seed, env_name, target_params, extra_meta=None):
    """Estimate stats, size a param-comparable MLP, train, and checkpoint."""
    train_cfg = cfg.get("training", {})
    b2_cfg = cfg.get("baseline2_model", {})
    depth = b2_cfg.get("num_layers", 4)
    M = train_cfg.get("M", 16)
    num_iterations = train_cfg.get("num_iterations", 2000)
    lr = train_cfg.get("learning_rate", 1e-3)
    eval_every = train_cfg.get("eval_every", 100)

    np.random.seed(seed)
    ctx_scale, st_mean, st_std, y_mean, y_std = estimate_stats(sample_task, n_warmup=50)

    in_dim = state_dim + 1 + slots * elem_dim
    width = b2_cfg.get("hidden_size") or width_for_target(in_dim, action_dim, depth, target_params)

    print("=" * 60)
    print(f"{env_name} — Context-Conditioned MLP (Baseline 2)")
    print("=" * 60)
    print(f"Seed: {seed} | context slots={slots} (filled up to N~{n_ctx}), elem_dim={elem_dim}")
    print(f"Model: MLP({in_dim} -> {action_dim}, width={width}, depth={depth})")
    print(f"Iterations: {num_iterations} | M={M} | LR: {lr}")
    print("=" * 60)

    key = jr.PRNGKey(seed)
    key, mk = jr.split(key)
    model = BaselineMLP(in_dim, action_dim, width, depth, mk)
    print(f"Parameter count: {count_params(model):,}  (SetONet target ≈ {target_params:,})")

    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    train_losses = []
    print("\nStarting training...")
    for it in range(num_iterations):
        X, Y = _build_batch(sample_task, M, slots, ctx_scale, st_mean, st_std, y_mean, y_std)
        loss, model, opt_state = train_step(model, optim, opt_state, jnp.asarray(X), jnp.asarray(Y))
        train_losses.append(float(loss))
        if it % eval_every == 0:
            print(f"  Iter {it:5d} | Train: {loss:.6f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(out / "baseline2.eqx", model)
    meta = dict(
        train_losses=np.array(train_losses), seed=seed,
        ctx_scale=ctx_scale, st_mean=st_mean, st_std=st_std, y_mean=y_mean, y_std=y_std,
        slots=slots, elem_dim=elem_dim, n_ctx=n_ctx,
        state_dim=state_dim, action_dim=action_dim, width=width, depth=depth,
    )
    if extra_meta:
        meta.update(extra_meta)
    np.savez(out / "training_history.npz", **meta)
    print(f"\nModel saved to {out / 'baseline2.eqx'}")
    return model
