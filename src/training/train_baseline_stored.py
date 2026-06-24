"""Shared BC training for the stored-dataset MLP baselines.

P2P-Dynamics, Quadrotor and Obstacle all reduce to the same problem once their
task parameters are extracted (see ``baseline_data.py``): fit an MLP
``[state, time, task_params] -> action`` on the flattened expert transitions of
the training-split tasks. The per-environment trainers are thin wrappers that
pick the right extractor and call :func:`fit`.

The MLP standardizes its ``[state, time]`` input block and its action targets
using statistics computed on the training data (saved to the checkpoint so the
evaluator reproduces them exactly). Task parameters arrive pre-scaled from the
extractor, so they are passed through unchanged.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path

from src.training.baseline_mlp import BaselineMLP, train_step, count_params


def build_flat(tasks, state_dim):
    """Stack all tasks' transitions into flat (X, Y) with normalized time.

    X row = [state(state_dim), time(1), param(P)] (raw state, scaled param);
    Y row = action. Returns float32 numpy arrays.
    """
    xs, ys = [], []
    for t in tasks:
        s = t["states"]                      # (n, H, state_dim)
        a = t["actions"]                     # (n, H, action_dim)
        n, h = s.shape[0], s.shape[1]
        time = np.linspace(0.0, 1.0, h, endpoint=False)[None, :, None]
        time = np.broadcast_to(time, (n, h, 1))
        param = np.broadcast_to(t["param"][None, None, :], (n, h, t["param"].shape[0]))
        x = np.concatenate([s, time, param], axis=-1).reshape(-1, state_dim + 1 + t["param"].shape[0])
        y = a.reshape(-1, a.shape[-1])
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32)


def fit(cfg, train_tasks, info, output_dir, seed, split_seed, env_name):
    """Train the task-conditioned MLP on flattened stored-dataset transitions."""
    state_dim = info["state_dim"]
    action_dim = info["action_dim"]
    param_dim = info["param_dim"]

    train_cfg = cfg.get("training", {})
    # A trainable, parameter-comparable MLP (shallow + wide). Falls back to
    # maml_model only if no baseline_model block is present.
    mlp_cfg = cfg.get("baseline_model", cfg.get("maml_model", {}))
    hidden_size = mlp_cfg.get("hidden_size", 256)
    num_layers = mlp_cfg.get("num_layers", 4)
    num_iterations = train_cfg.get("num_iterations", 2000)
    learning_rate = train_cfg.get("learning_rate", 1e-3)
    eval_interval = train_cfg.get("eval_every", 100)
    batch_size = int(train_cfg.get("baseline_batch_size", 512))

    np.random.seed(seed)
    X, Y = build_flat(train_tasks, state_dim)

    # Standardize the [state, time] block (first state_dim+1 cols) and the targets.
    st = state_dim + 1
    st_mean = X[:, :st].mean(axis=0)
    st_std = X[:, :st].std(axis=0) + 1e-8
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0) + 1e-8

    Xn = X.copy()
    Xn[:, :st] = (Xn[:, :st] - st_mean) / st_std
    Yn = (Y - y_mean) / y_std

    Xn = jnp.asarray(Xn)
    Yn = jnp.asarray(Yn)
    n_samples = Xn.shape[0]
    input_dim = state_dim + 1 + param_dim

    print("=" * 60)
    print(f"{env_name} — Task-Conditioned MLP Baseline")
    print("=" * 60)
    print(f"Seed: {seed} | split_seed: {split_seed}")
    print(f"Train tasks: {len(train_tasks)} | transitions: {n_samples:,}")
    print(f"Model: MLP({input_dim} -> {action_dim}, width={hidden_size}, depth={num_layers})")
    print(f"Iterations: {num_iterations} | batch: {batch_size} | LR: {learning_rate}")
    print("=" * 60)

    key = jr.PRNGKey(seed)
    key, model_key = jr.split(key)
    model = BaselineMLP(input_dim, action_dim, hidden_size, num_layers, model_key)
    print(f"Parameter count: {count_params(model):,}")

    optim = optax.adam(learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    train_losses = []
    print("\nStarting training...")
    for iteration in range(num_iterations):
        idx = np.random.randint(0, n_samples, size=batch_size)
        xb = Xn[idx]
        yb = Yn[idx]
        loss, model, opt_state = train_step(model, optim, opt_state, xb, yb)
        train_losses.append(float(loss))
        if iteration % eval_interval == 0:
            print(f"  Iter {iteration:5d} | Train: {loss:.6f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(output_path / "baseline.eqx", model)
    print(f"\nModel saved to {output_path / 'baseline.eqx'}")

    np.savez(
        output_path / "training_history.npz",
        train_losses=np.array(train_losses),
        seed=seed,
        split_seed=split_seed,
        st_mean=st_mean, st_std=st_std,
        y_mean=y_mean, y_std=y_std,
        state_dim=state_dim, action_dim=action_dim, param_dim=param_dim,
        hidden_size=hidden_size, num_layers=num_layers,
    )
    print(f"Training history saved to {output_path / 'training_history.npz'}")
    return model
