"""SetONet meta-training for Obstacle Avoidance.

Branch input: obstacle (x,y) positions -> radii (encodes obstacle layout)
Trunk input: (state + time) -> predicted control
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path

from src.setonet import SetONet
from src.envs.dataloader import ObstacleAvoidanceImitation
from src.training.setonet_meta import (
    get_inner_update, meta_task_loss, meta_train_step,
)


def meta_sample_batch(data_loader, data_type, num_tasks, K):
    """Sample meta-batch: support/query from same obstacle count group.

    Uses data_loader.sample() which groups tasks by obstacle count
    so all M tasks have the same obstacle array shape.
    """
    # sample() returns (states, time, actions, obstacles) with shape (M, K, ...)
    # All M tasks have the same num_obs since they're from the same count group
    support_batch = data_loader.sample(type_=data_type, M=num_tasks, K=K)
    query_batch = data_loader.sample(type_=data_type, M=num_tasks, K=K)
    return support_batch, query_batch


def compute_setonet_loss(model, batch):
    """SetONet loss: branch encodes obstacles, trunk predicts controls.

    batch = (states, time, actions, obstacles)
    """
    states, time, actions, obstacles = batch

    # Drop last timestep to align with actions
    states = states[:, :-1, :]
    time = jnp.expand_dims(time[:, :-1], axis=-1)
    state_time = jnp.concatenate([states, time], axis=-1)

    obs_positions = obstacles[:, :2]     # (num_obs, 2) — branch input
    obs_values = obstacles[:, 2:3]       # (num_obs, 1) — branch output (radii)

    # Predict: vmap over K trajectories, then over N timesteps
    pred = jax.vmap(
        jax.vmap(model, in_axes=(None, None, 0)),
        in_axes=(None, None, 0),
    )(obs_positions, obs_values, state_time)

    return jnp.mean(jnp.square(actions - pred))


def batch_meta_loss(model, batch, alpha, inner_update_fn):
    support_batch, query_batch = batch
    num_tasks = support_batch[0].shape[0]

    losses = []
    for i in range(num_tasks):
        sup = tuple(b[i] for b in support_batch)
        qry = tuple(b[i] for b in query_batch)
        loss = meta_task_loss(model, sup, qry, alpha,
                              compute_setonet_loss, inner_update_fn)
        losses.append(loss)

    return jnp.mean(jnp.array(losses))


def evaluate(model, data_loader, num_tasks, K, alpha,
             inner_update_fn, num_batches=10):
    total = 0.0
    for _ in range(num_batches):
        batch = meta_sample_batch(data_loader, "test", num_tasks, K)
        loss = batch_meta_loss(model, batch, alpha, inner_update_fn)
        total += float(loss)
    return total / num_batches


def run_training(cfg, data_dir, output_dir, variant, seed=42, device="cpu"):
    np.random.seed(seed)
    key = jr.PRNGKey(seed)

    meta_cfg = cfg.get("setonet_meta", {})
    model_cfg = cfg.get("model", {})
    inner_update_fn = get_inner_update(variant)

    # Load data
    data_path = str(Path(data_dir) / "trajectories.npy")
    print(f"Loading data from {data_path}...")
    expert_data = np.load(data_path, allow_pickle=True).item()

    split_seed = meta_cfg.get("split_seed", 42)
    data_loader = ObstacleAvoidanceImitation(
        expert_data, train_perc=0.9, normalize=True, seed=split_seed,
    )
    print(f"Train tasks: {len(data_loader.train_data)}, Test tasks: {len(data_loader.test_data)}")

    state_dim = 4
    action_dim = 2

    inner_lr = meta_cfg.get("inner_lr", 0.01)
    outer_lr = meta_cfg.get("outer_lr", 0.001)
    K = meta_cfg.get("K", 10)
    num_tasks = meta_cfg.get("num_tasks", 16)
    num_iterations = meta_cfg.get("num_iterations", 2000)
    eval_interval = meta_cfg.get("eval_every", 100)
    num_eval_batches = meta_cfg.get("num_eval_batches", 10)

    print("=" * 60)
    print(f"SetONet Meta-Training — Obstacle ({variant})")
    print("=" * 60)
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Variant: {variant}")
    print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
    print(f"K (trajectories): {K}")
    print(f"Tasks per batch: {num_tasks}")
    print(f"Iterations: {num_iterations}")
    print("=" * 60)

    key, model_key = jr.split(key)
    model = SetONet(
        input_size_src=2,          # obstacle (x, y)
        output_size_src=1,         # obstacle radius
        input_size_tgt=state_dim + 1,  # state + time
        output_size_tgt=action_dim,
        p=model_cfg.get("p", 128),
        phi_hidden_size=model_cfg.get("phi_hidden_size", 128),
        phi_output_size=model_cfg.get("phi_output_size", 128),
        rho_hidden_size=model_cfg.get("rho_hidden_size", 128),
        trunk_hidden_size=model_cfg.get("trunk_hidden_size", 128),
        n_phi_layers=model_cfg.get("n_phi_layers", 2),
        n_rho_layers=model_cfg.get("n_rho_layers", 2),
        n_trunk_layers=model_cfg.get("n_trunk_layers", 2),
        aggregation_type=model_cfg.get("aggregation_type", "attention"),
        attention_n_heads=model_cfg.get("attention_n_heads", 4),
        attention_n_tokens=model_cfg.get("attention_n_tokens", 4),
        use_bias=model_cfg.get("use_bias", True),
        key=model_key,
    )

    pretrained_path = Path(output_dir).parent / "pretrained" / "setonet.eqx"
    if pretrained_path.exists():
        model = eqx.tree_deserialise_leaves(pretrained_path, model)
        print(f"Loaded pretrained weights from {pretrained_path}")
    else:
        print("No pretrained checkpoint found, training from scratch")

    optim = optax.adam(outer_lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    def batch_loss_fn(model, batch, alpha):
        return batch_meta_loss(model, batch, alpha, inner_update_fn)

    train_losses = []
    eval_losses = []

    print("\nStarting meta-training...")

    for iteration in range(num_iterations):
        batch = meta_sample_batch(data_loader, "train", num_tasks, K)
        loss, model, opt_state = meta_train_step(
            model, optim, opt_state, batch, inner_lr, batch_loss_fn
        )
        train_losses.append(float(loss))

        if iteration % eval_interval == 0:
            eval_loss = evaluate(
                model, data_loader, num_tasks, K,
                inner_lr, inner_update_fn, num_eval_batches
            )
            eval_losses.append(eval_loss)
            print(f"  Iter {iteration:5d} | Train: {loss:.6f} | Eval: {eval_loss:.6f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(out / "setonet.eqx", model)
    print(f"\nModel saved to {out / 'setonet.eqx'}")

    np.savez(
        out / "training_history.npz",
        train_losses=np.array(train_losses),
        eval_losses=np.array(eval_losses),
        eval_iterations=np.arange(0, num_iterations, eval_interval)[:len(eval_losses)],
        variant=variant,
    )
    print(f"Training history saved to {out / 'training_history.npz'}")
