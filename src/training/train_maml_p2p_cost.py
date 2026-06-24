"""MAML baseline for P2P-Cost (Double Integrator LQR).

Support set: individual (state+time, control) pairs from source transitions.
Query set: expert trajectories (state+time -> control).
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path

from src.envs.dataloader import DoubleIntegratorLQRData
from src.training.maml import (
    MAMLMLP, compute_loss_flat, compute_loss_trajectories,
    inner_update, maml_task_loss, train_step,
)


def sample_meta_batch(data_loader, data_type, num_tasks, K_support, K_query):
    """Sample meta-batch: support = (state+time, control) pairs, query = trajectories."""
    support_states_list = []
    support_controls_list = []
    query_states_list = []
    query_controls_list = []

    for _ in range(num_tasks):
        task_data = data_loader.get_task(data_type, K=K_support, N=K_query)
        src_states, src_controls, src_costs, src_times, tgt_states, tgt_controls, goal = task_data

        # Support: (state+time, control) pairs
        src_states_with_time = np.concatenate([src_states, src_times], axis=-1)
        support_states_list.append(src_states_with_time)
        support_controls_list.append(src_controls)

        # Query: trajectories (already have time appended by dataloader)
        query_states_list.append(tgt_states)
        query_controls_list.append(tgt_controls)

    return (
        jnp.stack(support_states_list),
        jnp.stack(support_controls_list),
        jnp.stack(query_states_list),
        jnp.stack(query_controls_list),
    )


def batch_maml_loss(model, batch, alpha):
    """MAML loss over batch of tasks."""
    support_states, support_controls, query_states, query_controls = batch

    def task_loss(s_s, s_c, q_s, q_c):
        return maml_task_loss(
            model, s_s, s_c, q_s, q_c, alpha,
            support_loss_fn=compute_loss_flat,
            query_loss_fn=compute_loss_trajectories,
        )

    losses = jax.vmap(task_loss)(
        support_states, support_controls, query_states, query_controls
    )
    return jnp.mean(losses)


def evaluate(model, data_loader, num_tasks, K_support, K_query, alpha, num_batches=10):
    """Evaluate MAML on meta-test tasks."""
    total_loss = 0.0
    for _ in range(num_batches):
        batch = sample_meta_batch(data_loader, "test", num_tasks, K_support, K_query)
        loss = batch_maml_loss(model, batch, alpha)
        total_loss += float(loss)
    return total_loss / num_batches


def run_training(cfg, data_dir, output_dir, seed=42, device="cpu"):
    """Train MAML on P2P-Cost environment."""
    np.random.seed(seed)
    key = jr.PRNGKey(seed)

    maml_cfg = cfg.get("maml", {})
    model_cfg = cfg.get("maml_model", {})

    # Load dataset
    data_path = str(Path(data_dir) / "trajectories.npz")
    print(f"Loading data from {data_path}...")
    raw_data = np.load(data_path, allow_pickle=True)
    dataset = {key_: raw_data[key_] for key_ in raw_data.files}
    if "norm_stats" in dataset:
        dataset["norm_stats"] = dataset["norm_stats"].item()

    data_loader = DoubleIntegratorLQRData(dataset, train_perc=0.8, normalize=True)

    state_dim = dataset["states"].shape[-1]
    control_dim = dataset["actions"].shape[-1]

    # MAML params
    inner_lr = maml_cfg.get("inner_lr", 0.01)
    outer_lr = maml_cfg.get("outer_lr", 0.001)
    K_support = maml_cfg.get("K_support", 64)
    K_query = maml_cfg.get("K_query", 8)
    num_tasks = maml_cfg.get("num_tasks", 16)
    num_iterations = maml_cfg.get("num_iterations", 5000)
    eval_interval = maml_cfg.get("eval_every", 100)
    num_eval_batches = maml_cfg.get("num_eval_batches", 10)

    hidden_size = model_cfg.get("hidden_size", 128)
    num_layers = model_cfg.get("num_layers", 16)

    print("=" * 60)
    print("MAML Training — P2P-Cost")
    print("=" * 60)
    print(f"State dim: {state_dim}, Control dim: {control_dim}")
    print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
    print(f"K_support: {K_support}, K_query: {K_query}")
    print(f"Tasks per batch: {num_tasks}")
    print(f"Model: MLP({state_dim + 1} -> {control_dim}, hidden={hidden_size}, layers={num_layers})")
    print(f"Iterations: {num_iterations}")
    print("=" * 60)

    # Initialize model
    key, model_key = jr.split(key)
    model = MAMLMLP(
        input_dim=state_dim + 1,  # state + time
        output_dim=control_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        key=model_key,
    )

    optim = optax.adam(outer_lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    train_losses = []
    eval_losses = []

    print("\nStarting MAML training...")

    for iteration in range(num_iterations):
        batch = sample_meta_batch(data_loader, "train", num_tasks, K_support, K_query)
        loss, model, opt_state = train_step(
            model, optim, opt_state, batch, inner_lr, batch_maml_loss
        )
        train_losses.append(float(loss))

        if iteration % eval_interval == 0:
            eval_loss = evaluate(
                model, data_loader, num_tasks, K_support, K_query,
                inner_lr, num_eval_batches
            )
            eval_losses.append(eval_loss)
            print(f"  Iter {iteration:5d} | Train: {loss:.6f} | Eval: {eval_loss:.6f}")

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    eqx.tree_serialise_leaves(output_path / "maml.eqx", model)
    print(f"\nModel saved to {output_path / 'maml.eqx'}")

    np.savez(
        output_path / "training_history.npz",
        train_losses=np.array(train_losses),
        eval_losses=np.array(eval_losses),
        eval_iterations=np.arange(0, num_iterations, eval_interval)[:len(eval_losses)],
    )
    print(f"Training history saved to {output_path / 'training_history.npz'}")
