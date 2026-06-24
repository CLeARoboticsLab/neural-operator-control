"""MAML baseline for Quadrotor (varying dynamics, fixed goal).

Same structure as p2p_dynamics MAML — both support and query are expert
trajectories with (state+time) -> action mapping.
Uses QuadrotorDataLoader for data loading (same as SetONet quadrotor trainer).
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path

from src.training.train_quadrotor import QuadrotorDataLoader
from src.training.maml import (
    MAMLMLP, compute_loss_trajectories,
    maml_task_loss, train_step,
)


def sample_meta_batch(data_loader, data_type, num_tasks, K_support, K_query):
    """Sample meta-batch: support and query are (state+time, action) from expert trajectories."""
    support_states_list = []
    support_actions_list = []
    query_states_list = []
    query_actions_list = []

    for _ in range(num_tasks):
        # Sample support and query separately from the same task type
        s_data = data_loader.sample_task(data_type, K=K_support, N=1)
        q_data = data_loader.sample_task(data_type, K=K_query, N=1)

        # s_data = (random_states, random_actions, random_next_states,
        #           expert_trajectories, expert_actions)
        # expert_trajectories: (N, horizon+1, state_dim+1) with time
        # For MAML we use the random samples as flattened (state+time, action) pairs

        # Support: use random (state, action) samples reshaped as trajectory-like
        # Actually, the dataloader gives us K transitions — reshape to trajectory format
        s_states, s_actions, s_next, s_trajs, s_acts = s_data
        # s_trajs: (1, horizon+1, state_dim+1), s_acts: (1, horizon, action_dim)
        # But K transitions give us more diverse coverage, so use those
        # Add time dimension (use 0 since these are individual samples)
        time_col = np.zeros((s_states.shape[0], 1))
        support_states_with_time = np.concatenate([s_states, time_col], axis=-1)

        support_states_list.append(support_states_with_time)
        support_actions_list.append(s_actions)

        # Query: expert trajectories
        q_states, q_actions, q_next, q_trajs, q_acts = q_data
        time_col_q = np.zeros((q_states.shape[0], 1))
        query_states_with_time = np.concatenate([q_states, time_col_q], axis=-1)

        query_states_list.append(query_states_with_time)
        query_actions_list.append(q_actions)

    return (
        jnp.array(np.stack(support_states_list)),
        jnp.array(np.stack(support_actions_list)),
        jnp.array(np.stack(query_states_list)),
        jnp.array(np.stack(query_actions_list)),
    )


def compute_loss_flat(model, states, actions):
    """MSE loss on flat (state+time, action) pairs."""
    pred = jax.vmap(model)(states)
    return jnp.mean(jnp.square(pred - actions))


def batch_maml_loss(model, batch, alpha):
    """MAML loss over batch of tasks."""
    support_states, support_actions, query_states, query_actions = batch

    def task_loss(s_s, s_a, q_s, q_a):
        return maml_task_loss(
            model, s_s, s_a, q_s, q_a, alpha,
            support_loss_fn=compute_loss_flat,
            query_loss_fn=compute_loss_flat,
        )

    losses = jax.vmap(task_loss)(
        support_states, support_actions, query_states, query_actions
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
    """Train MAML on Quadrotor environment."""
    np.random.seed(seed)
    key = jr.PRNGKey(seed)

    maml_cfg = cfg.get("maml", {})
    model_cfg = cfg.get("maml_model", {})

    # Load dataset
    data_path = str(Path(data_dir) / "trajectories.npz")
    print(f"Loading data from {data_path}...")
    raw_data = np.load(data_path, allow_pickle=True)
    dataset = {k: raw_data[k] for k in raw_data.files}

    # Filter for fixed goal
    goal_idx = maml_cfg.get("goal_idx", 0)
    data_loader = QuadrotorDataLoader.load_for_goal(goal_idx, dataset, train_perc=0.8)

    state_dim = dataset["states"].shape[-1]   # 6
    action_dim = dataset["actions"].shape[-1]  # 2

    # MAML params
    inner_lr = maml_cfg.get("inner_lr", 0.01)
    outer_lr = maml_cfg.get("outer_lr", 0.001)
    K_support = maml_cfg.get("K_support", 64)
    K_query = maml_cfg.get("K_query", 32)
    num_tasks = maml_cfg.get("num_tasks", 16)
    num_iterations = maml_cfg.get("num_iterations", 1000)
    eval_interval = maml_cfg.get("eval_every", 100)
    num_eval_batches = maml_cfg.get("num_eval_batches", 10)

    hidden_size = model_cfg.get("hidden_size", 128)
    num_layers = model_cfg.get("num_layers", 17)

    print("=" * 60)
    print("MAML Training — Quadrotor")
    print("=" * 60)
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
    print(f"K_support: {K_support}, K_query: {K_query}")
    print(f"Tasks per batch: {num_tasks}")
    print(f"Model: MLP({state_dim + 1} -> {action_dim}, hidden={hidden_size}, layers={num_layers})")
    print(f"Iterations: {num_iterations}")
    print("=" * 60)

    # Initialize model
    key, model_key = jr.split(key)
    model = MAMLMLP(
        input_dim=state_dim + 1,  # state + time
        output_dim=action_dim,
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
