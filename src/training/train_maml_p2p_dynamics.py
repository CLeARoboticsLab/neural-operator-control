"""MAML baseline for P2P-Dynamics (varying dynamics, fixed goal).

Support set: expert trajectories (state+time -> action).
Query set: expert trajectories (state+time -> action).
Both support and query use full trajectory data, NOT random state-action pairs.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path

from src.envs.dataloader import VaryingDynamicsData
from src.training.maml import (
    MAMLMLP, compute_loss_trajectories,
    maml_task_loss, train_step,
)


def sample_meta_batch(data_set, data_type, num_tasks, N_support, N_query):
    """Sample meta-batch: both support and query are expert trajectories."""
    support_states_list = []
    support_actions_list = []
    query_states_list = []
    query_actions_list = []

    for _ in range(num_tasks):
        total_needed = N_support + N_query
        task_data = data_set.get_task(data_type, K=0, N=total_needed)
        (_, _, _, _,
         expert_trajectories, expert_actions, expert_goal_states, dynamics_params) = task_data

        # Split into support and query
        # expert_trajectories: (total, horizon+1, state_dim+1) with time
        support_trajs = expert_trajectories[:N_support, :-1, :]  # drop final timestep
        support_acts = expert_actions[:N_support]

        query_trajs = expert_trajectories[N_support:, :-1, :]
        query_acts = expert_actions[N_support:]

        support_states_list.append(support_trajs)
        support_actions_list.append(support_acts)
        query_states_list.append(query_trajs)
        query_actions_list.append(query_acts)

    return (
        jnp.stack(support_states_list),
        jnp.stack(support_actions_list),
        jnp.stack(query_states_list),
        jnp.stack(query_actions_list),
    )


def batch_maml_loss(model, batch, alpha):
    """MAML loss over batch of tasks."""
    support_states, support_actions, query_states, query_actions = batch

    def task_loss(s_s, s_a, q_s, q_a):
        return maml_task_loss(
            model, s_s, s_a, q_s, q_a, alpha,
            support_loss_fn=compute_loss_trajectories,
            query_loss_fn=compute_loss_trajectories,
        )

    losses = jax.vmap(task_loss)(
        support_states, support_actions, query_states, query_actions
    )
    return jnp.mean(losses)


def evaluate(model, data_set, num_tasks, N_support, N_query, alpha, num_batches=10):
    """Evaluate MAML on meta-test tasks."""
    total_loss = 0.0
    for _ in range(num_batches):
        batch = sample_meta_batch(data_set, "test", num_tasks, N_support, N_query)
        loss = batch_maml_loss(model, batch, alpha)
        total_loss += float(loss)
    return total_loss / num_batches


def run_training(cfg, data_dir, output_dir, seed=42, device="cpu"):
    """Train MAML on P2P-Dynamics environment."""
    np.random.seed(seed)
    key = jr.PRNGKey(seed)

    maml_cfg = cfg.get("maml", {})
    model_cfg = cfg.get("maml_model", {})

    # Load dataset
    data_path = str(Path(data_dir) / "trajectories.npz")
    print(f"Loading data from {data_path}...")
    raw_data = np.load(data_path, allow_pickle=True)
    dataset = {k: raw_data[k] for k in raw_data.files}
    if "dynamics_params" in dataset:
        dataset["dynamics_params"] = dataset["dynamics_params"].tolist() if hasattr(dataset["dynamics_params"], "tolist") else dataset["dynamics_params"]

    # Filter for single goal (goal_idx=0)
    goal_idx = maml_cfg.get("goal_idx", 0)
    data_set = VaryingDynamicsData.load_for_goal(goal_idx, dataset, train_perc=0.8)

    state_dim = dataset["states"].shape[-1]
    action_dim = dataset["actions"].shape[-1]

    # MAML params
    inner_lr = maml_cfg.get("inner_lr", 0.01)
    outer_lr = maml_cfg.get("outer_lr", 0.001)
    N_support = maml_cfg.get("N_support", 8)
    N_query = maml_cfg.get("N_query", 8)
    num_tasks = maml_cfg.get("num_tasks", 16)
    num_iterations = maml_cfg.get("num_iterations", 1500)
    eval_interval = maml_cfg.get("eval_every", 100)
    num_eval_batches = maml_cfg.get("num_eval_batches", 10)

    hidden_size = model_cfg.get("hidden_size", 128)
    num_layers = model_cfg.get("num_layers", 17)

    print("=" * 60)
    print("MAML Training — P2P-Dynamics")
    print("=" * 60)
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
    print(f"N_support: {N_support}, N_query: {N_query}")
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
        batch = sample_meta_batch(data_set, "train", num_tasks, N_support, N_query)
        loss, model, opt_state = train_step(
            model, optim, opt_state, batch, inner_lr, batch_maml_loss
        )
        train_losses.append(float(loss))

        if iteration % eval_interval == 0:
            eval_loss = evaluate(
                model, data_set, num_tasks, N_support, N_query,
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
