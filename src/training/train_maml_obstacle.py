"""MAML baseline for Obstacle Avoidance.

Unlike other environments, the MAML obstacle model takes (state + flattened obstacle positions)
as input — the obstacle layout IS the task, so it must be provided as input for the MLP
to adapt to via inner gradient steps.

Support set: flattened (state, obstacles) -> action samples from trajectories.
Query set: same structure.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path

from src.envs.dataloader import ObstacleAvoidanceImitation
from src.training.maml import MAMLMLP, train_step


MAX_OBSTACLES = 6  # Pad all tasks to this many obstacles


def pad_obstacles(obstacles, max_obs=MAX_OBSTACLES):
    """Pad obstacle positions to fixed size. Only keeps (x, y), ignores radius."""
    positions = np.array(obstacles)[:, :2]
    num_obs = positions.shape[0]
    if num_obs < max_obs:
        padding = np.zeros((max_obs - num_obs, 2))
        return np.concatenate([positions, padding], axis=0)
    return positions[:max_obs]


def sample_meta_batch(data_loader, data_type, num_tasks, K_support, K_query, max_obstacles):
    """Sample meta-batch with obstacle context concatenated to state."""
    support_inputs_list = []
    support_actions_list = []
    query_inputs_list = []
    query_actions_list = []

    for _ in range(num_tasks):
        # Support
        states_s, time_s, actions_s, obstacles_s = data_loader.get_task(data_type, K=K_support)
        # Query (from same task type, not guaranteed same task — MAML adapts)
        states_q, time_q, actions_q, obstacles_q = data_loader.get_task(data_type, K=K_query)

        # Flatten: states (K, timesteps, state_dim) -> (K*timesteps, state_dim)
        # Drop last timestep to align with actions
        s_states = np.array(states_s[:, :-1, :]).reshape(-1, states_s.shape[-1])
        s_actions = np.array(actions_s).reshape(-1, actions_s.shape[-1])

        # Pad obstacles and tile for each sample
        obs_padded = pad_obstacles(np.array(obstacles_s), max_obstacles)
        obs_flat = obs_padded.reshape(-1)  # (max_obs * 2,)
        obs_tiled = np.tile(obs_flat[None, :], (s_states.shape[0], 1))

        # Concatenate state + obstacle positions
        support_inputs = np.concatenate([s_states, obs_tiled], axis=-1)
        support_inputs_list.append(support_inputs)
        support_actions_list.append(s_actions)

        # Query
        q_states = np.array(states_q[:, :-1, :]).reshape(-1, states_q.shape[-1])
        q_actions = np.array(actions_q).reshape(-1, actions_q.shape[-1])

        obs_padded_q = pad_obstacles(np.array(obstacles_q), max_obstacles)
        obs_flat_q = obs_padded_q.reshape(-1)
        obs_tiled_q = np.tile(obs_flat_q[None, :], (q_states.shape[0], 1))

        query_inputs = np.concatenate([q_states, obs_tiled_q], axis=-1)
        query_inputs_list.append(query_inputs)
        query_actions_list.append(q_actions)

    return (
        jnp.stack(support_inputs_list),
        jnp.stack(support_actions_list),
        jnp.stack(query_inputs_list),
        jnp.stack(query_actions_list),
    )


def compute_loss(model, inputs, actions):
    """MSE loss on flattened (state+obstacles, action) pairs."""
    pred = jax.vmap(model)(inputs)
    return jnp.mean(jnp.square(pred - actions))


def inner_update(model, support_inputs, support_actions, alpha):
    """Inner loop: one gradient step."""
    _, grads = eqx.filter_value_and_grad(compute_loss)(
        model, support_inputs, support_actions
    )
    updates = jax.tree_util.tree_map(lambda g: -alpha * g, grads)
    return eqx.apply_updates(model, updates)


def batch_maml_loss(model, batch, alpha):
    """MAML loss over batch of tasks."""
    support_inputs, support_actions, query_inputs, query_actions = batch

    def task_loss(s_in, s_a, q_in, q_a):
        adapted = inner_update(model, s_in, s_a, alpha)
        return compute_loss(adapted, q_in, q_a)

    losses = jax.vmap(task_loss)(
        support_inputs, support_actions, query_inputs, query_actions
    )
    return jnp.mean(losses)


def evaluate(model, data_loader, num_tasks, K_support, K_query, alpha,
             max_obstacles, num_batches=10):
    """Evaluate MAML on meta-test tasks."""
    total_loss = 0.0
    for _ in range(num_batches):
        batch = sample_meta_batch(
            data_loader, "test", num_tasks, K_support, K_query, max_obstacles
        )
        loss = batch_maml_loss(model, batch, alpha)
        total_loss += float(loss)
    return total_loss / num_batches


def run_training(cfg, data_dir, output_dir, seed=42, device="cpu"):
    """Train MAML on Obstacle Avoidance environment."""
    np.random.seed(seed)
    key = jr.PRNGKey(seed)

    maml_cfg = cfg.get("maml", {})
    model_cfg = cfg.get("maml_model", {})

    # Load dataset
    data_path = str(Path(data_dir) / "trajectories.npy")
    print(f"Loading data from {data_path}...")
    expert_data = np.load(data_path, allow_pickle=True).item()

    split_seed = maml_cfg.get("split_seed", 42)
    data_loader = ObstacleAvoidanceImitation(
        expert_data,
        train_perc=0.9,
        normalize=True,
        seed=split_seed,
    )
    print(f"Train tasks: {len(data_loader.train_data)}, Test tasks: {len(data_loader.test_data)}")

    state_dim = 4
    action_dim = 2
    max_obstacles = MAX_OBSTACLES

    # MAML params
    inner_lr = maml_cfg.get("inner_lr", 0.01)
    outer_lr = maml_cfg.get("outer_lr", 0.001)
    K_support = maml_cfg.get("K_support", 8)
    K_query = maml_cfg.get("K_query", 8)
    num_tasks = maml_cfg.get("num_tasks", 16)
    num_iterations = maml_cfg.get("num_iterations", 2000)
    eval_interval = maml_cfg.get("eval_every", 100)
    num_eval_batches = maml_cfg.get("num_eval_batches", 10)

    hidden_size = model_cfg.get("hidden_size", 128)
    num_layers = model_cfg.get("num_layers", 22)

    input_dim = state_dim + (max_obstacles * 2)

    print("=" * 60)
    print("MAML Training — Obstacle Avoidance")
    print("=" * 60)
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Max obstacles: {max_obstacles}, Input dim: {input_dim}")
    print(f"Inner LR: {inner_lr}, Outer LR: {outer_lr}")
    print(f"K_support: {K_support}, K_query: {K_query}")
    print(f"Tasks per batch: {num_tasks}")
    print(f"Model: MLP({input_dim} -> {action_dim}, hidden={hidden_size}, layers={num_layers})")
    print(f"Iterations: {num_iterations}")
    print("=" * 60)

    # Initialize model
    key, model_key = jr.split(key)
    model = MAMLMLP(
        input_dim=input_dim,
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
        batch = sample_meta_batch(
            data_loader, "train", num_tasks, K_support, K_query, max_obstacles
        )
        loss, model, opt_state = train_step(
            model, optim, opt_state, batch, inner_lr, batch_maml_loss
        )
        train_losses.append(float(loss))

        if iteration % eval_interval == 0:
            eval_loss = evaluate(
                model, data_loader, num_tasks, K_support, K_query,
                inner_lr, max_obstacles, num_eval_batches
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
