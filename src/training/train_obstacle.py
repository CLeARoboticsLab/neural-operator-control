"""
Train SetONet on Obstacle environment (varying obstacle configurations).

Each task is defined by an obstacle configuration (positions + radii).
- Branch input: obstacle positions (x, y) with radius values
- Trunk input: (state, time) along expert trajectories
- Output: optimal controls (acceleration)

K (number of context trajectories) is randomly varied during training
to help the model generalize across different amounts of context data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from pathlib import Path
from tqdm import tqdm

from src.setonet import SetONet
from src.envs.dataloader import ObstacleAvoidanceImitation


@eqx.filter_jit
def train_step(batch_loss_fn, model, optim, opt_state, batch):
    """Single training step."""
    loss, grads = batch_loss_fn(model, batch)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def batch_loss(model, batch):
    """Compute loss over a batch of M tasks.

    Args:
        batch: (states, time, actions, obstacles) stacked over M tasks
            states:    (M, K, timesteps, state_dim)
            time:      (M, K, timesteps)
            actions:   (M, K, timesteps, action_dim)
            obstacles: (M, num_obs, 3)  — (x, y, radius), same num_obs per batch
    """
    states, time, actions, obstacles = batch

    # Drop last timestep from states/time to align with actions (N+1 -> N)
    states = states[:, :, :-1, :]
    time = jnp.expand_dims(time[:, :, :-1], axis=-1)
    state_time = jnp.concatenate([states, time], axis=-1)  # (M, K, N, state_dim+1)

    obs_positions = obstacles[:, :, :2]   # (M, num_obs, 2)
    obs_values = obstacles[:, :, 2:3]     # (M, num_obs, 1)

    # vmap: M tasks -> K trajectories -> N timesteps
    actions_pred = jax.vmap(
        jax.vmap(
            jax.vmap(model, in_axes=(None, None, 0)),
            in_axes=(None, None, 0),
        )
    )(obs_positions, obs_values, state_time)

    return jnp.mean(jnp.square(actions - actions_pred))


def eval_at_k(model, data_set, K, M, num_batches=10):
    """Evaluate model at a specific K value."""
    total_loss = 0.0
    for _ in range(num_batches):
        batch = data_set.sample(type_="test", M=M, K=K)
        loss = batch_loss(model, batch)
        if isinstance(loss, tuple):
            loss = loss[0]
        total_loss += float(loss)
    return total_loss / num_batches


def run_training(cfg: dict, data_dir: str, output_dir: str, seed: int = 42, device: str = "cpu"):
    """Run training for Obstacle environment."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # Load dataset
    data_path = Path(data_dir) / "trajectories.npy"
    print(f"Loading data from {data_path}...")
    expert_data = np.load(data_path, allow_pickle=True).item()

    normalize = train_cfg.get("normalize", True)
    split_seed = train_cfg.get("split_seed", 42)

    # Restrict to the canonical TRAIN split so the held-out test tasks (used by
    # the baseline comparison) are never seen during pretraining.
    from src.training.baseline_data import obstacle_split, restrict_obstacle_dataset
    train_keys, _test_keys = obstacle_split(data_dir, split_seed=split_seed)
    expert_data = restrict_obstacle_dataset(expert_data, train_keys)
    print(f"Restricted to {len(train_keys)} canonical train obstacle tasks")

    data_set = ObstacleAvoidanceImitation(
        expert_data,
        train_perc=train_cfg.get("train_perc", 0.8),
        normalize=normalize,
        seed=split_seed,
    )
    print(f"Train tasks: {len(data_set.train_data)}, Test tasks: {len(data_set.test_data)}")

    state_dim = 4
    action_dim = 2

    # K options (randomly sampled each iteration)
    K_options = train_cfg.get("K", [5, 10, 20])
    if isinstance(K_options, int):
        K_options = [K_options]
    K_options = [int(k) for k in K_options]

    M = train_cfg.get("M", 16)
    num_iterations = train_cfg.get("num_iterations", 2000)
    eval_interval = train_cfg.get("eval_every", 100)
    learning_rate = train_cfg.get("learning_rate", 1e-3)
    num_runs = train_cfg.get("num_runs", 1)

    print("=" * 60)
    print("Obstacle SetONet Training")
    print("=" * 60)
    print(f"K options: {K_options}")
    print(f"Tasks per batch (M): {M}")
    print(f"Iterations: {num_iterations}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)

    best_model = None
    best_eval_loss = float("inf")
    all_train_losses = []
    all_eval_losses = []

    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        key = jax.random.PRNGKey(run_seed)
        np.random.seed(run_seed)

        print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_seed}) ---")

        key, model_key = jax.random.split(key)
        model = SetONet(
            input_size_src=2,              # obstacle (x, y)
            output_size_src=1,             # obstacle radius
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

        # Cosine decay + L1 regularization (matching original)
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=num_iterations,
            alpha=1e-2,
        )

        l1_strength = 1e-5

        def l1_regularization(strength):
            def init_fn(params):
                return optax.EmptyState()

            def update_fn(grads, state, params=None):
                if params is None:
                    return grads, state
                l1_grads = jax.tree_util.tree_map(lambda p: strength * jnp.sign(p), params)
                new_grads = jax.tree_util.tree_map(lambda g, l1g: g + l1g, grads, l1_grads)
                return new_grads, state

            return optax.GradientTransformation(init_fn, update_fn)

        optim = optax.chain(
            optax.adam(lr_schedule),
            l1_regularization(l1_strength),
        )
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        train_losses = []
        eval_losses = []

        print("\nStarting training...")

        for iteration in tqdm(range(num_iterations), desc="Training"):
            K = int(np.random.choice(K_options))
            batch_data = data_set.sample(type_="train", M=M, K=K)

            loss, model, opt_state = train_step(batch_loss, model, optim, opt_state, batch_data)
            train_losses.append(float(loss))

            if iteration % eval_interval == 0:
                eval_loss = eval_at_k(model, data_set, K=max(K_options), M=M, num_batches=5)
                eval_losses.append(eval_loss)
                tqdm.write(f"  Iter {iteration:5d} | Train: {loss:.6f} | Eval: {eval_loss:.6f}")

        all_train_losses.append(train_losses)
        all_eval_losses.append(eval_losses)

        final_eval = eval_losses[-1] if eval_losses else float("inf")
        if final_eval < best_eval_loss:
            best_eval_loss = final_eval
            best_model = model

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    eqx.tree_serialise_leaves(output_path / "setonet.eqx", best_model)
    print(f"\nModel saved to {output_path / 'setonet.eqx'}")

    np.savez(
        output_path / "training_history.npz",
        train_losses=np.array(all_train_losses),
        eval_losses=np.array(all_eval_losses),
        eval_iterations=np.arange(0, num_iterations, eval_interval),
        seed=seed,
        K_options=np.array(K_options),
    )
    print(f"Training history saved to {output_path / 'training_history.npz'}")

    import yaml
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return best_model, all_train_losses, all_eval_losses
