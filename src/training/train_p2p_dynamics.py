"""
Train SetONet on P2P-Dynamics environment (varying dynamics, single goal).

Each task is defined by a dynamics configuration (friction, max velocity, max acceleration).
- Branch input: (state, action) pairs with next_state values
- Trunk input: (state, time) along expert trajectories
- Output: optimal controls
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from pathlib import Path

from src.setonet import SetONet, NormalizedSetONet
from src.normalization import GridEnvironmentNormalizer
from src.envs.dataloader import VaryingDynamicsData


@eqx.filter_jit
def train_step(batch_loss_fn, model, optim, opt_state, batch):
    """Single training step."""
    loss, grads = batch_loss_fn(model, batch)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def setonet_batch_loss(model, batch):
    """Compute SetONet loss on a batch.

    Branch: (state, action) -> next_state
    Trunk: expert trajectory states -> predicted actions
    """
    (random_states, random_actions, random_next_states,
     expert_trajectories, expert_actions) = batch

    # Branch input: (state, action), Branch output: next_state
    src_input = jnp.concatenate([random_states, random_actions], axis=-1)

    # Trunk input: expert trajectory states (excluding time dim at end)
    tgt_input = expert_trajectories[:, :, :-1, :]

    predicted_actions = jax.vmap(       # over tasks
        jax.vmap(                        # over trajectories
            jax.vmap(                    # over timesteps
                model, in_axes=(None, None, 0)
            ), in_axes=(None, None, 0)
        )
    )(src_input, random_next_states, tgt_input)

    loss = jnp.mean((predicted_actions - expert_actions) ** 2)
    return loss


def sample_batch(data_set, data_type, M, K, N):
    """Sample a batch for SetONet training.

    Returns (state, action, next_state) and expert trajectory data.
    """
    batch = data_set.get_task(type_=data_type, K=K, N=N)
    # get_task returns a single task; we need M tasks
    # Use a loop since the dataloader uses numpy random state
    all_batches = [data_set.get_task(type_=data_type, K=K, N=N) for _ in range(M)]

    random_states = np.stack([b[0] for b in all_batches])
    random_actions = np.stack([b[1] for b in all_batches])
    random_next_states = np.stack([b[2] for b in all_batches])
    expert_trajectories = np.stack([b[4] for b in all_batches])  # index 4 = expert_trajectories
    expert_actions = np.stack([b[5] for b in all_batches])       # index 5 = expert_actions

    return (
        jnp.array(random_states),
        jnp.array(random_actions),
        jnp.array(random_next_states),
        jnp.array(expert_trajectories),
        jnp.array(expert_actions),
    )


def evaluate(model, data_set, M, K, N, num_batches=5):
    """Evaluate model on test data."""
    losses = []
    for _ in range(num_batches):
        batch = sample_batch(data_set, "test", M, K, N)
        loss = setonet_batch_loss(model, batch)
        if isinstance(loss, tuple):
            loss = loss[0]
        losses.append(float(loss))
    return np.mean(losses)


def run_training(cfg: dict, data_dir: str, output_dir: str, seed: int = 42, device: str = "cpu"):
    """Run training for P2P-Dynamics environment."""
    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # Load dataset
    data_path = Path(data_dir) / "trajectories.npz"
    print(f"Loading data from {data_path}...")
    raw_data = dict(np.load(data_path, allow_pickle=True))

    # Restrict to the canonical TRAIN split so the held-out test tasks (used by
    # the baseline comparison) are never seen during pretraining.
    from src.training.baseline_data import dynamics_split, restrict_dynamics_dataset
    split_seed = train_cfg.get("split_seed", 42)
    train_configs, _test_configs, _g0 = dynamics_split(data_dir, split_seed=split_seed)
    raw_data = restrict_dynamics_dataset(raw_data, train_configs)
    print(f"Restricted to {len(train_configs)} canonical train dynamics configs")

    # Filter for a single goal (goal_idx=0)
    goal_indices = np.unique(raw_data["goal_indices"])
    goal_idx = int(goal_indices[0])
    print(f"Using goal index: {goal_idx}")

    data_set = VaryingDynamicsData.load_for_goal(goal_idx, raw_data, train_perc=0.8)

    state_dim = raw_data["states"].shape[-1]
    action_dim = raw_data["actions"].shape[-1]

    # Training params
    M = train_cfg.get("M", 32)
    K_config = train_cfg.get("K", 32)
    K_options = [int(k) for k in K_config] if isinstance(K_config, (list, tuple)) else [int(K_config)]
    N = train_cfg.get("N", 16)
    num_iterations = train_cfg.get("num_iterations", 2000)
    eval_interval = train_cfg.get("eval_every", 100)
    learning_rate = train_cfg.get("learning_rate", 1e-3)
    num_runs = train_cfg.get("num_runs", 1)

    print("=" * 60)
    print("P2P-Dynamics SetONet Training")
    print("=" * 60)
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Tasks per batch (M): {M}")
    print(f"Context samples (K): {K_options}")
    print(f"Expert trajectories (N): {N}")
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

        # Initialize model
        key, model_key = jax.random.split(key)
        base_model = SetONet(
            input_size_src=state_dim + action_dim,
            output_size_src=state_dim,
            input_size_tgt=state_dim + 1,  # state + time
            output_size_tgt=action_dim,
            p=model_cfg.get("p", 32),
            phi_hidden_size=model_cfg.get("phi_hidden_size", 128),
            phi_output_size=model_cfg.get("phi_output_size", 128),
            rho_hidden_size=model_cfg.get("rho_hidden_size", 128),
            trunk_hidden_size=model_cfg.get("trunk_hidden_size", 128),
            n_phi_layers=model_cfg.get("n_phi_layers", 4),
            n_rho_layers=model_cfg.get("n_rho_layers", 4),
            n_trunk_layers=model_cfg.get("n_trunk_layers", 4),
            aggregation_type=model_cfg.get("aggregation_type", "attention"),
            attention_n_heads=model_cfg.get("attention_n_heads", 4),
            attention_n_tokens=model_cfg.get("attention_n_tokens", 1),
            use_bias=model_cfg.get("use_bias", True),
            key=model_key,
        )

        # Wrap in NormalizedSetONet (min-max state/action scaling). max_acceleration
        # uses the data's actual max action (the solver produces unconstrained actions).
        if model_cfg.get("use_normalization", True):
            x_range = data_cfg["workspace"]["x_range"]
            max_vel = data_cfg["dynamics_ranges"]["max_velocity_range"][1]
            normalizer = GridEnvironmentNormalizer(
                position_range=(x_range[0], x_range[1]),
                max_velocity=max_vel,
                max_acceleration=float(data_set.max_action),
            )
            model = NormalizedSetONet(base_model, normalizer)
            print(f"Using NormalizedSetONet (max_action={float(data_set.max_action):.3f})")
        else:
            model = base_model

        # Cosine LR decay + global-norm gradient clipping (paper setup).
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate, decay_steps=num_iterations,
            alpha=train_cfg.get("cosine_decay_alpha", 0.01),
        )
        optim = optax.chain(
            optax.clip_by_global_norm(train_cfg.get("grad_clip_norm", 1.0)),
            optax.adam(lr_schedule),
        )
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        train_losses = []
        eval_losses = []

        print("\nStarting training...")

        for iteration in range(num_iterations):
            K = int(np.random.choice(K_options))
            batch = sample_batch(data_set, "train", M, K, N)
            loss, model, opt_state = train_step(setonet_batch_loss, model, optim, opt_state, batch)
            train_losses.append(float(loss))

            if iteration % eval_interval == 0:
                eval_loss = evaluate(model, data_set, M, max(K_options), N)
                eval_losses.append(eval_loss)
                print(f"  Iter {iteration:5d} | Train: {loss:.6f} | Eval: {eval_loss:.6f}")

        all_train_losses.append(train_losses)
        all_eval_losses.append(eval_losses)

        final_eval = eval_losses[-1] if eval_losses else float("inf")
        if final_eval < best_eval_loss:
            best_eval_loss = final_eval
            best_model = model

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "setonet.eqx"
    eqx.tree_serialise_leaves(model_path, best_model)
    print(f"\nModel saved to {model_path}")

    np.savez(
        output_path / "training_history.npz",
        train_losses=np.array(all_train_losses),
        eval_losses=np.array(all_eval_losses),
        eval_iterations=np.arange(0, num_iterations, eval_interval),
        seed=seed,
    )
    print(f"Training history saved to {output_path / 'training_history.npz'}")

    import yaml
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return best_model, all_train_losses, all_eval_losses
