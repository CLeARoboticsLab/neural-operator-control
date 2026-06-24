"""
Train SetONet on Quadrotor environment (varying dynamics, fixed goal).

Each task is defined by a dynamics configuration (mass, inertia, arm length).
- Branch input: (state, action) pairs with next_state values
- Trunk input: (state, time) along expert trajectories
- Output: optimal controls (thrust, torque)

Uses its own dataloader with action normalization (mean/std) since the
quadrotor has 6D state and physically meaningful action scales.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import optax
from pathlib import Path

from src.setonet import SetONet


class QuadrotorDataLoader:
    """Dataloader for quadrotor varying dynamics data with fixed goal.

    Handles train/test split by dynamics configuration and normalizes
    expert actions by mean/std computed from training data.
    """

    def __init__(self, dataset: dict, train_perc: float = 0.8):
        self.dataset = dataset
        self.dynamics_indices = dataset['dynamics_indices']
        self.unique_dynamics = np.unique(self.dynamics_indices)
        self.num_dynamics = len(self.unique_dynamics)

        np.random.shuffle(self.unique_dynamics)
        num_train = int(self.num_dynamics * train_perc)
        self.train_dynamics = set(self.unique_dynamics[:num_train])
        self.test_dynamics = set(self.unique_dynamics[num_train:])

        self.train_mask = np.array([d in self.train_dynamics for d in self.dynamics_indices])
        self.test_mask = np.array([d in self.test_dynamics for d in self.dynamics_indices])

        # Action normalization from training data
        train_actions = dataset['actions'][self.train_mask]
        self.action_mean = np.mean(train_actions, axis=(0, 1))
        self.action_std = np.maximum(np.std(train_actions, axis=(0, 1)), 1e-6)

        print(f"  Train dynamics: {len(self.train_dynamics)}, Test: {len(self.test_dynamics)}")
        print(f"  Train trajs: {np.sum(self.train_mask)}, Test trajs: {np.sum(self.test_mask)}")
        print(f"  Action norm: mean={self.action_mean}, std={self.action_std}")

    @classmethod
    def load_for_goal(cls, goal_idx: int, dataset: dict, train_perc: float = 0.8):
        """Load dataloader filtered for a specific goal."""
        goal_mask = dataset["goal_indices"] == goal_idx
        filtered = {}
        for key, value in dataset.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == len(goal_mask):
                filtered[key] = value[goal_mask]
            else:
                filtered[key] = value
        return cls(filtered, train_perc)

    def sample_task(self, data_type: str, K: int, N: int):
        """Sample data for a single task (dynamics configuration)."""
        available = list(self.train_dynamics if data_type == "train" else self.test_dynamics)
        config_idx = np.random.choice(available)

        mask = (self.train_mask if data_type == "train" else self.test_mask) & \
               (self.dynamics_indices == config_idx)
        traj_indices = np.where(mask)[0]

        if len(traj_indices) == 0:
            raise ValueError(f"No trajectories for config {config_idx}")

        states = self.dataset['states']
        actions = self.dataset['actions']

        # Collect transitions
        all_transitions = []
        for traj_idx in traj_indices:
            s = states[traj_idx]
            a = actions[traj_idx]
            for t in range(len(a)):
                all_transitions.append((s[t], a[t], s[t + 1]))

        total = len(all_transitions)
        src_idx = np.random.choice(total, size=K, replace=K > total)
        random_states = np.array([all_transitions[i][0] for i in src_idx])
        random_actions = np.array([all_transitions[i][1] for i in src_idx])
        random_next_states = np.array([all_transitions[i][2] for i in src_idx])

        # Expert trajectories with normalized time
        sampled = np.random.choice(traj_indices, size=N, replace=N > len(traj_indices))
        expert_traj = states[sampled]  # (N, horizon+1, state_dim)
        horizon = expert_traj.shape[1]
        time_norm = np.linspace(0.0, 1.0, horizon)[:, None]
        time_bc = np.repeat(time_norm[None, :, :], N, axis=0)
        expert_traj = np.concatenate([expert_traj, time_bc], axis=-1)

        expert_actions = (actions[sampled] - self.action_mean) / self.action_std

        return (random_states, random_actions, random_next_states,
                expert_traj, expert_actions)

    def sample_batch(self, data_type: str, M: int, K: int, N: int):
        """Sample M tasks as a stacked batch."""
        batches = [self.sample_task(data_type, K, N) for _ in range(M)]
        return tuple(
            jnp.array(np.stack([b[i] for b in batches], axis=0))
            for i in range(5)
        )


@eqx.filter_value_and_grad
def setonet_batch_loss(model, batch):
    """Compute SetONet loss on a quadrotor batch."""
    (random_states, random_actions, random_next_states,
     expert_trajectories, expert_actions) = batch

    src_input = jnp.concatenate([random_states, random_actions], axis=-1)
    src_output = random_next_states
    tgt_input = expert_trajectories[:, :, :-1, :]  # exclude last timestep

    predicted_actions = jax.vmap(
        jax.vmap(
            jax.vmap(
                model, in_axes=(None, None, 0)
            ), in_axes=(None, None, 0)
        )
    )(src_input, src_output, tgt_input)

    return jnp.mean((predicted_actions - expert_actions) ** 2)


@eqx.filter_jit
def train_step(model, optim, opt_state, batch):
    """Single training step."""
    loss, grads = setonet_batch_loss(model, batch)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def evaluate(model, data_loader, M, K, N, num_batches=5):
    """Evaluate model on test tasks."""
    losses = []
    for _ in range(num_batches):
        batch = data_loader.sample_batch("test", M, K, N)
        loss, _ = setonet_batch_loss(model, batch)
        losses.append(float(loss))
    return np.mean(losses)


def run_training(cfg: dict, data_dir: str, output_dir: str, seed: int = 42, device: str = "cpu"):
    """Run training for Quadrotor environment."""
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # Load dataset
    data_path = Path(data_dir) / "trajectories.npz"
    print(f"Loading data from {data_path}...")
    raw_data = dict(np.load(data_path, allow_pickle=True))

    # Restrict to the canonical TRAIN split (leak-free held-out test for comparison).
    from src.training.baseline_data import dynamics_split, restrict_dynamics_dataset
    split_seed = train_cfg.get("split_seed", 42)
    train_configs, _test_configs, _g0 = dynamics_split(data_dir, split_seed=split_seed)
    raw_data = restrict_dynamics_dataset(raw_data, train_configs)
    print(f"Restricted to {len(train_configs)} canonical train dynamics configs")

    # Filter for fixed goal
    goal_indices = np.unique(raw_data["goal_indices"])
    goal_idx = int(goal_indices[0])
    print(f"Using goal index: {goal_idx}")

    np.random.seed(seed)
    data_loader = QuadrotorDataLoader.load_for_goal(goal_idx, raw_data, train_perc=0.8)

    state_dim = raw_data["states"].shape[-1]   # 6
    action_dim = raw_data["actions"].shape[-1]  # 2

    M = train_cfg.get("M", 16)
    K_config = train_cfg.get("K", 64)
    K_options = [int(k) for k in K_config] if isinstance(K_config, (list, tuple)) else [int(K_config)]
    N = train_cfg.get("N", 5)
    num_iterations = train_cfg.get("num_iterations", 2000)
    eval_interval = train_cfg.get("eval_every", 100)
    learning_rate = train_cfg.get("learning_rate", 1e-3)
    num_runs = train_cfg.get("num_runs", 1)

    print("=" * 60)
    print("Quadrotor SetONet Training (Dynamics-based)")
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
        key = jr.PRNGKey(run_seed)
        np.random.seed(run_seed)

        print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_seed}) ---")

        key, model_key = jr.split(key)
        model = SetONet(
            input_size_src=state_dim + action_dim,
            output_size_src=state_dim,
            input_size_tgt=state_dim + 1,
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

        optim = optax.adam(learning_rate)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        train_losses = []
        eval_losses = []

        print("\nStarting training...")

        for iteration in range(num_iterations):
            K = int(np.random.choice(K_options))
            batch = data_loader.sample_batch("train", M, K, N)
            loss, model, opt_state = train_step(model, optim, opt_state, batch)
            train_losses.append(float(loss))

            if iteration % eval_interval == 0:
                eval_loss = evaluate(model, data_loader, M, max(K_options), N)
                eval_losses.append(eval_loss)
                print(f"  Iter {iteration:5d} | Train: {loss:.6f} | Eval: {eval_loss:.6f}")

        all_train_losses.append(train_losses)
        all_eval_losses.append(eval_losses)

        final_eval = eval_losses[-1] if eval_losses else float("inf")
        if final_eval < best_eval_loss:
            best_eval_loss = final_eval
            best_model = model
            best_action_mean = data_loader.action_mean
            best_action_std = data_loader.action_std

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
        action_mean=best_action_mean,
        action_std=best_action_std,
    )
    print(f"Training history saved to {output_path / 'training_history.npz'}")

    import yaml
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return best_model, all_train_losses, all_eval_losses
