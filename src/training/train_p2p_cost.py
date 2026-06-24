"""
Train SetONet on P2P-Cost environment (double integrator with LQR).

Each task is defined by a goal state determining the immediate cost landscape.
- Branch input: (state, control) pairs with immediate cost values
- Trunk input: (state, time) along expert LQR rollout trajectories
- Output: optimal finite-horizon tvLQR controls
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import optax
from pathlib import Path

from trajax import tvlqr as tvlqr_lib
from src.setonet import SetONet


def solve_tvlqr(A, B, goal_state, horizon, Q_weight, R_weight, Qf_weight):
    """Solve finite-horizon time-varying LQR via backward Riccati recursion."""
    state_dim = A.shape[0]
    control_dim = B.shape[1]

    Q = jnp.eye(state_dim) * Q_weight
    R = jnp.eye(control_dim) * R_weight
    Qf = jnp.eye(state_dim) * Qf_weight

    A_seq = jnp.tile(A[None, :, :], (horizon, 1, 1))
    B_seq = jnp.tile(B[None, :, :], (horizon, 1, 1))
    Q_seq = jnp.tile(Q[None, :, :], (horizon, 1, 1))
    R_seq = jnp.tile(R[None, :, :], (horizon, 1, 1))
    M_seq = jnp.zeros((horizon, state_dim, control_dim))
    c_seq = jnp.zeros((horizon, state_dim))

    q_seq = -Q @ goal_state
    q_seq = jnp.tile(q_seq[None, :], (horizon, 1))
    r_seq = jnp.zeros((horizon, control_dim))
    qf = -Qf @ goal_state

    Q_seq_full = jnp.concatenate([Q_seq, Qf[None, :, :]], axis=0)
    q_seq_full = jnp.concatenate([q_seq, qf[None, :]], axis=0)

    K, k, P, p = tvlqr_lib.tvlqr(
        Q_seq_full, q_seq_full, R_seq, r_seq, M_seq,
        A_seq, B_seq, c_seq
    )

    return K, k


def compute_immediate_cost(state, control, goal_state, Q, R):
    """c(x, u) = (x - x_goal)^T Q (x - x_goal) + u^T R u"""
    state_error = state - goal_state
    return state_error @ Q @ state_error + control @ R @ control


def compute_normalization_stats(key, num_tasks, num_task_samples, state_dim, control_dim,
                                Q_weight, R_weight, Qf_weight, horizon, A_jnp, B_jnp,
                                goal_range, state_range, src_vel_range, src_control_range):
    """Compute normalization statistics by sampling from the training distribution.

    This uses online sampling (random goals, states, controls) matching the
    training data generation, which covers the full range the model will see
    including OOD evaluation scenarios.
    """
    print("\nComputing normalization statistics (online)...")

    key_goal, key_src_pos, key_src_vel, key_ctrl, key_tgt = jr.split(key, 5)

    Q = jnp.eye(state_dim) * Q_weight
    R = jnp.eye(control_dim) * R_weight

    goal_keys = jr.split(key_goal, num_tasks)
    pos_keys = jr.split(key_src_pos, num_tasks)
    vel_keys = jr.split(key_src_vel, num_tasks)
    ctrl_keys = jr.split(key_ctrl, num_tasks)
    tgt_keys = jr.split(key_tgt, num_tasks)

    all_states_list, all_costs_list = [], []
    all_src_controls_list, all_tgt_controls_list = [], []

    for i in range(num_tasks):
        goal_pos = jr.uniform(goal_keys[i], (2,), minval=goal_range[0], maxval=goal_range[1])
        goal_state = jnp.array([goal_pos[0], goal_pos[1], 0.0, 0.0])

        pos = jr.uniform(pos_keys[i], (num_task_samples, 2), minval=state_range[0], maxval=state_range[1])
        vel = jr.uniform(vel_keys[i], (num_task_samples, 2), minval=src_vel_range[0], maxval=src_vel_range[1])
        states = jnp.concatenate([pos, vel], axis=-1)

        src_controls = jr.uniform(ctrl_keys[i], (num_task_samples, control_dim),
                                  minval=src_control_range[0], maxval=src_control_range[1])

        costs = jax.vmap(lambda s, u: compute_immediate_cost(s, u, goal_state, Q, R))(states, src_controls)

        from trajax import tvlqr as tvlqr_lib
        K_tv, k_tv = solve_tvlqr(A_jnp, B_jnp, goal_state, horizon, Q_weight, R_weight, Qf_weight)
        tgt_pos_key, tgt_vel_key = jr.split(tgt_keys[i])
        start_pos = jr.uniform(tgt_pos_key, (2,), minval=state_range[0], maxval=state_range[1])
        start_vel = jr.uniform(tgt_vel_key, (2,), minval=src_vel_range[0], maxval=src_vel_range[1])
        start_state = jnp.concatenate([start_pos, start_vel])

        def step(state, t):
            u = K_tv[t] @ state + k_tv[t]
            next_state = A_jnp @ state + B_jnp @ u
            return next_state, u

        _, tgt_controls = jax.lax.scan(step, start_state, jnp.arange(horizon))

        all_states_list.append(states)
        all_costs_list.append(costs)
        all_src_controls_list.append(src_controls)
        all_tgt_controls_list.append(tgt_controls)

    all_states = jnp.concatenate(all_states_list, axis=0)
    all_costs = jnp.concatenate(all_costs_list, axis=0)
    all_src_controls = jnp.stack(all_src_controls_list, axis=0)
    all_tgt_controls = jnp.stack(all_tgt_controls_list, axis=0)

    stats = {
        'state_mean': jnp.mean(all_states, axis=0),
        'state_std': jnp.std(all_states, axis=0) + 1e-8,
        'cost_mean': jnp.mean(all_costs),
        'cost_std': jnp.std(all_costs) + 1e-8,
        'max_action': float(jnp.maximum(
            jnp.abs(all_src_controls).max(),
            jnp.abs(all_tgt_controls).max()
        )),
    }

    print(f"  State mean: {stats['state_mean']}")
    print(f"  State std:  {stats['state_std']}")
    print(f"  Cost mean:  {float(stats['cost_mean']):.4f}, std: {float(stats['cost_std']):.4f}")
    print(f"  Max action: {stats['max_action']:.4f}")

    return stats


def sample_task_batch(key, num_tasks, num_src_samples, horizon,
                      A_jnp, B_jnp, state_dim, control_dim,
                      Q_weight, R_weight, Qf_weight,
                      norm_stats=None,
                      goal_range=(-5, 5), state_range=(-5, 5),
                      src_vel_range=(-10.0, 10.0), src_control_range=(-5.0, 5.0)):
    """Sample a batch of tasks, each with a different goal state."""
    key_goal, key_src, key_tgt, key_ctrl = jr.split(key, 4)

    goal_keys = jr.split(key_goal, num_tasks)
    src_pos_keys = jr.split(key_src, num_tasks)
    key_src_vel = jr.fold_in(key_src, 12345)
    src_vel_keys = jr.split(key_src_vel, num_tasks)
    tgt_keys = jr.split(key_tgt, num_tasks)
    ctrl_keys = jr.split(key_ctrl, num_tasks)

    Q = jnp.eye(state_dim) * Q_weight
    R = jnp.eye(control_dim) * R_weight

    time_normalized = jnp.linspace(0.0, 1.0, horizon, endpoint=False)

    def sample_task_data(goal_key, src_pos_key, src_vel_key, ctrl_key, tgt_key):
        goal_pos = jr.uniform(goal_key, (2,), minval=goal_range[0], maxval=goal_range[1])
        goal_state = jnp.array([goal_pos[0], goal_pos[1], 0.0, 0.0])

        src_pos = jr.uniform(src_pos_key, (num_src_samples, 2),
                             minval=state_range[0], maxval=state_range[1])
        src_vel = jr.uniform(src_vel_key, (num_src_samples, 2),
                             minval=src_vel_range[0], maxval=src_vel_range[1])
        src_states = jnp.concatenate([src_pos, src_vel], axis=-1)

        src_controls = jr.uniform(ctrl_key, (num_src_samples, control_dim),
                                  minval=src_control_range[0], maxval=src_control_range[1])

        src_imm_cost = jax.vmap(
            lambda s, u: compute_immediate_cost(s, u, goal_state, Q, R)
        )(src_states, src_controls).reshape(-1, 1)

        K_tv, k_tv = solve_tvlqr(A_jnp, B_jnp, goal_state, horizon,
                                  Q_weight, R_weight, Qf_weight)

        tgt_pos_key, tgt_vel_key = jr.split(tgt_key)
        start_pos = jr.uniform(tgt_pos_key, (2,), minval=state_range[0], maxval=state_range[1])
        start_vel = jr.uniform(tgt_vel_key, (2,), minval=src_vel_range[0], maxval=src_vel_range[1])
        start_state = jnp.concatenate([start_pos, start_vel])

        def step(state, t):
            u = K_tv[t] @ state + k_tv[t]
            next_state = A_jnp @ state + B_jnp @ u
            return next_state, (state, u)

        _, (tgt_states, tgt_controls) = jax.lax.scan(step, start_state, jnp.arange(horizon))

        return src_states, src_controls, src_imm_cost, tgt_states, tgt_controls

    src_states, src_controls, src_immediate_cost, tgt_states, tgt_controls = jax.vmap(
        sample_task_data
    )(goal_keys, src_pos_keys, src_vel_keys, ctrl_keys, tgt_keys)

    if norm_stats is not None:
        src_states = (src_states - norm_stats['state_mean']) / norm_stats['state_std']
        tgt_states = (tgt_states - norm_stats['state_mean']) / norm_stats['state_std']
        src_immediate_cost = (src_immediate_cost - norm_stats['cost_mean']) / norm_stats['cost_std']
        if 'max_action' in norm_stats:
            src_controls = src_controls / norm_stats['max_action']
            tgt_controls = tgt_controls / norm_stats['max_action']

    time_broadcast = jnp.broadcast_to(
        time_normalized[None, :, None], (num_tasks, horizon, 1)
    )
    tgt_states = jnp.concatenate([tgt_states, time_broadcast], axis=-1)

    return src_states, src_controls, src_immediate_cost, tgt_states, tgt_controls


@eqx.filter_jit
def train_step(model, optim, opt_state, batch):
    """Single training step."""

    @eqx.filter_value_and_grad
    def compute_loss(model, batch):
        src_states, src_controls, src_immediate_cost, tgt_states, tgt_controls = batch

        def task_loss_fn(src_s, src_c, src_cost, tgt_s, tgt_c):
            src_inputs = jnp.concatenate([src_s, src_c], axis=-1)
            pred_controls = jax.vmap(
                lambda tgt: model(src_inputs, src_cost, tgt)
            )(tgt_s)
            return jnp.mean(jnp.square(pred_controls - tgt_c))

        task_losses = jax.vmap(task_loss_fn)(
            src_states, src_controls, src_immediate_cost, tgt_states, tgt_controls
        )
        return jnp.mean(task_losses)

    loss, grads = compute_loss(model, batch)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state


def evaluate(model, key, num_eval_tasks, num_src_samples, horizon,
             A_jnp, B_jnp, state_dim, control_dim, Q_weight, R_weight, Qf_weight,
             goal_range, state_range, src_vel_range, norm_stats=None):
    """Evaluate model on held-out tasks."""
    src_states, src_controls, src_immediate_cost, tgt_states, tgt_controls = sample_task_batch(
        key, num_eval_tasks, num_src_samples, horizon,
        A_jnp, B_jnp, state_dim, control_dim, Q_weight, R_weight, Qf_weight,
        norm_stats=norm_stats,
        goal_range=goal_range, state_range=state_range, src_vel_range=src_vel_range
    )

    def task_loss_fn(src_s, src_c, src_cost, tgt_s, tgt_c):
        src_inputs = jnp.concatenate([src_s, src_c], axis=-1)
        pred_controls = jax.vmap(
            lambda tgt: model(src_inputs, src_cost, tgt)
        )(tgt_s)
        return jnp.mean(jnp.square(pred_controls - tgt_c))

    task_losses = jax.vmap(task_loss_fn)(
        src_states, src_controls, src_immediate_cost, tgt_states, tgt_controls
    )
    return jnp.mean(task_losses)


def sample_batch_stored(data_loader, data_type, M, K, N):
    """Sample M tasks from the stored DoubleIntegratorLQRData (paper protocol).

    Branch context = K on-policy (state, control) transitions + immediate costs
    drawn from the task's stored LQR trajectories; trunk targets = the N expert
    trajectories flattened into (N*horizon) query points.
    Returns (src_states, src_controls, src_costs, tgt_states, tgt_controls).
    """
    batches = [data_loader.get_task(data_type, K, N) for _ in range(M)]
    src_s = np.stack([b[0] for b in batches])
    src_c = np.stack([b[1] for b in batches])
    src_cost = np.stack([b[2] for b in batches])
    tgt_s = np.stack([b[4].reshape(-1, b[4].shape[-1]) for b in batches])
    tgt_c = np.stack([b[5].reshape(-1, b[5].shape[-1]) for b in batches])
    return (jnp.asarray(src_s), jnp.asarray(src_c), jnp.asarray(src_cost),
            jnp.asarray(tgt_s), jnp.asarray(tgt_c))


def evaluate_stored(model, data_loader, M, K, N, num_batches=5):
    """Mean BC loss on held-out (test-split) tasks of the stored dataset."""
    def task_loss(ss, sc, sco, ts, tc):
        si = jnp.concatenate([ss, sc], axis=-1)
        pred = jax.vmap(lambda t: model(si, sco, t))(ts)
        return jnp.mean((pred - tc) ** 2)
    losses = []
    for _ in range(num_batches):
        b = sample_batch_stored(data_loader, "test", M, K, N)
        losses.append(float(jnp.mean(jax.vmap(task_loss)(*b))))
    return float(np.mean(losses))


def run_training(cfg: dict, data_dir: str, output_dir: str, seed: int = 42, device: str = "cpu"):
    """Train SetONet on P2P-Cost from the STORED LQR dataset (paper protocol,
    train_setonet_lqr_dataloader.py): branch context = K on-policy (state, control,
    cost) transitions; trunk imitates N expert trajectories."""
    import random as _random
    from src.envs.dataloader import DoubleIntegratorLQRData

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    state_dim, control_dim = 4, 2

    data_path = Path(data_dir) / "trajectories.npz"
    print(f"Loading stored LQR dataset from {data_path}...")
    raw = np.load(data_path, allow_pickle=True)
    dataset = {k: raw[k] for k in raw.files}
    if "norm_stats" in dataset:
        dataset["norm_stats"] = dataset["norm_stats"].item()

    M = train_cfg["M"]
    K_cfg = train_cfg["K"]
    K_options = [int(k) for k in K_cfg] if isinstance(K_cfg, (list, tuple)) else [int(K_cfg)]
    N = int(train_cfg.get("N", 32))
    num_iterations = train_cfg["num_iterations"]
    eval_interval = train_cfg["eval_every"]
    num_eval_tasks = train_cfg.get("num_eval_tasks", 20)
    learning_rate = train_cfg["learning_rate"]
    num_runs = train_cfg.get("num_runs", 1)
    train_perc = train_cfg.get("train_perc", 0.8)

    print("=" * 60)
    print("P2P-Cost SetONet Training (stored dataset, on-policy context)")
    print(f"Seed: {seed} | M: {M} | K: {K_options} | N: {N} | iters: {num_iterations} | runs: {num_runs}")
    print("=" * 60)

    best_model = None
    best_eval_loss = float("inf")
    best_norm_stats = None
    best_max_action = None
    all_train_losses = []
    all_eval_losses = []

    for run_idx in range(num_runs):
        run_seed = seed + run_idx
        np.random.seed(run_seed)
        _random.seed(run_seed)
        run_key = jr.PRNGKey(run_seed)
        print(f"\n--- Run {run_idx + 1}/{num_runs} (seed={run_seed}) ---")

        data_loader = DoubleIntegratorLQRData(dataset, train_perc=train_perc, normalize=True)
        norm_stats = data_loader.norm_stats
        max_action = float(data_loader.max_action)

        run_key, model_key = jr.split(run_key)
        model = SetONet(
            input_size_src=state_dim + control_dim,
            output_size_src=1,
            input_size_tgt=state_dim + 1,
            output_size_tgt=control_dim,
            p=model_cfg["p"],
            phi_hidden_size=model_cfg["phi_hidden_size"],
            phi_output_size=model_cfg["phi_output_size"],
            rho_hidden_size=model_cfg["rho_hidden_size"],
            trunk_hidden_size=model_cfg["trunk_hidden_size"],
            n_phi_layers=model_cfg["n_phi_layers"],
            n_rho_layers=model_cfg["n_rho_layers"],
            n_trunk_layers=model_cfg["n_trunk_layers"],
            aggregation_type=model_cfg["aggregation_type"],
            attention_n_heads=model_cfg["attention_n_heads"],
            attention_n_tokens=model_cfg["attention_n_tokens"],
            use_bias=model_cfg["use_bias"],
            key=model_key,
        )

        optim = optax.adam(learning_rate)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        train_losses, eval_losses = [], []
        print("\nStarting training...")
        for iteration in range(num_iterations):
            K = int(np.random.choice(K_options))
            batch = sample_batch_stored(data_loader, "train", M, K, N)
            loss, model, opt_state = train_step(model, optim, opt_state, batch)
            train_losses.append(float(loss))
            if iteration % eval_interval == 0:
                eval_loss = evaluate_stored(model, data_loader, num_eval_tasks, max(K_options), N)
                eval_losses.append(eval_loss)
                print(f"  Iter {iteration:5d} | Train: {loss:.6f} | Eval: {eval_loss:.6f}")

        all_train_losses.append(train_losses)
        all_eval_losses.append(eval_losses)
        final_eval = eval_losses[-1] if eval_losses else float("inf")
        if final_eval < best_eval_loss:
            best_eval_loss = final_eval
            best_model = model
            best_norm_stats = norm_stats
            best_max_action = max_action

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(output_path / "setonet.eqx", best_model)
    print(f"\nModel saved to {output_path / 'setonet.eqx'}")

    np.savez(
        output_path / "training_history.npz",
        train_losses=np.array(all_train_losses),
        eval_losses=np.array(all_eval_losses),
        seed=seed,
        state_mean=np.array(best_norm_stats['state_mean']),
        state_std=np.array(best_norm_stats['state_std']),
        cost_mean=np.array(best_norm_stats['cost_mean']),
        cost_std=np.array(best_norm_stats['cost_std']),
        max_action=np.array(best_max_action),
    )
    print(f"Training history saved to {output_path / 'training_history.npz'}")
    import yaml
    with open(output_path / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return best_model, all_train_losses, all_eval_losses
