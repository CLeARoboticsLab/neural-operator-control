"""
Figure 7: Cost-based fine-tuning across two environments.

(a) P2P-Cost OOD: goals outside training region, cost-based adaptation
(b) Obstacle: held-out obstacle tasks, cost-based adaptation

Both use differentiable rollout through known dynamics to compute cost gradients
for policy fine-tuning (no expert demonstrations needed).

Usage (matches Makefile):
    # P2P-Cost OOD
    python src/adaptation/cost_adapt.py \
        --config configs/p2p_cost.yaml \
        --data data/p2p_cost \
        --checkpoint checkpoints/p2p_cost/pretrained \
        --ood \
        --output outputs/results/cost_adapt_p2p_ood

    # Obstacle
    python src/adaptation/cost_adapt.py \
        --config configs/obstacle.yaml \
        --data data/obstacle \
        --checkpoint checkpoints/obstacle/pretrained \
        --output outputs/results/cost_adapt_obstacle
"""

import argparse
import copy
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from pathlib import Path

from src.setonet import SetONet

DEFAULT_SEED = 42


# ── Shared helpers ──────────────────────────────────────────────────────────

def clip_grads(grads, max_norm=1.0):
    leaves = jax.tree_util.tree_leaves(grads)
    total_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-8))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def load_setonet(checkpoint_path, model_cfg, input_src, output_src,
                 input_tgt, output_tgt, key):
    model = SetONet(
        input_size_src=input_src, output_size_src=output_src,
        input_size_tgt=input_tgt, output_size_tgt=output_tgt,
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
        key=key,
    )
    return eqx.tree_deserialise_leaves(str(checkpoint_path), model)


def make_filter_spec(model, mode='ft'):
    """Create filter spec for different fine-tuning modes."""
    if mode == 'ft':
        # Freeze trunk, train everything else
        filt = jax.tree_util.tree_map(lambda x: eqx.is_array(x), model)
        filt = eqx.tree_at(lambda m: m.trunk, filt,
                           replace=jax.tree_util.tree_map(lambda _: False, model.trunk))
    elif mode == 'last_branch':
        filt = jax.tree_util.tree_map(lambda _: False, model)
        filt = eqx.tree_at(lambda m: m.rho.layers[-1], filt,
                           replace=jax.tree_util.tree_map(lambda x: eqx.is_array(x), model.rho.layers[-1]))
    elif mode == 'last_both':
        filt = jax.tree_util.tree_map(lambda _: False, model)
        filt = eqx.tree_at(
            lambda m: (m.trunk.layers[-1], m.rho.layers[-1]), filt,
            replace=(
                jax.tree_util.tree_map(lambda x: eqx.is_array(x), model.trunk.layers[-1]),
                jax.tree_util.tree_map(lambda x: eqx.is_array(x), model.rho.layers[-1]),
            ))
    return filt


# =============================================================================
# P2P-Cost OOD (Figure 7a)
# =============================================================================

# Constants matching training
DT = 0.1
STATE_DIM = 4
CONTROL_DIM = 2
HORIZON = 50
ROLLOUT_STEPS = 40
CONTROL_CLIP = 20.0

A_jnp = jnp.array([[1,0,DT,0],[0,1,0,DT],[0,0,1,0],[0,0,0,1]])
B_jnp = jnp.array([[0,0],[0,0],[DT,0],[0,DT]])
Q_jnp = jnp.eye(4) * 1.0
R_jnp = jnp.eye(2) * 0.1
Qf_jnp = jnp.eye(4) * 10.0


def generate_border_goals(half_width):
    """Generate 16 goals on the border of a square."""
    corners = [(half_width, half_width), (-half_width, half_width),
               (-half_width, -half_width), (half_width, -half_width)]
    goals = []
    for i in range(4):
        goals.append(corners[i])
        nx, ny = corners[(i + 1) % 4]
        cx, cy = corners[i]
        for j in range(1, 4):
            frac = j / 4.0
            goals.append((cx + frac * (nx - cx), cy + frac * (ny - cy)))
    return goals


def sample_cost_context(goal_state, norm_stats, key, K=128):
    """Sample random (state, control) -> cost context for a goal."""
    k1, k2, k3 = jr.split(key, 3)
    src_pos = jr.uniform(k1, (K, 2), minval=-15.0, maxval=15.0)
    src_vel = jr.uniform(k2, (K, 2), minval=-10.0, maxval=10.0)
    src_states = jnp.concatenate([src_pos, src_vel], axis=-1)
    src_controls = jr.uniform(k3, (K, 2), minval=-5.0, maxval=5.0)
    src_cost = jax.vmap(
        lambda s, u: (s - goal_state) @ Q_jnp @ (s - goal_state) + u @ R_jnp @ u
    )(src_states, src_controls).reshape(-1, 1)
    s_norm = (src_states - norm_stats['state_mean']) / norm_stats['state_std']
    c_norm = (src_cost - norm_stats['cost_mean']) / norm_stats['cost_std']
    u_norm = src_controls / norm_stats['max_action']
    return jnp.concatenate([s_norm, u_norm], axis=-1), c_norm


def rollout_p2p(model, src_inputs, src_costs, start_state, norm_stats):
    """Roll out SetONet with discrete LQR dynamics."""
    X = np.zeros((ROLLOUT_STEPS + 1, 4))
    U = np.zeros((ROLLOUT_STEPS, 2))
    X[0] = start_state
    state = jnp.array(start_state)
    for t in range(ROLLOUT_STEPS):
        s_norm = (state - norm_stats['state_mean']) / norm_stats['state_std']
        query = jnp.concatenate([s_norm, jnp.array([t / ROLLOUT_STEPS])])
        u = model(src_inputs, src_costs, query) * norm_stats['max_action']
        u = jnp.clip(u, -CONTROL_CLIP, CONTROL_CLIP)
        U[t] = np.array(u)
        state = A_jnp @ state + B_jnp @ u
        X[t + 1] = np.array(state)
    return X, U


def differentiable_rollout_p2p(model, src_inputs, src_costs, start_state,
                                goal_state, norm_stats):
    """Differentiable rollout for cost-based FT."""
    def step_fn(carry, t):
        state, cum_cost = carry
        s_norm = (state - norm_stats['state_mean']) / norm_stats['state_std']
        query = jnp.concatenate([s_norm, jnp.array([t / ROLLOUT_STEPS])])
        action = model(src_inputs, src_costs, query) * norm_stats['max_action']
        action = jnp.clip(action, -CONTROL_CLIP, CONTROL_CLIP)
        err = state - goal_state
        cost = 1.0 * (err @ Q_jnp @ err) + 0.1 * (action @ R_jnp @ action)
        return (A_jnp @ state + B_jnp @ action, cum_cost + cost), None

    (final, cum_cost), _ = jax.lax.scan(step_fn, (start_state, 0.0), jnp.arange(ROLLOUT_STEPS))
    terminal_err = final - goal_state
    return cum_cost + 10.0 * (terminal_err @ Qf_jnp @ terminal_err)


def finetune_p2p(model, src_inputs, src_costs, train_starts, goal_state,
                  norm_stats, grad_steps, lr, mode='ft'):
    """Cost-based fine-tuning for P2P-Cost."""
    ft = copy.deepcopy(model)
    filt = make_filter_spec(ft, mode)
    trainable, frozen = eqx.partition(ft, filt)
    opt = optax.adam(lr)
    opt_state = opt.init(trainable)

    goal_jnp = jnp.array(goal_state)

    @eqx.filter_jit
    def step(tr, os):
        def loss_fn(tr):
            m = eqx.combine(tr, frozen)
            total = jnp.array(0.0)
            for s in train_starts:
                total = total + differentiable_rollout_p2p(
                    m, src_inputs, src_costs, jnp.array(s), goal_jnp, norm_stats)
            return total / len(train_starts)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(tr)
        updates, os_new = opt.update(grads, os)
        return eqx.apply_updates(tr, updates), os_new, loss

    for _ in range(grad_steps):
        trainable, opt_state, _ = step(trainable, opt_state)
    return eqx.combine(trainable, frozen)


def run_p2p_ood(cfg, data_dir, checkpoint_path, output_dir, seed):
    """Run P2P-Cost OOD evaluation (Figure 7a)."""
    from trajax import tvlqr
    print("=" * 60)
    print("Figure 7a: P2P-Cost OOD Cost-Based Adaptation")
    print("=" * 60)

    key = jr.PRNGKey(seed)

    # Load norm stats from training history (matches what model was trained with)
    hist = np.load(str(Path(checkpoint_path) / "training_history.npz"))
    norm_stats = {
        'state_mean': jnp.array(hist['state_mean']),
        'state_std': jnp.array(hist['state_std']),
        'cost_mean': float(hist['cost_mean']),
        'cost_std': float(hist['cost_std']),
        'max_action': float(hist['max_action']),
    }
    print(f"Norm stats from training: state_std={norm_stats['state_std']}, max_action={norm_stats['max_action']:.2f}")

    # Load model
    key, mk = jr.split(key)
    model = load_setonet(Path(checkpoint_path) / "setonet.eqx",
                         cfg["model"], 6, 1, 5, 2, mk)

    half_width = 15.0
    start_center = np.array([-20.0, -20.0])
    start_range = 1.0
    ft_steps = 25
    ft_lr = 1e-4
    num_starts = 2

    goal_positions = generate_border_goals(half_width)
    print(f"{len(goal_positions)} OOD goals on ±{half_width} border")

    methods = {
        'SetONet': {'trajectories': [], 'distances': []},
        'FT': {'trajectories': [], 'distances': []},
        'Last-Branch': {'trajectories': [], 'distances': []},
        'Last-Both': {'trajectories': [], 'distances': []},
    }
    lqr_trajectories = []

    key, starts_key = jr.split(key)

    for g_idx, (gx, gy) in enumerate(goal_positions):
        goal_state = np.array([gx, gy, 0.0, 0.0])

        key, ctx_key = jr.split(key)
        src_inputs, src_costs = sample_cost_context(jnp.array(goal_state), norm_stats, ctx_key)

        # Generate start states
        starts_key, k_train, k_eval = jr.split(starts_key, 3)
        train_perturb = jr.uniform(k_train, (num_starts, 2), minval=-start_range, maxval=start_range)
        eval_perturb = jr.uniform(k_eval, (num_starts, 2), minval=-start_range, maxval=start_range)
        train_starts = [np.array([float(p[0] + start_center[0]), float(p[1] + start_center[1]), 0, 0])
                        for p in train_perturb]
        eval_starts = [np.array([float(p[0] + start_center[0]), float(p[1] + start_center[1]), 0, 0])
                       for p in eval_perturb]

        # Fine-tune
        ft_model = finetune_p2p(model, src_inputs, src_costs, train_starts, goal_state,
                                 norm_stats, ft_steps, ft_lr, 'ft')
        lb_model = finetune_p2p(model, src_inputs, src_costs, train_starts, goal_state,
                                 norm_stats, ft_steps, ft_lr, 'last_branch')
        lboth_model = finetune_p2p(model, src_inputs, src_costs, train_starts, goal_state,
                                    norm_stats, ft_steps, ft_lr, 'last_both')

        for start_state in eval_starts:
            # LQR expert
            A_seq = jnp.tile(A_jnp[None], (HORIZON, 1, 1))
            B_seq = jnp.tile(B_jnp[None], (HORIZON, 1, 1))
            R_seq = jnp.tile(R_jnp[None], (HORIZON, 1, 1))
            M_seq = jnp.zeros((HORIZON, 4, 2))
            c_seq = jnp.zeros((HORIZON, 4))
            q_seq = jnp.tile((-Q_jnp @ jnp.array(goal_state))[None], (HORIZON, 1))
            r_seq = jnp.zeros((HORIZON, 2))
            qf = -Qf_jnp @ jnp.array(goal_state)
            Q_full = jnp.concatenate([jnp.tile(Q_jnp[None], (HORIZON, 1, 1)), Qf_jnp[None]], axis=0)
            q_full = jnp.concatenate([q_seq, qf[None]], axis=0)
            K_g, k_g, _, _ = tvlqr.tvlqr(Q_full, q_full, R_seq, r_seq, M_seq, A_seq, B_seq, c_seq)
            X_lqr, _ = tvlqr.rollout(K_g, k_g, jnp.array(start_state), A_seq, B_seq, c_seq)
            lqr_trajectories.append(np.array(X_lqr)[:ROLLOUT_STEPS + 1])

            # Pretrained
            X_pre, _ = rollout_p2p(model, src_inputs, src_costs, start_state, norm_stats)
            methods['SetONet']['trajectories'].append(X_pre)
            methods['SetONet']['distances'].append(float(np.linalg.norm(X_pre[-1, :2] - goal_state[:2])))

            # FT variants
            for name, m in [('FT', ft_model), ('Last-Branch', lb_model), ('Last-Both', lboth_model)]:
                X, _ = rollout_p2p(m, src_inputs, src_costs, start_state, norm_stats)
                methods[name]['trajectories'].append(X)
                methods[name]['distances'].append(float(np.linalg.norm(X[-1, :2] - goal_state[:2])))

        print(f"  Goal {g_idx+1}/{len(goal_positions)}: ({gx:.1f}, {gy:.1f})")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'goal_positions': np.array(goal_positions),
        'start_center': start_center,
        'half_width': half_width,
        'train_range': 10.0,
    }
    for name, data in methods.items():
        key_name = name.lower().replace('-', '_')
        save_dict[f'{key_name}_distances'] = np.array(data['distances'])
        save_dict[f'{key_name}_trajectories'] = np.array(data['trajectories'])
    save_dict['lqr_trajectories'] = np.array(lqr_trajectories)

    np.savez(output_path / "results.npz", **save_dict)
    print(f"\nResults saved to {output_path / 'results.npz'}")

    # Print summary
    print("\nDistance to Goal (mean ± std):")
    for name, data in methods.items():
        d = np.array(data['distances'])
        print(f"  {name:15s}: {np.mean(d):.2f} ± {np.std(d):.2f}")


# =============================================================================
# Obstacle Avoidance (Figure 7b)
# =============================================================================

def dynamics_step_rk4(state, action, dt):
    """Double integrator RK4 step."""
    def f(s, a):
        return jnp.array([s[2], s[3], a[0], a[1]])
    k1 = f(state, action)
    k2 = f(state + 0.5 * dt * k1, action)
    k3 = f(state + 0.5 * dt * k2, action)
    k4 = f(state + dt * k3, action)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def collision_cost(position, obstacles, margin=0.2, sharpness=15.0):
    """Soft collision cost."""
    def single(obs):
        dist = jnp.sqrt(jnp.sum((position - obs[:2])**2) + 1e-6)
        return jnp.exp(-sharpness * (dist - obs[2] - margin))
    return jnp.sum(jax.vmap(single)(obstacles))


def rollout_obstacle(model, start_state, obstacles, horizon, dt,
                     normalize, max_pos, max_vel, max_action):
    """Roll out obstacle policy."""
    obs_norm = obstacles / max_pos if normalize else obstacles
    obs_positions = jnp.array(obs_norm[:, :2])
    obs_values = jnp.array(obs_norm[:, 2:3])
    times = jnp.linspace(0, 1, horizon + 1)

    states = [start_state]
    actions = []
    state = jnp.array(start_state)
    for t in range(horizon):
        if normalize:
            s_norm = jnp.concatenate([state[:2] / max_pos, state[2:] / max_vel])
        else:
            s_norm = state
        tgt = jnp.concatenate([s_norm, jnp.array([times[t]])])
        action = model(obs_positions, obs_values, tgt) * max_action
        actions.append(action)
        state = dynamics_step_rk4(state, action, dt)
        states.append(state)
    return jnp.stack(states), jnp.stack(actions)


def count_collisions(states, obstacles, margin=0.2):
    """Count collision timesteps."""
    n = 0
    for t in range(len(states)):
        for obs in obstacles:
            dist = np.sqrt((states[t, 0] - obs[0])**2 + (states[t, 1] - obs[1])**2)
            if dist < obs[2] + margin:
                n += 1
    return n


def differentiable_rollout_obstacle(model, start_state, obstacles, goal_pos,
                                     horizon, dt, normalize, max_pos, max_vel,
                                     max_action, cost_cfg):
    """Differentiable rollout for obstacle cost-based FT."""
    obs_norm = obstacles / max_pos if normalize else obstacles
    obs_positions = jnp.array(obs_norm[:, :2])
    obs_values = jnp.array(obs_norm[:, 2:3])
    times = jnp.linspace(0, 1, horizon + 1)

    def step_fn(carry, t_idx):
        state, cum_cost = carry
        if normalize:
            s_norm = jnp.concatenate([state[:2] / max_pos, state[2:] / max_vel])
        else:
            s_norm = state
        tgt = jnp.concatenate([s_norm, jnp.array([times[t_idx]])])
        action = model(obs_positions, obs_values, tgt) * max_action

        c_coll = collision_cost(state[:2], obstacles, cost_cfg['margin'], cost_cfg['sharpness'])
        c_goal = jnp.sum((state[:2] - goal_pos)**2)
        c_ctrl = jnp.sum(action**2)
        cost = (cost_cfg['collision_weight'] * c_coll +
                cost_cfg['goal_weight'] * c_goal +
                cost_cfg['control_weight'] * c_ctrl)

        next_state = dynamics_step_rk4(state, action, dt)
        return (next_state, cum_cost + cost), None

    (final, cum_cost), _ = jax.lax.scan(step_fn, (start_state, 0.0), jnp.arange(horizon))
    terminal = cost_cfg['terminal_weight'] * jnp.sum((final[:2] - goal_pos)**2 + final[2:]**2)
    return cum_cost + terminal


def finetune_obstacle(model, obstacles, goal_pos, train_starts,
                       horizon, dt, normalize, max_pos, max_vel, max_action,
                       cost_cfg, grad_steps, lr, mode='ft'):
    """Cost-based fine-tuning for obstacle avoidance."""
    ft = copy.deepcopy(model)
    filt = make_filter_spec(ft, mode)
    trainable, frozen = eqx.partition(ft, filt)
    opt = optax.adam(lr)
    opt_state = opt.init(trainable)

    obstacles_jnp = jnp.array(obstacles)
    goal_jnp = jnp.array(goal_pos)

    @eqx.filter_jit
    def step(tr, os):
        def loss_fn(tr):
            m = eqx.combine(tr, frozen)
            total = jnp.array(0.0)
            for s in train_starts:
                total = total + differentiable_rollout_obstacle(
                    m, jnp.array(s), obstacles_jnp, goal_jnp,
                    horizon, dt, normalize, max_pos, max_vel, max_action, cost_cfg)
            return total / len(train_starts)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(tr)
        updates, os_new = opt.update(grads, os)
        return eqx.apply_updates(tr, updates), os_new, loss

    for _ in range(grad_steps):
        trainable, opt_state, _ = step(trainable, opt_state)
    return eqx.combine(trainable, frozen)


def run_obstacle(cfg, data_dir, checkpoint_path, output_dir, seed):
    """Run obstacle avoidance cost-based evaluation (Figure 7b)."""
    print("=" * 60)
    print("Figure 7b: Obstacle Avoidance Cost-Based Adaptation")
    print("=" * 60)

    from src.envs.dataloader import ObstacleAvoidanceImitation

    key = jr.PRNGKey(seed)

    expert_data = np.load(str(Path(data_dir) / "trajectories.npy"), allow_pickle=True).item()
    split_seed = cfg.get("training", {}).get("split_seed", 42)
    data_loader = ObstacleAvoidanceImitation(
        expert_data, train_perc=0.8, normalize=True, seed=split_seed)

    max_pos = float(data_loader.max_pos)
    max_vel = float(data_loader.max_vel)
    max_action = float(data_loader.max_action)
    normalize = True

    T = cfg["data"]["T"]
    N_steps = cfg["data"]["N"]
    dt = T / N_steps
    qf = cfg["data"]["qf"]
    goal_pos = np.array([2.0, 2.0])

    key, mk = jr.split(key)
    model = load_setonet(Path(checkpoint_path) / "setonet.eqx",
                         cfg["model"], 2, 1, 5, 2, mk)

    cost_cfg = {
        'collision_weight': 10.0, 'goal_weight': 0.0,
        'control_weight': dt, 'terminal_weight': qf,
        'margin': 0.2, 'sharpness': 15.0,
    }

    num_tasks = 10
    ft_steps = 200
    ft_lr = 1e-5
    num_ft_starts = 10
    collision_margin = 0.2

    # Select test tasks
    test_tasks = data_loader.test_data[:num_tasks]
    print(f"Evaluating {len(test_tasks)} test tasks")

    all_results = {
        'Expert': {'collisions': [], 'trajectories_by_task': []},
        'SetONet': {'collisions': [], 'trajectories_by_task': []},
        'FT': {'collisions': [], 'trajectories_by_task': []},
        'Last-Branch': {'collisions': [], 'trajectories_by_task': []},
        'Last-Both': {'collisions': [], 'trajectories_by_task': []},
    }

    for task_i, task in enumerate(test_tasks):
        obstacles = task['obstacles']
        expert_states_raw = task['states']  # (num_traj, 4, N+1)
        num_traj = expert_states_raw.shape[0]

        # Get starting conditions from expert data
        starts = [np.array(expert_states_raw[j, :, 0]) for j in range(min(num_ft_starts, num_traj))]

        # Fine-tune
        print(f"  Task {task_i+1}/{len(test_tasks)}: {len(obstacles)} obstacles, FT...")
        ft_model = finetune_obstacle(model, obstacles, goal_pos, starts,
                                      N_steps, dt, normalize, max_pos, max_vel, max_action,
                                      cost_cfg, ft_steps, ft_lr, 'ft')
        lb_model = finetune_obstacle(model, obstacles, goal_pos, starts,
                                      N_steps, dt, normalize, max_pos, max_vel, max_action,
                                      cost_cfg, ft_steps, ft_lr, 'last_branch')
        lboth_model = finetune_obstacle(model, obstacles, goal_pos, starts,
                                         N_steps, dt, normalize, max_pos, max_vel, max_action,
                                         cost_cfg, ft_steps, ft_lr, 'last_both')

        # Evaluate all methods on all trajectories
        task_trajs = {name: [] for name in all_results}
        for j in range(num_traj):
            start = np.array(expert_states_raw[j, :, 0])
            expert_traj = expert_states_raw[j].T  # (N+1, 4)

            # Expert
            all_results['Expert']['collisions'].append(
                count_collisions(expert_traj, obstacles, collision_margin))
            task_trajs['Expert'].append(expert_traj)

            # Pretrained
            X_pre, _ = rollout_obstacle(model, start, obstacles, N_steps, dt,
                                         normalize, max_pos, max_vel, max_action)
            all_results['SetONet']['collisions'].append(
                count_collisions(np.array(X_pre), obstacles, collision_margin))
            task_trajs['SetONet'].append(np.array(X_pre))

            # FT variants
            for name, m in [('FT', ft_model), ('Last-Branch', lb_model), ('Last-Both', lboth_model)]:
                X, _ = rollout_obstacle(m, start, obstacles, N_steps, dt,
                                         normalize, max_pos, max_vel, max_action)
                all_results[name]['collisions'].append(
                    count_collisions(np.array(X), obstacles, collision_margin))
                task_trajs[name].append(np.array(X))

        # Store per-task data for plotting
        for name in all_results:
            all_results[name]['trajectories_by_task'].append({
                'obstacles': obstacles,
                'trajectories': np.array(task_trajs[name]),
            })

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_dict = {'goal_position': goal_pos}
    for name, data in all_results.items():
        key_name = name.lower().replace('-', '_')
        save_dict[f'{key_name}_collisions'] = np.array(data['collisions'])

    # Save per-task trajectory data for plotting (as pickle-able dict)
    plot_data = []
    for task_i in range(len(test_tasks)):
        task_plot = {
            'obstacles': test_tasks[task_i]['obstacles'],
            'num_obstacles': len(test_tasks[task_i]['obstacles']),
        }
        for name in all_results:
            key_name = name.lower().replace('-', '_')
            task_plot[f'{key_name}_trajectories'] = all_results[name]['trajectories_by_task'][task_i]['trajectories']
        plot_data.append(task_plot)
    save_dict['plot_data'] = np.array(plot_data, dtype=object)

    np.savez(output_path / "results.npz", **save_dict, allow_pickle=True)
    print(f"\nResults saved to {output_path / 'results.npz'}")

    print("\nCollision Count (total):")
    for name, data in all_results.items():
        c = np.sum(data['collisions'])
        print(f"  {name:15s}: {c}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cost-based adaptation (Figure 7)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ood", action="store_true", help="Run P2P-Cost OOD variant")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generator = cfg.get("generator", "")

    if args.ood or generator == "p2p_cost":
        run_p2p_ood(cfg, args.data, args.checkpoint, args.output, args.seed)
    elif generator == "obstacle":
        run_obstacle(cfg, args.data, args.checkpoint, args.output, args.seed)
    else:
        print(f"Unknown generator: {generator}")


if __name__ == "__main__":
    main()
