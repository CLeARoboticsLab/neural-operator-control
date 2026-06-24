import os
# Force single-threaded BLAS/LAPACK inside each IPOPT solve so that
# process-level parallelism (multiple workers) actually gives a speedup.
# Must be set before numpy/casadi are imported.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import casadi as ca
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def generate_obstacles(min_obstacles, max_obstacles, max_x, max_y, radius_min, radius_max,
                       goal_pos, min_clearance=2.0):
    """Generate random obstacles that don't overlap with each other or goal."""
    num_obstacles = int(np.random.randint(min_obstacles, max_obstacles + 1))
    placed_obstacles = []

    for i in range(num_obstacles):
        for attempt in range(100):
            x = float(np.random.uniform(-max_x, max_x))
            y = float(np.random.uniform(-max_y, max_y))
            radius = float(np.random.uniform(radius_min, radius_max))
            candidate_pos = np.array([x, y])

            if np.linalg.norm(candidate_pos - goal_pos) <= min_clearance:
                continue

            overlaps = any(
                np.linalg.norm(candidate_pos - e[:2]) < (radius + e[2] + min_clearance)
                for e in placed_obstacles
            )
            if not overlaps:
                placed_obstacles.append(np.array([x, y, radius]))
                break

    return np.vstack(placed_obstacles)


# =============================================================================
# Parametrized NLP builder (called once per worker process)
# =============================================================================

def _build_solver(N, T, qf, num_obs_fixed=4, margin=0.2):
    """
    Build a fully parametrized IPOPT solver.

    Parameters: [p0x, p0y,  obs0_x, obs0_y, obs0_r,  obs1_x, ...]
    lbg is always [0, ...] so this solver works for any obstacle config.
    """
    dt = T / N
    p_target_full = np.array([2.0, 2.0, 0.0, 0.0])

    U = ca.SX.sym('U', 2, N)
    U_flat = ca.reshape(U, -1, 1)
    params = ca.SX.sym('params', 2 + num_obs_fixed * 3)

    x_sym = ca.SX.sym('x', 4)
    u_sym = ca.SX.sym('u', 2)
    xdot_func = ca.Function('f', [x_sym, u_sym],
                            [ca.vertcat(x_sym[2], x_sym[3], u_sym[0], u_sym[1])])

    def rk4_step(xk, uk):
        k1 = xdot_func(xk, uk)
        k2 = xdot_func(xk + dt / 2 * k1, uk)
        k3 = xdot_func(xk + dt / 2 * k2, uk)
        k4 = xdot_func(xk + dt * k3, uk)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    Xk = ca.vertcat(params[0], params[1], 0.0, 0.0)
    all_states = [Xk]
    cost = 0.0
    for k in range(N):
        Uk = U[:, k]
        Xk = rk4_step(Xk, Uk)
        all_states.append(Xk)
        cost += ca.dot(Uk, Uk) * dt
    cost += qf * ca.dot(Xk - p_target_full, Xk - p_target_full)

    g = []
    for k in range(1, N + 1):
        px, py = all_states[k][0], all_states[k][1]
        for i in range(num_obs_fixed):
            cx = params[2 + i * 3 + 0]
            cy = params[2 + i * 3 + 1]
            r  = params[2 + i * 3 + 2]
            g.append((px - cx) ** 2 + (py - cy) ** 2 - (r + margin) ** 2)

    lbg = [0.0] * len(g)
    ubg = [ca.inf] * len(g)

    nlp = {'x': U_flat, 'f': cost, 'g': ca.vertcat(*g), 'p': params}
    opts = {
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.constr_viol_tol': 1e-6,
        'ipopt.acceptable_constr_viol_tol': 1e-4,
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    return solver, lbg, ubg, dt


# =============================================================================
# Worker function (module-level for pickling; solver cached per process)
# =============================================================================

# Lazy per-process cache — populated on first call in each worker.
_cache = {}


NUM_OBS_FIXED = 6  # Max obstacles; tasks with fewer are padded with far-away dummies


def _pad_obstacles(obstacles, num_obs_fixed=NUM_OBS_FIXED):
    """Pad obstacles array to exactly num_obs_fixed rows.

    Missing slots are filled with dummy obstacles at (999, 999) with radius 0,
    so their avoidance constraints are trivially satisfied.
    """
    n = len(obstacles)
    if n >= num_obs_fixed:
        return obstacles[:num_obs_fixed]
    dummy = np.array([[999.0, 999.0, 0.0]] * (num_obs_fixed - n))
    return np.vstack([obstacles, dummy])


def _solve_task_worker(args):
    """
    Solve all trajectories for one task.
    Builds and caches the parametrized solver on first call per process.
    """
    task_num, obstacles, pts, N, T, qf = args

    # Lazy-build solver once per worker process
    if 'solver' not in _cache:
        solver, lbg, ubg, dt = _build_solver(N, T, qf, num_obs_fixed=NUM_OBS_FIXED)
        x_sym = ca.SX.sym('x', 4)
        u_sym = ca.SX.sym('u', 2)
        xdot_func = ca.Function('f', [x_sym, u_sym],
                                [ca.vertcat(x_sym[2], x_sym[3], u_sym[0], u_sym[1])])
        _cache.update(solver=solver, lbg=lbg, ubg=ubg, dt=dt,
                      N=N, xdot_func=xdot_func)

    solver    = _cache['solver']
    lbg       = _cache['lbg']
    ubg       = _cache['ubg']
    dt        = _cache['dt']
    N         = _cache['N']
    xdot_func = _cache['xdot_func']

    obs_padded = _pad_obstacles(obstacles, NUM_OBS_FIXED)
    obs_flat = obs_padded.flatten()
    traj_x, traj_u = [], []
    num_failed = 0

    for p0 in pts:
        params_val = np.concatenate([[p0[0], p0[1]], obs_flat])
        try:
            sol = solver(p=params_val, lbg=lbg, ubg=ubg)
            stats = solver.stats()

            if stats['return_status'] not in ['Solve_Succeeded',
                                               'Solved_To_Acceptable_Level']:
                num_failed += 1
                continue

            if np.min(np.array(sol['g']).flatten()) < -1e-4:
                num_failed += 1
                continue

            u_opt = sol['x'].full().flatten().reshape((2, N), order='F')

            X = np.zeros((4, N + 1))
            X[:, 0] = np.array([p0[0], p0[1], 0.0, 0.0])
            for k in range(N):
                xk, uk = X[:, k], u_opt[:, k]
                k1 = np.array(xdot_func(xk, uk)).flatten()
                k2 = np.array(xdot_func(xk + dt / 2 * k1, uk)).flatten()
                k3 = np.array(xdot_func(xk + dt / 2 * k2, uk)).flatten()
                k4 = np.array(xdot_func(xk + dt * k3, uk)).flatten()
                X[:, k + 1] = xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            traj_x.append(X)
            traj_u.append(u_opt)

        except Exception:
            num_failed += 1

    return task_num, obstacles, traj_x, traj_u, num_failed


def generate_obstacle_dataset(num_tasks, num_traj, T, N, qf, num_workers=4, save_path=None, seed=42):
    """Generate the obstacle avoidance dataset.

    Args:
        num_tasks: Number of task instances
        num_traj: Trajectories per task (2x start points generated)
        T: Total time horizon
        N: Number of time steps
        qf: Terminal state cost weight
        num_workers: Parallel IPOPT solver workers
        save_path: Output path (optional)
        seed: Random seed

    Returns:
        Dataset dictionary keyed by task
    """
    np.random.seed(seed)

    num_obs = [2, 4, 6]
    p_target = np.array([2.0, 2.0])
    n_workers = min(mp.cpu_count(), num_workers)

    # Generate all task configs
    print(f"Generating {num_tasks} task configurations...")
    all_tasks = []
    for itr in range(num_tasks):
        num_ = np.random.choice(num_obs)
        obstacles = generate_obstacles(
            num_, num_,
            max_x=2.0, max_y=2.0,
            radius_min=0.5, radius_max=0.5,
            goal_pos=p_target, min_clearance=0.5,
        )
        y_vary = np.linspace(-3.0, 1.0, num_traj)
        pts = [np.array([-3.0, y_]) for y_ in y_vary]
        pts += [np.array([y_, -3.0]) for y_ in y_vary]
        all_tasks.append((itr + 1, obstacles, pts, N, T, qf))

    # Solve in parallel
    print(f"Solving with {n_workers} workers (OMP_NUM_THREADS=1 per worker)...")

    ctx = mp.get_context('spawn')
    dataset = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_solve_task_worker, task): task[0]
                   for task in all_tasks}

        for fut in tqdm(as_completed(futures), total=len(all_tasks), desc="Tasks"):
            task_num, obstacles, traj_x, traj_u, num_failed = fut.result()

            if num_failed > 0:
                print(f"Task {task_num}: {num_failed} trajectories failed")

            if not traj_x:
                print(f"Task {task_num}: all trajectories failed, skipping.")
                continue

            dataset[f"task_{task_num}"] = {
                "obstacles": obstacles,
                "states":    np.array(traj_x),
                "actions":   np.array(traj_u),
            }

    print(f"\nGenerated {len(dataset)}/{num_tasks} tasks.")

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, dataset, allow_pickle=True)
        print(f"Saved to {save_path}")

    return dataset


def run_from_yaml(cfg, output_dir, seed):
    """Entry point called by generate_data.py dispatcher."""
    from pathlib import Path

    data_cfg = cfg['data']
    save_path = str(Path(output_dir) / "trajectories.npy")

    return generate_obstacle_dataset(
        num_tasks=data_cfg['num_tasks'],
        num_traj=data_cfg['num_traj'],
        T=data_cfg['T'],
        N=data_cfg['N'],
        qf=data_cfg['qf'],
        num_workers=data_cfg.get('num_workers', 4),
        save_path=save_path,
        seed=seed,
    )


def main():
    """Standalone entry point using argparse."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Generate obstacle avoidance dataset")
    parser.add_argument("--config", default="configs/obstacle.yaml", help="Path to config YAML")
    parser.add_argument("--output", default="data/obstacle", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_from_yaml(cfg, args.output, args.seed)
    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
