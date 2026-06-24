"""Task-parameter extraction for the stored-dataset environments.

These environments (P2P-Dynamics, Quadrotor, Obstacle) ship a fixed dataset on
disk rather than generating data online like P2P-Cost. For the task-conditioned
MLP baseline we need, per task, an explicit parameter vector plus the expert
(state, action) trajectories.

Each extractor returns ``(train_tasks, test_tasks, info)`` where a *task* is::

    {"param": (P,), "states": (n_traj, H, state_dim), "actions": (n_traj, H, action_dim)}

with ``states`` already aligned to ``actions`` (last timestep dropped) and time
*not* yet appended (the trainer/evaluator append normalized time identically).

The train/test split is deterministic in ``split_seed`` so the evaluator can
re-extract the exact held-out tasks the model was trained against. Task-parameter
scaling is applied here (and is therefore identical at train and eval time):
- dynamics envs: z-score with the *training* tasks' param mean/std
- obstacle: obstacle (x,y) divided by a global position scale, padded with -1
"""

import numpy as np
from pathlib import Path


def _split_indices(n, train_perc, split_seed):
    rng = np.random.RandomState(split_seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_perc)
    return idx[:n_train], idx[n_train:]


def dynamics_split(data_dir, split_seed=42, train_perc=0.8):
    """Canonical train/test split for VaryingDynamics-style envs.

    Returns (train_config_ids, test_config_ids, goal_idx). Used by both the
    baseline and the (filtered) pretrained SetONet trainer so the held-out test
    tasks are identical and leak-free.
    """
    raw = dict(np.load(Path(data_dir) / "trajectories.npz", allow_pickle=True))
    dyn_idx = np.asarray(raw["dynamics_indices"])
    goal_idx = np.asarray(raw["goal_indices"])
    g0 = int(np.unique(goal_idx)[0])
    configs = np.unique(dyn_idx[goal_idx == g0])
    tr, te = _split_indices(len(configs), train_perc, split_seed)
    return configs[tr], configs[te], g0


def obstacle_split(data_dir, split_seed=132, train_perc=0.9):  # paper obstacle defaults
    """Canonical train/test split for the Obstacle env. Returns (train_keys, test_keys)."""
    data = np.load(Path(data_dir) / "trajectories.npy", allow_pickle=True).item()
    task_keys = sorted([k for k in data.keys() if k.startswith("task_")])
    tr, te = _split_indices(len(task_keys), train_perc, split_seed)
    return [task_keys[i] for i in tr], [task_keys[i] for i in te]


def restrict_dynamics_dataset(raw, config_ids):
    """Keep only trajectories whose dynamics config is in ``config_ids``.

    Used by the pretrained SetONet trainers so they train on the canonical train
    split (and never see the held-out test tasks). Per-trajectory arrays are
    masked; per-config arrays (dynamics_params, goal_states) and scalars are kept.
    """
    config_ids = np.asarray(config_ids)
    dyn_idx = np.asarray(raw["dynamics_indices"])
    mask = np.isin(dyn_idx, config_ids)
    n = len(dyn_idx)
    out = {}
    for k, v in raw.items():
        if isinstance(v, np.ndarray) and v.ndim > 0 and v.shape[0] == n:
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def restrict_obstacle_dataset(data, keys):
    """Keep only the given task keys (plus any non-task metadata entries)."""
    keep = set(keys)
    return {k: v for k, v in data.items()
            if (not k.startswith("task_")) or k in keep}


def extract_dynamics(data_dir, split_seed=42, train_perc=0.8):
    """Extractor for P2P-Dynamics and Quadrotor (VaryingDynamics-style npz).

    Task = a dynamics configuration; param = its physical dynamics vector
    (z-scored across the training tasks).
    """
    raw = dict(np.load(Path(data_dir) / "trajectories.npz", allow_pickle=True))
    states = np.asarray(raw["states"])          # (T, H+1, state_dim)
    actions = np.asarray(raw["actions"])        # (T, H, action_dim)
    dyn_idx = np.asarray(raw["dynamics_indices"])
    dyn_params = np.asarray(raw["dynamics_params"], dtype=np.float64)  # (num_dyn, P)
    goal_idx = np.asarray(raw["goal_indices"])

    # Match the SetONet trainers: restrict to a single (first) goal.
    g0 = int(np.unique(goal_idx)[0])
    keep = goal_idx == g0

    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]

    # Canonical, deterministic split (shared with the pretrained trainer).
    train_configs, test_configs, _g0 = dynamics_split(data_dir, split_seed, train_perc)

    # Param standardization from training configs only.
    param_mean = dyn_params[train_configs].mean(axis=0)
    param_std = dyn_params[train_configs].std(axis=0) + 1e-8

    def build(config_ids):
        tasks = []
        for c in config_ids:
            mask = keep & (dyn_idx == c)
            tr = np.where(mask)[0]
            if len(tr) == 0:
                continue
            full = states[tr]                   # (n, H+1, state_dim)
            s = full[:, :-1, :]                 # states aligned with actions
            ns = full[:, 1:, :]                 # next-states (SetONet branch values)
            a = actions[tr]
            param = (dyn_params[c] - param_mean) / param_std
            tasks.append({"param": param.astype(np.float32),
                          "states": s.astype(np.float32),
                          "next_states": ns.astype(np.float32),
                          "actions": a.astype(np.float32),
                          "param_raw": dyn_params[c].astype(np.float32),
                          "config_id": int(c)})
        return tasks

    info = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "param_dim": int(dyn_params.shape[-1]),
        "kind": "dynamics",
    }
    return build(train_configs), build(test_configs), info


def extract_obstacle(data_dir, split_seed=132, train_perc=0.9):  # paper obstacle defaults
    """Extractor for the Obstacle environment (dict-of-tasks .npy).

    Task = an obstacle configuration; param = the obstacle (x,y) positions scaled
    by a global position scale and padded with -1 up to the dataset's maximum
    obstacle count.
    """
    data = np.load(Path(data_dir) / "trajectories.npy", allow_pickle=True).item()
    task_keys = sorted([k for k in data.keys() if k.startswith("task_")])

    # Global, deterministic scales over the whole dataset.
    all_obs = np.concatenate([np.asarray(data[k]["obstacles"]) for k in task_keys], axis=0)
    pos_scale = float(np.abs(all_obs[:, :2]).max()) + 1e-8
    max_obs = max(int(np.asarray(data[k]["obstacles"]).shape[0]) for k in task_keys)

    train_keys, test_keys = obstacle_split(data_dir, split_seed, train_perc)

    def make_param(obstacles):
        xy = np.asarray(obstacles)[:, :2] / pos_scale           # (n_obs, 2)
        flat = xy.reshape(-1)
        padded = -np.ones(max_obs * 2, dtype=np.float32)
        padded[:flat.shape[0]] = flat
        return padded

    def build(keys):
        tasks = []
        for k in keys:
            t = data[k]
            s = np.transpose(np.asarray(t["states"]), (0, 2, 1))[:, :-1, :]  # (n, H, 4)
            a = np.transpose(np.asarray(t["actions"]), (0, 2, 1))            # (n, H, 2)
            tasks.append({"param": make_param(t["obstacles"]),
                          "states": s.astype(np.float32),
                          "actions": a.astype(np.float32),
                          "obstacles": np.asarray(t["obstacles"]).astype(np.float32),
                          "task_key": k})
        return tasks

    info = {
        "state_dim": 4,
        "action_dim": 2,
        "param_dim": int(max_obs * 2),
        "max_obs": int(max_obs),
        "pos_scale": pos_scale,
        "kind": "obstacle",
    }
    return build(train_keys), build(test_keys), info
