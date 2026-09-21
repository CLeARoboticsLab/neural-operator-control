"""Data loader for the iMuJoCo imitation-learning benchmark (Patacchiola et al., 2023).

The benchmark ships pre-collected SAC expert rollouts, one ``.npz`` file per
dynamics configuration (task). Each file holds a single ``data`` array of shape
``(num_steps, obs_size + act_size + extra + 1)`` whose last column is a ``done``
flag that separates episodes.

Only HalfCheetah-v3 is used in the paper, but the loader also knows the
dimensions of Hopper-v3 and Walker2d-v3 so the same pipeline can be reused.

Context representation (see paper Appendix B): a context point is
``location = (state, action)`` and ``value = next_state``; the trunk maps a
query ``state`` to the expert ``action``.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np


ENV_DIMS = {
    "Hopper-v3": {"obs_size": 11, "act_size": 3, "prefix": "hopper"},
    "HalfCheetah-v3": {"obs_size": 17, "act_size": 6, "prefix": "halfcheetah"},
    "Walker2d-v3": {"obs_size": 17, "act_size": 6, "prefix": "walker"},
}


def discover_files(data_dir: str, prefix: str) -> List[str]:
    """Recursively find the benchmark ``.npz`` files for one environment."""
    found = []
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".npz") and prefix in f.lower():
                found.append(os.path.join(root, f))
    return sorted(found)


class IMuJoCoImitation:
    """Per-task episode store with a config-level train/test split.

    Args:
        data_dir: directory containing (possibly nested) ``*<prefix>*.npz`` files.
        env_name: one of ``ENV_DIMS``.
        train_perc: fraction of configurations used for training (paper: 0.8).
        seed: seed of the configuration split (paper: 42).
        normalize: z-score states and actions with training-set statistics.
    """

    def __init__(
        self,
        data_dir: str,
        env_name: str,
        train_perc: float = 0.8,
        seed: int = 42,
        normalize: bool = True,
    ):
        if env_name not in ENV_DIMS:
            raise ValueError(f"Unknown env: {env_name}. Must be one of {list(ENV_DIMS)}")

        self.env_name = env_name
        self.obs_size = ENV_DIMS[env_name]["obs_size"]
        self.act_size = ENV_DIMS[env_name]["act_size"]
        self.prefix = ENV_DIMS[env_name]["prefix"]
        self.normalize = normalize

        files_list = discover_files(data_dir, self.prefix)
        if not files_list:
            raise FileNotFoundError(
                f"No .npz files for {env_name} under {data_dir}. "
                f"Run `make data-halfcheetah` (downloads the iMuJoCo dataset)."
            )
        self.all_files = files_list

        rng = random.Random(seed)
        shuffled = list(files_list)
        rng.shuffle(shuffled)
        num_train = int(len(shuffled) * train_perc)
        self.train_files = sorted(shuffled[:num_train])
        self.test_files = sorted(shuffled[num_train:])

        self.train_data = self._load_files(self.train_files)
        self.test_data = self._load_files(self.test_files)

        self.state_mean = np.zeros(self.obs_size, dtype=np.float32)
        self.state_std = np.ones(self.obs_size, dtype=np.float32)
        self.action_mean = np.zeros(self.act_size, dtype=np.float32)
        self.action_std = np.ones(self.act_size, dtype=np.float32)
        if self.normalize:
            all_s, all_a = [], []
            for task_episodes in self.train_data:
                for ep in task_episodes:
                    s, a, _ = self._get_transitions_raw(ep)
                    all_s.append(s)
                    all_a.append(a)
            all_s = np.concatenate(all_s, axis=0)
            all_a = np.concatenate(all_a, axis=0)
            self.state_mean = np.mean(all_s, axis=0).astype(np.float32)
            self.state_std = (np.std(all_s, axis=0) + 1e-8).astype(np.float32)
            self.action_mean = np.mean(all_a, axis=0).astype(np.float32)
            self.action_std = (np.std(all_a, axis=0) + 1e-8).astype(np.float32)

        print(f"IMuJoCoImitation: {env_name}")
        print(f"  Configs: {len(files_list)} total, "
              f"{len(self.train_files)} train, {len(self.test_files)} test")
        print(f"  obs_size={self.obs_size}, act_size={self.act_size}, normalize={self.normalize}")
        print(f"  Episodes: {sum(len(t) for t in self.train_data)} train, "
              f"{sum(len(t) for t in self.test_data)} test")

    # ------------------------------------------------------------------ loading

    def _load_files(self, files: List[str]) -> List[List[np.ndarray]]:
        """Parse each file into a list of ``(T, obs+act)`` episode arrays."""
        all_tasks = []
        for fpath in files:
            data = np.load(fpath)["data"]
            dones = np.where(data[:, -1] == 1.0)[0]
            episodes = []
            curr_idx = 0
            for done_idx in dones:
                ep = data[curr_idx:done_idx + 1]
                obs_act = ep[:, :self.obs_size + self.act_size].astype(np.float32)
                if len(obs_act) >= 2:
                    episodes.append(obs_act)
                curr_idx = done_idx + 1
            all_tasks.append(episodes)
        return all_tasks

    def _get_transitions_raw(self, episode: np.ndarray):
        states = episode[:-1, :self.obs_size]
        actions = episode[:-1, self.obs_size:self.obs_size + self.act_size]
        next_states = episode[1:, :self.obs_size]
        return states, actions, next_states

    def get_transitions(self, episode: np.ndarray):
        """``(state, action, next_state)`` transitions of one episode (normalized)."""
        states, actions, next_states = self._get_transitions_raw(episode)
        if self.normalize:
            states = (states - self.state_mean) / self.state_std
            actions = (actions - self.action_mean) / self.action_std
            next_states = (next_states - self.state_mean) / self.state_std
        return states, actions, next_states

    def denormalize_actions(self, actions):
        if self.normalize:
            return actions * self.action_std + self.action_mean
        return actions

    # ----------------------------------------------------------------- sampling

    def sample_window(self, episode: np.ndarray, H: int, rng: Optional[np.random.RandomState] = None):
        """Random contiguous window of at most ``H`` transitions from an episode."""
        s, a, ns = self.get_transitions(episode)
        if len(s) <= H:
            return s, a, ns
        gen = rng if rng is not None else np.random
        start = gen.randint(0, len(s) - H + 1)
        return s[start:start + H], a[start:start + H], ns[start:start + H]

    @staticmethod
    def pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) >= target_len:
            return arr[:target_len]
        padding = np.zeros((target_len - len(arr), arr.shape[1]), dtype=arr.dtype)
        return np.concatenate([arr, padding], axis=0)

    def _episodes_to_arrays(self, episodes, indices, H, target_len):
        s_l, a_l, ns_l = [], [], []
        for idx in indices:
            s, a, ns = self.sample_window(episodes[idx], H)
            s_l.append(s)
            a_l.append(a)
            ns_l.append(ns)
        s = self.pad_or_truncate(np.concatenate(s_l, axis=0), target_len)
        a = self.pad_or_truncate(np.concatenate(a_l, axis=0), target_len)
        ns = self.pad_or_truncate(np.concatenate(ns_l, axis=0), target_len)
        return s, a, ns

    def sample(self, type_: str = "train", M: int = 16, K: int = 3, H: int = 100):
        """Sample ``M`` tasks; for each, ``K`` context episodes and ``K`` query episodes.

        Returns ``(context_sa, context_ns, query_states, query_actions)`` with shapes
        ``(M, K*H, obs+act)``, ``(M, K*H, obs)``, ``(M, K*H, obs)``, ``(M, K*H, act)``.
        """
        data = self.train_data if type_ == "train" else self.test_data
        target_len = K * H
        ctx_sa_all, ctx_ns_all, q_s_all, q_a_all = [], [], [], []

        for _ in range(M):
            episodes = data[np.random.randint(len(data))]
            ep_indices = np.random.choice(len(episodes), size=2 * K, replace=True)
            ctx_s, ctx_a, ctx_ns = self._episodes_to_arrays(episodes, ep_indices[:K], H, target_len)
            q_s, q_a, _ = self._episodes_to_arrays(episodes, ep_indices[K:], H, target_len)
            ctx_sa_all.append(np.concatenate([ctx_s, ctx_a], axis=-1))
            ctx_ns_all.append(ctx_ns)
            q_s_all.append(q_s)
            q_a_all.append(q_a)

        return (
            jnp.array(np.stack(ctx_sa_all)),
            jnp.array(np.stack(ctx_ns_all)),
            jnp.array(np.stack(q_s_all)),
            jnp.array(np.stack(q_a_all)),
        )

    # ---------------------------------------------------------------- task data

    def get_task_data(self, type_: str, task_idx: int) -> Dict:
        data = self.train_data if type_ == "train" else self.test_data
        episodes = data[task_idx]
        states, actions, next_states = [], [], []
        for ep in episodes:
            s, a, ns = self.get_transitions(ep)
            states.append(s)
            actions.append(a)
            next_states.append(ns)
        return {"states": states, "actions": actions, "next_states": next_states,
                "num_episodes": len(episodes)}
