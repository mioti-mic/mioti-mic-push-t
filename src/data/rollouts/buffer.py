from __future__ import annotations

import numpy as np


class RolloutBuffer:
    """
    Almacena steps secuencialmente (episode_id, t, obs, action, reward, done...)
    y campos mínimos del info para análisis offline.
    """

    def __init__(self, capacity_steps: int, obs_dim: int, act_dim: int):
        self.capacity = int(capacity_steps)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.episode_id = np.zeros((self.capacity,), dtype = np.int64)
        self.t = np.zeros((self.capacity,), dtype = np.int32)

        self.obs_state = np.zeros((self.capacity, self.obs_dim), dtype = np.float32)
        self.action = np.zeros((self.capacity, self.act_dim), dtype = np.float32)

        self.reward = np.zeros((self.capacity,), dtype = np.float32)
        self.terminated = np.zeros((self.capacity,), dtype = np.bool_)
        self.truncated = np.zeros((self.capacity,), dtype = np.bool_)
        self.done = np.zeros((self.capacity,), dtype = np.bool_)

        self.seed = np.zeros((self.capacity,), dtype = np.int64)

        # info mínimo útil (offline metrics / debugging)
        self.is_success = np.zeros((self.capacity,), dtype = np.bool_)
        self.coverage = np.zeros((self.capacity,), dtype = np.float32)
        self.n_contacts = np.zeros((self.capacity,), dtype = np.int32)

        self._idx = 0

    def __len__(self) -> int:
        return self._idx

    def is_full(self) -> bool:
        return self._idx >= self.capacity

    def store_step(
        self,
        *,
        episode_id: int,
        t: int,
        obs_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        seed: int,
        info: dict,
    ) -> None:
        if self.is_full():
            raise IndexError(
                f"RolloutBuffer lleno: capacity_steps={self.capacity}. "
                "Aumenta capacity o reduce episodes*horizon."
            )

        i = self._idx
        self.episode_id[i] = int(episode_id)
        self.t[i] = int(t)

        self.obs_state[i] = np.asarray(obs_state, dtype = np.float32)
        self.action[i] = np.asarray(action, dtype = np.float32)

        self.reward[i] = float(reward)
        self.terminated[i] = bool(terminated)
        self.truncated[i] = bool(truncated)
        self.done[i] = bool(terminated or truncated)

        self.seed[i] = int(seed)

        # info (defensivo: defaults razonables)
        self.is_success[i] = bool(info.get("is_success", False))
        self.coverage[i] = float(info.get("coverage", 0.0))
        self.n_contacts[i] = int(info.get("n_contacts", 0))

        self._idx += 1

    def to_hf_dict(self) -> dict:
        """Devuelve dict columnar listo para Dataset.from_dict(features=...)."""
        n = self._idx
        return {
            "episode_id": self.episode_id[:n],
            "t": self.t[:n],
            "obs_state": self.obs_state[:n],
            "action": self.action[:n],
            "reward": self.reward[:n],
            "terminated": self.terminated[:n],
            "truncated": self.truncated[:n],
            "done": self.done[:n],
            "seed": self.seed[:n],
            "is_success": self.is_success[:n],
            "coverage": self.coverage[:n],
            "n_contacts": self.n_contacts[:n],
        }
