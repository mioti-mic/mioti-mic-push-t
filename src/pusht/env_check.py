from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Dict

import numpy as np


@dataclass
class SmokeTestResult:
    env_id: str
    seed: int
    horizon: int
    steps: int
    total_reward: float
    done: bool
    obs_shape: Optional[tuple]
    frame_shape: Optional[tuple]
    gymnasium_version: str
    gym_pusht_version: str


def _versions() -> Dict[str, str]:
    import gymnasium as gymn
    import gym_pusht
    return {
        "gymnasium": getattr(gymn, "__version__", "unknown"),
        "gym_pusht": getattr(gym_pusht, "__version__", "unknown"),
    }


def run_smoke_test(
    env_id: str = "gym_pusht/PushT-v0",
    seed: int = 0,
    horizon: int = 300,
    render: bool = False,
) -> SmokeTestResult:
    """Create PushT env, run random rollout, optional render to rgb_array.

    Raises:
        Exception: If env creation/step/render fails.
    """
    import gymnasium as gym
    import gym_pusht

    render_mode = "rgb_array" if render else None
    env = gym.make(env_id, render_mode = render_mode)

    obs, info = env.reset(seed = seed)

    total_reward = 0.0
    steps = 0
    done = False

    for t in range(horizon):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps = t + 1
        done = bool(terminated or truncated)
        if done:
            break

    frame_shape = None
    if render:
        frame = env.render()
        frame_shape = getattr(frame, "shape", None)

    env.close()

    v = _versions()
    obs_shape = getattr(obs, "shape", None)

    return SmokeTestResult(
        env_id = env_id,
        seed = seed,
        horizon = horizon,
        steps = steps,
        total_reward = total_reward,
        done = done,
        obs_shape = obs_shape,
        frame_shape = frame_shape,
        gymnasium_version = v["gymnasium"],
        gym_pusht_version = v["gym_pusht"],
    )


def to_dict(result: SmokeTestResult) -> Dict[str, Any]:
    return asdict(result)
