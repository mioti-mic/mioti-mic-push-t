# scripts/pusht_env_smoke_lerobot_runtime.py
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

import lerobot
import gymnasium as gym
import gym_pusht


def main() -> int:
    p = argparse.ArgumentParser(description = "Smoke test: PushT env runs with lerobot[pusht] runtime installed.")
    p.add_argument("--env-id", type = str, default = "gym_pusht/PushT-v0")
    p.add_argument("--seed", type = int, default = 0)
    p.add_argument("--horizon", type = int, default = 50)
    p.add_argument("--render", action = "store_true", help = "Render one rgb_array frame at the end.")
    args = p.parse_args()

    
    render_mode = "rgb_array" if args.render else None
    env = gym.make(args.env_id, obs_type = "state", render_mode = render_mode)

    obs, info = env.reset(seed=args.seed)

    total_reward = 0.0
    steps = 0
    done = False
    last_info: Dict[str, Any] = {}

    for t in range(args.horizon):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps = t + 1
        done = bool(terminated or truncated)
        if isinstance(info, dict):
            last_info = info
        if done:
            break

    frame_shape: Optional[list[int]] = None
    if args.render:
        frame = env.render()
        shape = getattr(frame, "shape", None)
        if shape is not None:
            frame_shape = list(shape)

    env.close()

    result = {
        "env_id": args.env_id,
        "seed": args.seed,
        "horizon": args.horizon,
        "steps": steps,
        "done": done,
        "total_reward": total_reward,
        "obs_shape": list(getattr(obs, "shape", [])) if getattr(obs, "shape", None) is not None else None,
        "frame_shape": frame_shape,
        "versions": {
            "lerobot": getattr(lerobot, "__version__", "unknown"),
            "gymnasium": getattr(gym, "__version__", "unknown"),
        },

        # No serializamos todo info (puede traer arrays); dejamos keys como evidencia
        "last_info_keys": sorted(list(last_info.keys())) if isinstance(last_info, dict) else [],
    }

    print("PUSHT ENV (LEROBOT RUNTIME): OK")
    print(json.dumps(result, indent = 2, sort_keys = True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
