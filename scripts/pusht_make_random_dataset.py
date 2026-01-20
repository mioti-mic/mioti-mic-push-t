from __future__ import annotations

import argparse
import os
import subprocess
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np

from src.data.rollouts.buffer import RolloutBuffer
from src.data.rollouts.export_hf import ExportConfig, export_save_and_push


def get_git_commit() -> str:
    # intenta leer commit; si no hay git en Colab, devuelve ""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr = subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def generate_random_rollouts(
    *,
    num_episodes: int,
    horizon: int,
    master_seed: int,
    env_id: str,
    env_kwargs: Dict,
    obs_dim: int = 5,
    act_dim: int = 2,
) -> Tuple[Dict, List[Dict]]:
    # capacity exacta en MVP (si cortas en done puede sobrar algo, pero no pasa nada):
    buf = RolloutBuffer(capacity_steps = num_episodes * horizon, obs_dim = obs_dim, act_dim = act_dim)

    episodes_rows: List[Dict] = []

    for ep in range(num_episodes):
        ep_seed = int(master_seed + ep)  # determinista, simple

        env = gym.make(env_id, **env_kwargs)
        obs, info = env.reset(seed = ep_seed)

        ep_return = 0.0
        ep_len = 0
        terminated = False
        truncated = False

        for t in range(horizon):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)

            buf.store_step(
                episode_id = ep,
                t = t,
                obs_state = obs,
                action = action,
                reward = reward,
                terminated = terminated,
                truncated = truncated,
                seed = ep_seed,
                info = info if isinstance(info, dict) else {},
            )

            ep_return += float(reward)
            ep_len += 1
            obs = next_obs

            if terminated or truncated:
                break

        episodes_rows.append(
            {
                "episode_id": int(ep),
                "seed": int(ep_seed),
                "length": int(ep_len),
                "return": float(ep_return),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "env_id": str(env_id),
                "policy": "random",
                "horizon": int(horizon),
            }
        )

        env.close()

    return buf.to_hf_dict(), episodes_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", type = str, required = True, help = "HF dataset repo, e.g. mioti-mic/mioti-mic-push-t")
    p.add_argument("--env-id", type = str, default = "gym_pusht/PushT-v0")
    p.add_argument("--episodes", type = int, default = 200)
    p.add_argument("--horizon", type = int, default = 200)
    p.add_argument("--master-seed", type = int, default = 0)
    p.add_argument("--private", action = "store_true")
    args = p.parse_args()

    # PushT config (tu estado actual)
    env_kwargs = {"obs_type": "state", "render_mode": None}

    steps_dict, episodes_rows = generate_random_rollouts(
        num_episodes = args.episodes,
        horizon = args.horizon,
        master_seed = args.master_seed,
        env_id = args.env_id,
        env_kwargs = env_kwargs,
        obs_dim = 5,
        act_dim = 2,
    )

    cfg = ExportConfig(
        repo_id = args.repo_id,
        env_id = args.env_id,
        policy = "random",
        horizon = args.horizon,
        master_seed = args.master_seed,
        obs_dim = 5,
        act_dim = 2,
        git_commit = get_git_commit(),
        private = bool(args.private),
        schema_version = "v1",
    )

    result = export_save_and_push(steps_dict = steps_dict, episodes_rows = episodes_rows, cfg = cfg)
    print(f"[OK] Uploaded run_id={result['run_id']}")
    print(f"[OK] Local artifacts: {result['out_dir']}")


if __name__ == "__main__":
    main()
