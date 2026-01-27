#!/usr/bin/env python3
"""
Record videos without gymnasium.RecordVideo (no moviepy required).
Uses imageio + ffmpeg plugin.

Install:
  python3.10 -m pip install -U imageio imageio-ffmpeg

Usage:
  python3.10 record_best_multigoal_imageio.py \
    --best-model runs_xarm_push_multigoal_sac/.../best/best_model.zip \
    --vecnorm   runs_xarm_push_multigoal_sac/.../best/vecnormalize.pkl \
    --outdir    videos_best \
    --episodes-per-goal 5 \
    --device cuda \
    --fps 30 \
    --deterministic
"""

import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import imageio.v2 as imageio
import gymnasium as gym
import gym_xarm  # noqa: F401

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ENV_ID = "gym_xarm/XarmLift-v0"

GOAL_DIRS: Dict[int, np.ndarray] = {
    0: np.array([+1.0,  0.0], dtype=np.float64),  # forward (+X)
    1: np.array([ 0.0, +1.0], dtype=np.float64),  # left    (+Y)
    2: np.array([ 0.0, -1.0], dtype=np.float64),  # right   (-Y)
    3: np.array([-1.0,  0.0], dtype=np.float64),  # back    (-X)
}
GOAL_NAMES = {0: "forward", 1: "left", 2: "right", 3: "back"}


@dataclass
class PushRewardConfig:
    success_dist: float = 0.10
    lateral_tol: float = 0.05


class GoalConditionedDirectionalPush(gym.Wrapper):
    """Adds 4-d one-hot goal to observation (28->32) and computes is_success for logs."""

    def __init__(self, env: gym.Env, cfg: PushRewardConfig):
        super().__init__(env)
        self.cfg = cfg

        low = env.observation_space.low.astype(np.float64).reshape(-1)
        high = env.observation_space.high.astype(np.float64).reshape(-1)
        assert low.shape[0] == 28

        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, np.zeros(4, dtype=np.float64)]),
            high=np.concatenate([high, np.ones(4, dtype=np.float64)]),
            shape=(32,),
            dtype=np.float64,
        )

        self._goal_id = 0
        self._goal_onehot = np.zeros(4, dtype=np.float64)
        self._start_obj_xy: Optional[np.ndarray] = None

    def _get_obj(self) -> np.ndarray:
        return np.array(self.env.unwrapped.obj, dtype=np.float64)

    def _set_goal(self, goal_id: int):
        self._goal_id = int(goal_id)
        self._goal_onehot = np.zeros(4, dtype=np.float64)
        self._goal_onehot[self._goal_id] = 1.0

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64).reshape(-1)
        return np.concatenate([obs, self._goal_onehot], axis=0)

    def reset(self, **kwargs):
        options = kwargs.get("options", None)
        options = {} if options is None else dict(options)
        goal_id = int(options.get("goal_id", 0))
        self._set_goal(goal_id)

        obs, info = self.env.reset(**kwargs)
        obj = self._get_obj()
        self._start_obj_xy = obj[:2].copy()

        info = dict(info) if isinstance(info, dict) else {}
        info["goal_id"] = self._goal_id
        info["goal_name"] = GOAL_NAMES[self._goal_id]
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obj = self._get_obj()
        assert self._start_obj_xy is not None
        disp_xy = obj[:2] - self._start_obj_xy

        dir_xy = GOAL_DIRS[self._goal_id]
        orth_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)
        proj = float(np.dot(disp_xy, dir_xy))
        orth = float(np.dot(disp_xy, orth_xy))

        success = (proj >= self.cfg.success_dist) and (abs(orth) <= self.cfg.lateral_tol)

        info = dict(info) if isinstance(info, dict) else {}
        info["goal_id"] = self._goal_id
        info["goal_name"] = GOAL_NAMES[self._goal_id]
        info["is_success"] = 1.0 if success else 0.0
        info["debug_proj"] = proj
        info["debug_orth"] = orth

        return self._augment_obs(obs), float(reward), terminated, truncated, info


def make_wrapped_env(render_mode: Optional[str]):
    env = gym.make(ENV_ID, render_mode=render_mode)
    env = GoalConditionedDirectionalPush(env, cfg=PushRewardConfig())
    return env


def load_vecnorm_stats(vecnorm_path: str):
    # Must match obs shape (32) -> use wrapped env for loading
    dummy = DummyVecEnv([lambda: make_wrapped_env(render_mode=None)])
    vn = VecNormalize.load(vecnorm_path, dummy)
    vn.training = False
    vn.norm_reward = False
    obs_rms = vn.obs_rms
    clip_obs = float(vn.clip_obs)
    eps = float(getattr(vn, "epsilon", 1e-8))
    dummy.close()
    return obs_rms, clip_obs, eps


def normalize_obs(obs: np.ndarray, obs_rms, clip_obs: float, eps: float) -> np.ndarray:
    obs = obs.astype(np.float32)
    mean = obs_rms.mean.astype(np.float32)
    var = obs_rms.var.astype(np.float32)
    obs = (obs - mean) / np.sqrt(var + eps)
    return np.clip(obs, -clip_obs, clip_obs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best-model", required=True, type=str)
    p.add_argument("--vecnorm", required=True, type=str)
    p.add_argument("--outdir", default="videos_best", type=str)
    p.add_argument("--episodes-per-goal", default=3, type=int)
    p.add_argument("--max-steps", default=300, type=int)
    p.add_argument("--fps", default=30, type=int)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    args = p.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"run_{timestamp}")
    os.makedirs(outdir, exist_ok=True)

    model = SAC.load(args.best_model, device=args.device)
    obs_rms, clip_obs, eps = load_vecnorm_stats(args.vecnorm)

    env = make_wrapped_env(render_mode="rgb_array")

    for goal_id in range(4):
        goal_name = GOAL_NAMES[goal_id]
        goal_dir = os.path.join(outdir, goal_name)
        os.makedirs(goal_dir, exist_ok=True)

        for ep in range(args.episodes_per_goal):
            obs, info = env.reset(options={"goal_id": goal_id})

            frames = []
            done = False
            steps = 0
            ep_succ = 0.0

            # capture initial frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            while (not done) and (steps < args.max_steps):
                obs_n = normalize_obs(obs, obs_rms, clip_obs, eps)
                action, _ = model.predict(obs_n, deterministic=args.deterministic)

                obs, rew, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                steps += 1

                frame = env.render()
                if frame is not None:
                    frames.append(frame)

                if "is_success" in info:
                    ep_succ = max(ep_succ, float(info["is_success"]))

            fname = f"{goal_name}_ep{ep+1:02d}_succ{int(ep_succ)}_steps{steps:03d}.{args.format}"
            fpath = os.path.join(goal_dir, fname)

            if args.format == "mp4":
                imageio.mimsave(fpath, frames, fps=args.fps)
            else:
                imageio.mimsave(fpath, frames, duration=1.0 / args.fps)

            print(f"[{goal_name}] ep {ep+1}/{args.episodes_per_goal} saved {fpath} success={ep_succ:.0f}")

    env.close()
    print(f"All videos saved under: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
