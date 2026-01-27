#!/usr/bin/env python3
import os
import argparse
import subprocess
import numpy as np

import gymnasium as gym
import gym_xarm  # noqa: F401

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


ENV_ID = "gym_xarm/XarmLift-v0"


def make_env(seed: int):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def start_ffmpeg(width, height, fps, out_path):
    """
    Start ffmpeg subprocess that reads raw RGB frames from stdin.
    """
    cmd = [
        "ffmpeg",
        "-y",                       # overwrite
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",                  # stdin
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vecnorm-path", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out-dir", default="videos_xarm_lift")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Build env ---
    base_env = make_env(args.seed)
    venv = DummyVecEnv([lambda: base_env])

    # --- Load VecNormalize ---
    venv = VecNormalize.load(args.vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    # --- Load model ---
    model = SAC.load(args.model_path, env=venv, device=args.device)

    print("Model loaded.")
    print("Recording with ffmpeg.")

    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_rew = 0.0

        # First frame to get resolution
        frame = base_env.render()
        height, width, _ = frame.shape

        video_path = os.path.join(args.out_dir, f"xarm_lift_ep_{ep}.mp4")
        ffmpeg = start_ffmpeg(width, height, args.fps, video_path)

        step = 0
        while not done:
            # write frame
            ffmpeg.stdin.write(frame.tobytes())

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = venv.step(action)

            ep_rew += float(reward[0])
            frame = base_env.render()
            step += 1

        # write last frame
        ffmpeg.stdin.write(frame.tobytes())
        ffmpeg.stdin.close()
        ffmpeg.wait()

        print(f"[Episode {ep+1}] steps={step} return={ep_rew:.3f} video={video_path}")

    venv.close()
    print("\nDone. Videos saved using ffmpeg (no moviepy).")


if __name__ == "__main__":
    main()
