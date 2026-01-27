"""
Overnight-safe demo collection for gym_xarm/XarmLift-v0 with 5 tasks (5 HF datasets),
using SB3 SAC teachers and saving RGB from 2 cameras.

Key fixes vs previous version:
- Only 2 Mujoco envs total (push + lift) -> avoids OOM.
- Push env goal is changed in-place (no 4 separate envs).
- Throttling + render_every_n_steps to avoid freezing desktop.
- Images stored as JPEG bytes to reduce RAM/GC pressure.
- Sharded parquet writing per task.

Requirements:
  pip install stable-baselines3 gymnasium mujoco datasets pillow pyarrow huggingface_hub

Run:
  export MUJOCO_GL=egl
  export HF_TOKEN=...
  tmux new -s xarm_collect
  python collect_xarm_demos_safe.py
"""

import os
import time
import json
import signal
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from PIL import Image as PILImage
import io

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import datasets
from huggingface_hub import HfApi, login


# -----------------------------
# Config
# -----------------------------

@dataclass
class Paths:
    # 4-direction policy (push)
    push_best_model_zip: str = "runs_xarm_push_multigoal_sac/xarm_push_multigoal_sac_20260126_182911/best/best_model.zip"
    push_vecnormalize_pkl: str = "runs_xarm_push_multigoal_sac/xarm_push_multigoal_sac_20260126_182911/best/vecnormalize.pkl"

    # lift policy
    lift_best_model_zip: str = "runs_xarm_lift_sac_phased/xarm_lift_sac_phased_20260126_132859/best/best_model.zip"
    lift_vecnormalize_pkl: str = "runs_xarm_lift_sac_phased/xarm_lift_sac_phased_20260126_132859/best/vecnormalize.pkl"


@dataclass
class CollectConfig:
    env_id: str = "gym_xarm/XarmLift-v0"
    render_mode: str = "rgb_array"
    camera_names: Tuple[str, str] = ("camera0", "camera1")

    # runtime
    max_seconds: int = 8 * 60 * 60  # 8h
    deterministic_teacher: bool = True

    # throttling / stability
    fps_sleep: float = 0.03              # yields CPU/GPU; prevents desktop freeze
    render_every_n_steps: int = 2        # render both cameras every N steps (keeps 2 cams)

    # episode control
    max_ep_steps: int = 400

    # sharding
    shard_steps: int = 2000
    out_root: str = "hf_shards_xarm_safe"

    # HF repos (5 separate datasets)
    hf_repos: Dict[str, str] = None

    # JPEG compression for images
    jpeg_quality: int = 85


# -----------------------------
# Goal wrapper for push model
# -----------------------------

class GoalConditionedDirectionalPush(gym.ObservationWrapper):
    """
    Appends a one-hot goal vector of length 4 to the original observation.
    """
    def __init__(self, env, goal_dim: int = 4):
        super().__init__(env)
        self.goal_dim = goal_dim
        self.current_goal = np.zeros(self.goal_dim, dtype=np.float32)

        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box), "Expected Box observation space"
        low = np.concatenate([obs_space.low, -np.ones(self.goal_dim, dtype=np.float32)], axis=0)
        high = np.concatenate([obs_space.high, np.ones(self.goal_dim, dtype=np.float32)], axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def set_goal_idx(self, idx: int):
        assert 0 <= idx < self.goal_dim
        g = np.zeros(self.goal_dim, dtype=np.float32)
        g[idx] = 1.0
        self.current_goal = g

    def observation(self, observation):
        return np.concatenate([observation.astype(np.float32), self.current_goal], axis=0)


# -----------------------------
# Helpers
# -----------------------------

STOP_REQUESTED = False

def _handle_sig(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[!] Stop requested. Finalizing shards and (optionally) pushing...\n")

signal.signal(signal.SIGINT, _handle_sig)
signal.signal(signal.SIGTERM, _handle_sig)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def hf_login_if_needed():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var not set. Export HF_TOKEN before running.")
    login(token=token)

def get_gym_env_from_vecnorm(vecnorm: VecNormalize):
    return vecnorm.venv.envs[0]

def get_original_obs(vecnorm: VecNormalize):
    # SB3 VecNormalize provides original obs in many versions
    if hasattr(vecnorm, "get_original_obs"):
        o = vecnorm.get_original_obs()
        if isinstance(o, np.ndarray) and o.ndim == 2:
            return o[0].copy()
        return o
    return None

def try_render(env, camera_name: Optional[str] = None) -> np.ndarray:
    if camera_name is None:
        return env.render()
    try:
        return env.render(camera_name=camera_name)
    except TypeError:
        return env.render()

def encode_jpeg_bytes(rgb: np.ndarray, quality: int = 85) -> bytes:
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    img = PILImage.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()


# -----------------------------
# Sharded writers
# -----------------------------

class TaskShardWriter:
    def __init__(self, task: str, out_root: str, shard_steps: int):
        self.task = task
        self.task_dir = os.path.join(out_root, task)
        ensure_dir(self.task_dir)

        self.shard_steps = shard_steps
        self.buffer: List[Dict[str, Any]] = []
        self.shard_idx = 0
        self.total = 0

        self.manifest_path = os.path.join(self.task_dir, "manifest.jsonl")

    def add(self, row: Dict[str, Any]):
        self.buffer.append(row)
        self.total += 1
        if len(self.buffer) >= self.shard_steps:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        shard_name = f"shard_{self.shard_idx:05d}.parquet"
        shard_path = os.path.join(self.task_dir, shard_name)

        ds = datasets.Dataset.from_list(self.buffer)
        ds.to_parquet(shard_path)

        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "shard": shard_name,
                "rows": len(self.buffer),
                "total": self.total,
                "ts": now_ts(),
            }) + "\n")

        print(f"[{self.task}] wrote {len(self.buffer)} rows -> {shard_path}")
        self.buffer.clear()
        self.shard_idx += 1

    def finalize(self):
        self.flush()

def load_task_dataset(task_dir: str) -> datasets.Dataset:
    parquet_files = sorted(
        os.path.join(task_dir, f) for f in os.listdir(task_dir) if f.endswith(".parquet")
    )
    if not parquet_files:
        raise RuntimeError(f"No parquet shards found in {task_dir}")

    ds = datasets.load_dataset("parquet", data_files=parquet_files, split="train")

    # Images are stored as JPEG bytes -> keep as Binary
    # (LeRobot / your loader can decode later; HF viewer won't auto-display unless cast.)
    return ds


# -----------------------------
# Env / model loaders (2 envs total)
# -----------------------------

def make_push_vecnorm_and_model(paths: Paths, cfg: CollectConfig):
    def _make():
        env = gym.make(cfg.env_id, render_mode=cfg.render_mode)
        env = GoalConditionedDirectionalPush(env, goal_dim=4)
        env.set_goal_idx(0)  # default; will change per task
        return env

    venv = DummyVecEnv([_make])
    vec: VecNormalize = VecNormalize.load(paths.push_vecnormalize_pkl, venv)
    vec.training = False
    vec.norm_reward = False

    model = SAC.load(paths.push_best_model_zip, env=vec)
    return vec, model

def make_lift_vecnorm_and_model(paths: Paths, cfg: CollectConfig):
    def _make():
        env = gym.make(cfg.env_id, render_mode=cfg.render_mode)
        return env

    venv = DummyVecEnv([_make])
    vec: VecNormalize = VecNormalize.load(paths.lift_vecnormalize_pkl, venv)
    vec.training = False
    vec.norm_reward = False

    model = SAC.load(paths.lift_best_model_zip, env=vec)
    return vec, model


# -----------------------------
# Main collection
# -----------------------------

def collect(paths: Paths, cfg: CollectConfig):
    ensure_dir(cfg.out_root)

    if cfg.hf_repos is None:
        cfg.hf_repos = {
            "up": "mioti-mic/gym_xarm-XarmLift_v0-up",
            "forward": "mioti-mic/gym_xarm-XarmLift_v0-forward",
            "back": "mioti-mic/gym_xarm-XarmLift_v0-backward",
            "left": "mioti-mic/gym_xarm-XarmLift_v0-left",
            "right": "mioti-mic/gym_xarm-XarmLift_v0-right",
        }

    writers = {
        "up": TaskShardWriter("up", cfg.out_root, cfg.shard_steps),
        "forward": TaskShardWriter("forward", cfg.out_root, cfg.shard_steps),
        "back": TaskShardWriter("back", cfg.out_root, cfg.shard_steps),
        "left": TaskShardWriter("left", cfg.out_root, cfg.shard_steps),
        "right": TaskShardWriter("right", cfg.out_root, cfg.shard_steps),
    }

    # Push goal mapping (your confirmed mapping)
    # 0 forward, 1 left, 2 right, 3 back
    push_goal_idx = {"forward": 0, "left": 1, "right": 2, "back": 3}

    # Instructions (placeholder; you’ll paraphrase later)
    instructions = {
        "up": "Levanta el cubo hacia arriba.",
        "forward": "Mueve el cubo hacia delante.",
        "back": "Mueve el cubo hacia atrás.",
        "left": "Mueve el cubo hacia la izquierda.",
        "right": "Mueve el cubo hacia la derecha.",
    }

    print("[load] push teacher (single env)...")
    push_vec, push_model = make_push_vecnorm_and_model(paths, cfg)
    push_env = get_gym_env_from_vecnorm(push_vec)
    assert hasattr(push_env, "set_goal_idx"), "Push env wrapper missing set_goal_idx"

    print("[load] lift teacher (single env)...")
    lift_vec, lift_model = make_lift_vecnorm_and_model(paths, cfg)
    lift_env = get_gym_env_from_vecnorm(lift_vec)

    start = time.time()
    ep_id = 0

    # Cycle tasks to keep balance
    task_cycle = ["up", "forward", "left", "right", "back"]
    task_i = 0

    # To reduce render calls, we cache last rendered frames
    last_frames = {cam: None for cam in cfg.camera_names}

    try:
        while not STOP_REQUESTED and (time.time() - start) < cfg.max_seconds:
            task = task_cycle[task_i % len(task_cycle)]
            task_i += 1
            ep_id += 1

            if task == "up":
                vec, model, env = lift_vec, lift_model, lift_env
            else:
                # set goal for push
                gi = push_goal_idx[task]
                push_env.set_goal_idx(gi)
                vec, model, env = push_vec, push_model, push_env

            obs = vec.reset()
            steps = 0
            done = False

            while not STOP_REQUESTED and steps < cfg.max_ep_steps:
                action, _ = model.predict(obs, deterministic=cfg.deterministic_teacher)
                obs2, reward, done_arr, infos = vec.step(action)

                done = bool(done_arr[0])

                # proprio: prefer original (pre-normalization) obs
                proprio = get_original_obs(vec)
                if proprio is None:
                    proprio = obs[0].copy()

                # Render only every N steps, but store frames every step
                if steps % cfg.render_every_n_steps == 0:
                    for cam in cfg.camera_names:
                        rgb = try_render(env, camera_name=cam)
                        last_frames[cam] = encode_jpeg_bytes(rgb, quality=cfg.jpeg_quality)

                row = {
                    "task": task,
                    "instruction": instructions[task],
                    "episode_id": int(ep_id),
                    "timestep": int(steps),
                    "reward": float(reward[0]) if isinstance(reward, np.ndarray) else float(reward),
                    "done": bool(done),
                    "proprio": np.asarray(proprio, dtype=np.float32).tolist(),
                    "action": np.asarray(action[0], dtype=np.float32).tolist(),
                    "timestamp": now_ts(),
                    # store JPEG bytes for both cameras
                    f"image_{cfg.camera_names[0]}_jpg": last_frames[cfg.camera_names[0]],
                    f"image_{cfg.camera_names[1]}_jpg": last_frames[cfg.camera_names[1]],
                }

                writers[task].add(row)

                obs = obs2
                steps += 1

                if cfg.fps_sleep > 0:
                    time.sleep(cfg.fps_sleep)

                if done:
                    break

    finally:
        for w in writers.values():
            w.finalize()

        try:
            push_vec.close()
        except Exception:
            pass
        try:
            lift_vec.close()
        except Exception:
            pass

        print("\n[finalize] collection finished; shards finalized.\n")


def push_to_hub(cfg: CollectConfig):
    hf_login_if_needed()
    api = HfApi()

    # Map internal task keys to your repo names
    task_to_repo = {
        "up": cfg.hf_repos["up"],
        "forward": cfg.hf_repos["forward"],
        "back": cfg.hf_repos["back"],
        "left": cfg.hf_repos["left"],
        "right": cfg.hf_repos["right"],
    }

    for task, repo_id in task_to_repo.items():
        task_dir = os.path.join(cfg.out_root, task)
        print(f"[push] loading {task} dataset from {task_dir}")
        ds = load_task_dataset(task_dir)

        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        except Exception as e:
            print(f"[push] create_repo warning (likely exists): {e}")

        print(f"[push] pushing {len(ds)} rows -> {repo_id}")
        ds.push_to_hub(repo_id, split="train")
        print(f"[push] done: {repo_id}\n")


if __name__ == "__main__":
    cfg = CollectConfig(
        camera_names=("camera0", "camera1"),
        hf_repos={
            "up": "mioti-mic/gym_xarm-XarmLift_v0-up",
            "forward": "mioti-mic/gym_xarm-XarmLift_v0-forward",
            "back": "mioti-mic/gym_xarm-XarmLift_v0-backward",
            "left": "mioti-mic/gym_xarm-XarmLift_v0-left",
            "right": "mioti-mic/gym_xarm-XarmLift_v0-right",
        },
        fps_sleep=0.03,
        render_every_n_steps=2,   # keeps 2 cams but halves render load
        shard_steps=2000,
        jpeg_quality=85,
        max_ep_steps=400,
    )

    paths = Paths()

    print("[start] collecting demos (overnight-safe)...")
    collect(paths, cfg)

    print("[start] pushing to Hugging Face...")
    push_to_hub(cfg)
    print("[done] all datasets pushed.")
