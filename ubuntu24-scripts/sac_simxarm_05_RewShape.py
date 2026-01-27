#!/usr/bin/env python3
"""
SAC for gym_xarm/XarmLift-v0 with a phased reward wrapper:
- Phase 1: reach shaping (potential-based progress)
- Phase 2: grasp proxy reward (being close + "closing" action proxy)
- Phase 3: lift reward (object height above initial, plus success bonus)
Also logs max object height in eval.

Works with SB3 VecEnv API (step returns 4 values).
Linux-friendly SubprocVecEnv start_method="fork".
"""

import os
import time
import argparse
from typing import List, Tuple, Optional

import numpy as np
import gymnasium as gym
import gym_xarm  # noqa: F401

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed


ENV_ID = "gym_xarm/XarmLift-v0"


# =========================
# Reward wrapper (phased)
# =========================
class PhasedLiftReward(gym.Wrapper):
    """
    Replaces env reward with a shaped, phased reward:
    - Reach: potential-based progress towards object (distance reduction)
    - Grasp proxy: if close in XY and Z and (proxy) closing action indicates grasp attempt
    - Lift: reward for increasing object z above init_z, + bonus at target

    Requirements (you confirmed these exist):
      env.unwrapped.obj: np.ndarray shape (3,)
      env.unwrapped.eef: np.ndarray shape (3,)
      env.unwrapped._init_z: float
      env.unwrapped.z_target: float

    We do NOT require env.unwrapped._action (absent).
    We infer "closing" from action[-1] if action_dim>=1.
    """

    def __init__(
        self,
        env: gym.Env,
        # weights
        w_reach_progress: float = 2.0,
        w_reach_close: float = 0.2,
        w_grasp: float = 1.0,
        w_lift: float = 10.0,
        success_bonus: float = 25.0,
        # shaping parameters / thresholds
        reach_scale: float = 10.0,   # affects tanh(k*d)
        close_dist: float = 0.04,    # consider "close" when reach_dist < this
        close_xy: float = 0.05,      # close in XY
        close_z: float = 0.05,       # close in Z
        grip_close_threshold: float = 0.2,  # action[-1] > this means "closing"
        # penalties
        action_l2_penalty: float = 1e-3,
    ):
        super().__init__(env)
        self.w_reach_progress = float(w_reach_progress)
        self.w_reach_close = float(w_reach_close)
        self.w_grasp = float(w_grasp)
        self.w_lift = float(w_lift)
        self.success_bonus = float(success_bonus)

        self.reach_scale = float(reach_scale)
        self.close_dist = float(close_dist)
        self.close_xy = float(close_xy)
        self.close_z = float(close_z)
        self.grip_close_threshold = float(grip_close_threshold)
        self.action_l2_penalty = float(action_l2_penalty)

        self._prev_phi: Optional[float] = None
        self._max_obj_z: float = -np.inf

    def _get_obj_eef(self) -> Tuple[np.ndarray, np.ndarray]:
        u = self.env.unwrapped
        obj = np.array(u.obj, dtype=np.float64)
        eef = np.array(u.eef, dtype=np.float64)
        return obj, eef

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obj, eef = self._get_obj_eef()
        d = np.linalg.norm(obj - eef)
        # Potential: use tanh to keep bounded and stable
        self._prev_phi = float(np.tanh(self.reach_scale * d))
        self._max_obj_z = float(obj[2])
        # Enrich info for debugging
        info = dict(info) if isinstance(info, dict) else {}
        info["debug_obj_z"] = float(obj[2])
        info["debug_eef_z"] = float(eef[2])
        info["debug_reach_dist"] = float(d)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obj, eef = self._get_obj_eef()
        self._max_obj_z = max(self._max_obj_z, float(obj[2]))

        # Distances
        diff = obj - eef
        reach_dist = float(np.linalg.norm(diff))
        reach_xy = float(np.linalg.norm(diff[:2]))
        reach_z = float(abs(diff[2]))

        # Potential-based progress shaping
        phi = float(np.tanh(self.reach_scale * reach_dist))
        progress = 0.0 if self._prev_phi is None else (self._prev_phi - phi)
        self._prev_phi = phi

        # Reach reward: progress + small dense term for being close
        r_reach = self.w_reach_progress * progress
        if reach_dist < self.close_dist:
            # reward being close without dominating
            r_reach += self.w_reach_close * (1.0 - (reach_dist / self.close_dist))

        # Grasp proxy:
        # We assume action[-1] > threshold indicates "closing gripper".
        # If you later confirm the sign is opposite, we can flip it.
        grip_cmd = float(action[-1]) if np.size(action) >= 1 else 0.0
        closing = (grip_cmd > self.grip_close_threshold)
        close_enough = (reach_xy < self.close_xy) and (reach_z < self.close_z)

        r_grasp = 0.0
        if close_enough and closing:
            # encourage "close while aligned"
            # scaled by closeness
            closeness = 1.0 - min(1.0, reach_dist / max(self.close_dist, 1e-6))
            r_grasp = self.w_grasp * max(0.0, closeness)

        # Lift reward: object height above initial
        u = self.env.unwrapped
        init_z = float(u._init_z)
        z_target = float(u.z_target)
        # clip lift progress to [0, z_target-init_z]
        lift_progress = float(np.clip(obj[2] - init_z, 0.0, max(1e-6, z_target - init_z)))
        r_lift = self.w_lift * lift_progress

        # Success bonus: prefer env's is_success if provided; else based on z threshold
        info = dict(info) if isinstance(info, dict) else {}
        env_success = None
        if "is_success" in info:
            try:
                env_success = float(info["is_success"])
            except Exception:
                env_success = None

        # Fallback success: if object reaches target-1cm
        fallback_success = 1.0 if obj[2] >= (z_target - 0.01) else 0.0
        success = env_success if env_success is not None else fallback_success
        r_success = self.success_bonus * float(success)

        # Small action penalty to avoid saturating and jitter
        act = np.array(action, dtype=np.float64).ravel()
        r_pen = -self.action_l2_penalty * float(np.dot(act, act))

        new_reward = float(r_reach + r_grasp + r_lift + r_success + r_pen)

        # Add debug info
        info["debug_replaced_reward"] = True
        info["debug_r_reach"] = float(r_reach)
        info["debug_r_grasp"] = float(r_grasp)
        info["debug_r_lift"] = float(r_lift)
        info["debug_r_success"] = float(r_success)
        info["debug_r_pen"] = float(r_pen)
        info["debug_obj_z"] = float(obj[2])
        info["debug_max_obj_z"] = float(self._max_obj_z)
        info["debug_reach_dist"] = float(reach_dist)
        info["debug_grip_cmd"] = float(grip_cmd)

        return obs, new_reward, terminated, truncated, info


# =========================
# Env factories
# =========================
def make_env(rank: int, seed: int, log_dir: str, phased_reward: bool):
    def _init():
        env = gym.make(ENV_ID)
        if phased_reward:
            env = PhasedLiftReward(env)
        env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
        env.reset(seed=seed + rank)
        return env
    return _init


def build_train_env(
    n_envs: int,
    seed: int,
    log_dir: str,
    phased_reward: bool,
    norm_reward: bool,
) -> VecNormalize:
    os.makedirs(log_dir, exist_ok=True)
    set_random_seed(seed)

    if n_envs <= 1:
        venv = DummyVecEnv([make_env(0, seed, log_dir, phased_reward)])
    else:
        venv = SubprocVecEnv([make_env(i, seed, log_dir, phased_reward) for i in range(n_envs)], start_method="fork")

    venv = VecMonitor(venv)
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=bool(norm_reward),
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )
    return venv


def build_eval_env(seed: int, train_env: VecNormalize, phased_reward: bool) -> VecNormalize:
    def _init():
        env = gym.make(ENV_ID)
        if phased_reward:
            env = PhasedLiftReward(env)
        env = Monitor(env)
        env.reset(seed=seed + 10_000)
        return env

    venv = DummyVecEnv([_init])
    venv = VecMonitor(venv)

    eval_env = VecNormalize(venv, training=False, norm_obs=True, norm_reward=False)
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    eval_env.clip_obs = train_env.clip_obs
    eval_env.clip_reward = train_env.clip_reward
    return eval_env


# =========================
# Eval callback (SB3 VecEnv: 4 returns)
# =========================
class SB3VecEvalCallback(BaseCallback):
    """
    Eval on VecEnv with n_envs=1.
    Logs:
      - eval/mean_reward
      - eval/success_rate  (from info['is_success'] if present; else fallback success using obj_z)
      - eval/mean_max_obj_z
    Saves best model by success_rate, tie-breaker mean_reward.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        log_path: str,
        best_model_save_path: str,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.log_path = log_path
        self.best_model_save_path = best_model_save_path
        self.deterministic = deterministic

        self.best_success = -1.0
        self.best_mean_reward = -np.inf

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if (self.num_timesteps % self.eval_freq) != 0:
            return True

        t0 = time.time()
        mean_reward, success_rate, mean_max_obj_z = self._evaluate()
        dt = time.time() - t0

        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/success_rate", float(success_rate))
        self.logger.record("eval/mean_max_obj_z", float(mean_max_obj_z))
        self.logger.record("eval/eval_time_sec", float(dt))

        eval_file = os.path.join(self.log_path, "eval_log.tsv")
        write_header = not os.path.exists(eval_file) or os.path.getsize(eval_file) == 0
        with open(eval_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write("timesteps\tmean_reward\tsuccess_rate\tmean_max_obj_z\teval_time_sec\n")
            f.write(f"{self.num_timesteps}\t{mean_reward:.6f}\t{success_rate:.4f}\t{mean_max_obj_z:.6f}\t{dt:.3f}\n")

        improved = False
        if (success_rate > self.best_success) or (
            np.isclose(success_rate, self.best_success) and (mean_reward > self.best_mean_reward)
        ):
            improved = True
            self.best_success = float(success_rate)
            self.best_mean_reward = float(mean_reward)

            self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            train_venv = self.model.get_env()
            if isinstance(train_venv, VecNormalize):
                train_venv.save(os.path.join(self.best_model_save_path, "vecnormalize.pkl"))

        if self.verbose >= 1:
            print(
                f"[Eval @ {self.num_timesteps}] mean_reward={mean_reward:.3f} "
                f"success_rate={success_rate:.3f} mean_max_obj_z={mean_max_obj_z:.3f} "
                f"{'(BEST)' if improved else ''} (eval {dt:.2f}s)"
            )

        return True

    def _evaluate(self) -> Tuple[float, float, float]:
        ep_rewards: List[float] = []
        ep_success: List[float] = []
        ep_maxz: List[float] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = np.array([False])
            ep_rew = 0.0
            ep_succ = 0.0
            maxz = -np.inf

            while not bool(done[0]):
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, rewards, done, infos = self.eval_env.step(action)  # 4 returns
                info0 = infos[0] if isinstance(infos, (list, tuple)) else infos

                ep_rew += float(rewards[0])

                if isinstance(info0, dict):
                    if "is_success" in info0:
                        try:
                            ep_succ = max(ep_succ, float(info0["is_success"]))
                        except Exception:
                            pass
                    # our wrapper adds this
                    if "debug_max_obj_z" in info0:
                        try:
                            maxz = max(maxz, float(info0["debug_max_obj_z"]))
                        except Exception:
                            pass

            ep_rewards.append(ep_rew)
            ep_success.append(ep_succ)
            # If maxz never updated, set NaN
            ep_maxz.append(float(maxz) if np.isfinite(maxz) else float("nan"))

        return float(np.mean(ep_rewards)), float(np.mean(ep_success)), float(np.nanmean(ep_maxz))


# =========================
# Training
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--logdir", type=str, default="runs_xarm_lift_sac_phased")
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--n-eval-episodes", type=int, default=30)

    # reward wrapper on/off (keep default ON)
    parser.add_argument("--phased-reward", action="store_true", default=True)
    parser.add_argument("--no-phased-reward", dest="phased_reward", action="store_false")

    # normalization options
    parser.add_argument("--norm-reward", action="store_true", default=False)
    parser.add_argument("--norm-reward-on", dest="norm_reward", action="store_true")
    parser.add_argument("--norm-reward-off", dest="norm_reward", action="store_false")

    args = parser.parse_args()

    run_name = time.strftime("xarm_lift_sac_phased_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_log_dir = os.path.join(run_dir, "train_monitors")
    eval_log_dir = os.path.join(run_dir, "eval")
    best_dir = os.path.join(run_dir, "best")
    final_dir = os.path.join(run_dir, "final")
    tb_dir = os.path.join(run_dir, "tb")
    for d in (train_log_dir, eval_log_dir, best_dir, final_dir, tb_dir):
        os.makedirs(d, exist_ok=True)

    # Build envs
    train_env = build_train_env(
        n_envs=args.n_envs,
        seed=args.seed,
        log_dir=train_log_dir,
        phased_reward=args.phased_reward,
        norm_reward=args.norm_reward,
    )
    eval_env = build_eval_env(seed=args.seed, train_env=train_env, phased_reward=args.phased_reward)

    # SAC config: more exploration + slightly stronger function approximator
    policy_kwargs = dict(net_arch=[512, 512])

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=100_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=2,
        ent_coef="auto_0.2",
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_dir,
        device=args.device,
        seed=args.seed,
    )

    eval_cb = SB3VecEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        log_path=eval_log_dir,
        best_model_save_path=best_dir,
        deterministic=True,
        verbose=1,
    )

    print(f"Run dir: {run_dir}")
    print(f"Env: {ENV_ID}")
    print(f"n_envs={args.n_envs}, total_timesteps={args.total_timesteps}, device={args.device}")
    print(f"phased_reward={args.phased_reward}, norm_reward={args.norm_reward}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_cb,
        progress_bar=True,
    )

    # Save final
    model.save(os.path.join(final_dir, "final_model"))
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(final_dir, "vecnormalize.pkl"))

    print("Training complete.")
    print(f"Best model:  {os.path.join(best_dir, 'best_model.zip')}")
    print(f"Final model: {os.path.join(final_dir, 'final_model.zip')}")
    print("Remember: load vecnormalize.pkl for correct inference.")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
