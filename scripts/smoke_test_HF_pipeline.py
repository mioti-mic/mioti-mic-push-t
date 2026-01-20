def main():
    from src.data.rollouts.buffer import RolloutBuffer
    from src.data.rollouts.export_hf import build_steps_features
    import datasets
    import gymnasium as gym

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
    obs, info = env.reset(seed=0)

    buf = RolloutBuffer(capacity_steps=5, obs_dim=5, act_dim=2)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    buf.store_step(
        episode_id=0,
        t=0,
        obs_state=obs,
        action=action,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        seed=0,
        info=info,
    )

    steps_dict = buf.to_hf_dict()

    ds = datasets.Dataset.from_dict(
        steps_dict,
        features=build_steps_features(obs_dim=5, act_dim=2),
    )

    assert len(ds) == 1
    assert ds.features["obs_state"].length == 5
    assert ds.features["action"].length == 2
    assert "reward" in ds.column_names
    assert "coverage" in ds.column_names

    env.close()
    print("[OK] PushT dataset smoke test passed")
