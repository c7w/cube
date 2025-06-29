import numpy as np
import pytest
from trash.base_env import CubeBenchEnv

def test_env_reset_and_step():
    env = CubeBenchEnv(max_steps=5, scramble_moves=2, observation_mode="symbolic")
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6, 9)
    assert info['step_count'] == 0
    done = False
    steps = 0
    while not done and steps < 5:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        steps += 1
        if terminated or truncated:
            done = True
    env.close() 