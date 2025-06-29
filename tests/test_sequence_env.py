"""
Unit tests for SequenceCubeEnv environment.
"""

import numpy as np
import pytest
from environment.action_space import CUBE_ONLY_ACTION_SPACE
from environment.envs.sequence_env import make_sequence_env
from environment.utils.cube_simulator import CubeSimulator


SOLVED_SEQUENCE = 'R' * 9 + 'O' * 9 + 'B' * 9 + 'G' * 9 + 'Y' * 9 + 'W' * 9
def make_env():
    return make_sequence_env(scramble_moves=0, reward_function_config={'type': 'dummy'})

def test_env_initialization():
    env = make_env()
    assert env is not None
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    assert hasattr(env, 'action_space')
    assert hasattr(env, 'observation_space')


def test_reset():
    env = make_env()
    obs, info = env.reset()
    assert isinstance(obs, str)
    assert len(obs) == 54
    assert isinstance(info, dict)
    # Check that reset returns a solved cube sequence (color chars)
    assert obs == SOLVED_SEQUENCE


def test_step():
    env = make_env()
    env.reset()
    
    # Test a valid action
    obs, reward, terminated, truncated, info = env.step(0)  # R move
    assert isinstance(obs, str)
    assert len(obs) == 54
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_space():
    env = make_env()
    assert env.action_space.n == 12  # 6 faces * 2 directions


def test_observation_space():
    env = make_env()
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)


def test_invalid_action():
    env = make_env()
    env.reset()
    
    with pytest.raises(Exception):
        env.step(12)  # Invalid action index (should be >= 12 for cube-only space)
    with pytest.raises(Exception):
        env.step(-1)  # Invalid action index


def test_state_consistency():
    env = make_env()
    obs, _ = env.reset()
    
    # Perform a move and its inverse
    # R is action 6, R' is action 7 (based on CubeAction enum order)
    obs1, _, _, _, _ = env.step(6)  # R
    obs2, _, _, _, _ = env.step(7)  # R'
    
    # Should be back to original state
    assert obs2 == obs


def test_sequence_validation():
    env = make_env()
    obs, _ = env.reset()
    # Valid sequence should contain only color chars
    assert all(c in 'WYROBG' for c in obs)
    assert len(obs) == 54


def test_reward_function():
    env = make_env()
    obs, _ = env.reset()
    
    # Test that reward is calculated
    obs, reward, _, _, _ = env.step(0)
    assert isinstance(reward, (int, float))


if __name__ == "__main__":
    pytest.main([__file__]) 