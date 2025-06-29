"""
Test suite for cube/envs/sequence_env.py

Direct tests without mocking - tests the actual implementation.
"""

import pytest
import numpy as np
from gymnasium import spaces

# Direct imports from the actual modules
from cube.envs.sequence_env import make_sequence_env, SequenceRenderer, SequenceCubeEnvironment


class TestSequenceRenderer:
    """Test the SequenceRenderer class"""
    
    def test_renderer_creation(self):
        """Test that SequenceRenderer can be created"""
        renderer = SequenceRenderer()
        assert renderer is not None
        assert hasattr(renderer, 'get_observation')
    
    def test_get_observation_with_zeros(self):
        """Test get_observation with zero state"""
        renderer = SequenceRenderer()
        state = np.zeros(54, dtype=int)
        viewpoint = None
        
        result = renderer.get_observation(state, viewpoint)
        
        assert isinstance(result, str)
        assert len(result) == 54
    
    def test_get_observation_with_different_values(self):
        """Test get_observation with different state values"""
        renderer = SequenceRenderer()
        state = np.array([0, 1, 2, 3, 4, 5] * 9)  # Mix of different values
        viewpoint = None
        
        result = renderer.get_observation(state, viewpoint)
        
        assert isinstance(result, str)
        assert len(result) == 54


class TestSequenceCubeEnvironment:
    """Test the SequenceCubeEnvironment class"""
    
    def test_environment_creation_with_dependencies(self):
        """Test that environment can be created with proper dependencies"""
        # This will test if all imports work and if the constructor runs
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace()
            reward_function = get_reward_function(type="dummy")
            renderer = SequenceRenderer()
            
            env = SequenceCubeEnvironment(
                state_manager, action_space_config, reward_function, renderer, max_steps=100
            )
            
            assert env is not None
            assert hasattr(env, 'max_steps')
            assert env.max_steps == 100
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Environment creation failed: {e}")
    
    def test_abstract_methods_exist(self):
        """Test that required abstract methods are implemented"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace()
            reward_function = get_reward_function(type="dummy")
            renderer = SequenceRenderer()
            
            env = SequenceCubeEnvironment(
                state_manager, action_space_config, reward_function, renderer
            )
            
            # Test that abstract methods are implemented
            assert hasattr(env, '_setup_observation_space')
            assert hasattr(env, 'get_observation')
            assert hasattr(env, '_reset_viewpoint')
            assert hasattr(env, '_update_viewpoint')
            
            # Test that they can be called without errors
            env._setup_observation_space()
            env._reset_viewpoint()
            env._update_viewpoint('test_action')
            
            observation = env.get_observation()
            assert observation is not None
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Abstract method test failed: {e}")
    
    def test_observation_space_setup(self):
        """Test observation space setup"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace()
            reward_function = get_reward_function(type="dummy")
            renderer = SequenceRenderer()
            
            env = SequenceCubeEnvironment(
                state_manager, action_space_config, reward_function, renderer
            )
            
            # Check that observation space is set up
            assert hasattr(env, 'observation_space')
            assert isinstance(env.observation_space, spaces.Text)
            assert env.observation_space.max_length == 54
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Observation space test failed: {e}")


class TestMakeSequenceEnv:
    """Test the make_sequence_env factory function"""
    
    def test_make_sequence_env_default_params(self):
        """Test make_sequence_env with default parameters"""
        try:
            env = make_sequence_env()
            
            assert env is not None
            assert hasattr(env, 'max_steps')
            assert env.max_steps == 1000  # Default value
            assert hasattr(env, 'state_manager')
            assert hasattr(env, 'action_space_config')
            assert hasattr(env, 'reward_function')
            assert hasattr(env, 'renderer')
            assert isinstance(env.renderer, SequenceRenderer)
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"make_sequence_env default params test failed: {e}")
    
    def test_make_sequence_env_custom_params(self):
        """Test make_sequence_env with custom parameters"""
        try:
            custom_reward_config = {"type": "dummy"}
            env = make_sequence_env(max_steps=500, reward_function_config=custom_reward_config)
            
            assert env is not None
            assert env.max_steps == 500
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"make_sequence_env custom params test failed: {e}")
    
    def test_make_sequence_env_functionality(self):
        """Test that created environment has basic functionality"""
        try:
            env = make_sequence_env()
            
            # Test that it has required Gymnasium interface
            assert hasattr(env, 'reset')
            assert hasattr(env, 'step')
            assert hasattr(env, 'action_space')
            assert hasattr(env, 'observation_space')
            
            # Test reset functionality
            observation, info = env.reset()
            assert observation is not None
            assert isinstance(info, dict)
            
            # Test that observation is a string of length 54
            assert isinstance(observation, str)
            assert len(observation) == 54
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"make_sequence_env functionality test failed: {e}")


class TestIntegration:
    """Integration tests for the sequence environment"""
    
    def test_full_environment_cycle(self):
        """Test complete environment reset and step cycle"""
        try:
            env = make_sequence_env(max_steps=10)
            
            # Test reset
            obs, info = env.reset()
            assert isinstance(obs, str)
            assert len(obs) == 54
            assert isinstance(info, dict)
            assert 'step_count' in info
            
            # Test step
            action = 0  # First action in action space
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert isinstance(obs, str)
            assert len(obs) == 54
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Full environment cycle test failed: {e}")
    
    def test_action_space_compatibility(self):
        """Test that action space works correctly"""
        try:
            env = make_sequence_env()
            
            # Check action space
            assert hasattr(env, 'action_space')
            action_space = env.action_space
            
            # Should be discrete action space
            assert isinstance(action_space, spaces.Discrete)
            
            # Should have some actions (at least 12 cube actions)
            assert action_space.n >= 12
            
            # Test that we can sample actions
            for _ in range(5):
                action = action_space.sample()
                assert 0 <= action < action_space.n
                
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Action space compatibility test failed: {e}")
    
    def test_observation_space_compatibility(self):
        """Test that observation space works correctly"""
        try:
            env = make_sequence_env()
            
            # Check observation space
            assert hasattr(env, 'observation_space')
            obs_space = env.observation_space
            
            # Should be Text space with length 54
            assert isinstance(obs_space, spaces.Text)
            assert obs_space.max_length == 54
            
            # Test that observations conform to space
            obs, _ = env.reset()
            assert obs_space.contains(obs)
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Observation space compatibility test failed: {e}")
    
    def test_multiple_steps(self):
        """Test taking multiple steps in the environment"""
        try:
            env = make_sequence_env(max_steps=5)
            env.reset()
            
            observations = []
            for i in range(3):
                obs, reward, terminated, truncated, info = env.step(i % env.action_space.n)
                observations.append(obs)
                
                # All observations should be strings of length 54
                assert isinstance(obs, str)
                assert len(obs) == 54
                
                # Step count should increase
                assert info['step_count'] == i + 1
                
                if terminated or truncated:
                    break
            
            # Should have collected some observations
            assert len(observations) > 0
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Multiple steps test failed: {e}")
    
    def test_reset_consistency(self):
        """Test that reset works consistently"""
        try:
            env = make_sequence_env()
            
            observations = []
            infos = []
            
            # Reset multiple times
            for _ in range(3):
                obs, info = env.reset()
                observations.append(obs)
                infos.append(info)
                
                assert isinstance(obs, str)
                assert len(obs) == 54
                assert isinstance(info, dict)
                assert info['step_count'] == 0
            
            # Should have collected observations
            assert len(observations) == 3
            assert len(infos) == 3
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Reset consistency test failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 