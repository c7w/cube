"""
Test suite for cube/envs/face_view_env.py

Direct tests without mocking - tests the actual implementation.
"""

import pytest
import numpy as np
from gymnasium import spaces
from PIL import Image

# Direct imports from the actual modules
from cube.envs.face_view_env import make_face_view_env, FaceViewRenderer, FaceViewEnvironment, FaceViewActions


class TestFaceViewRenderer:
    """Test the FaceViewRenderer class"""
    
    def test_renderer_creation_text_mode(self):
        """Test that FaceViewRenderer can be created in text mode"""
        renderer = FaceViewRenderer(observation_type="text")
        assert renderer is not None
        assert renderer.observation_type == "text"
        assert hasattr(renderer, 'get_observation')
        assert hasattr(renderer, 'get_text_observation')
        assert hasattr(renderer, 'get_image_observation')
    
    def test_renderer_creation_image_mode(self):
        """Test that FaceViewRenderer can be created in image mode"""
        renderer = FaceViewRenderer(observation_type="image", image_size=(84, 84))
        assert renderer is not None
        assert renderer.observation_type == "image" 
        assert renderer.image_size == (84, 84)
        assert not renderer.use_label
    
    def test_renderer_creation_image_with_label(self):
        """Test that FaceViewRenderer can be created with labels"""
        renderer = FaceViewRenderer(observation_type="image", image_size=(84, 84), use_label=True)
        assert renderer is not None
        assert renderer.observation_type == "image"
        assert renderer.use_label == True
    
    def test_renderer_label_assertion_error(self):
        """Test that label with text mode raises assertion error"""
        with pytest.raises(AssertionError):
            FaceViewRenderer(observation_type="text", use_label=True)
    
    def test_text_observation_basic(self):
        """Test text observation with basic state"""
        renderer = FaceViewRenderer(observation_type="text")
        state = np.zeros(54, dtype=int)  # All zeros
        viewpoint = (0, 1, 2, 3, 4, 5, 6, 7, 8)  # Front face indices
        
        try:
            result = renderer.get_text_observation(state, viewpoint)
            assert isinstance(result, str)
            assert len(result) == 9  # 3x3 face
        except Exception as e:
            pytest.fail(f"Text observation test failed: {e}")
    
    def test_text_observation_different_values(self):
        """Test text observation with different state values"""
        renderer = FaceViewRenderer(observation_type="text")
        state = np.array([0, 1, 2, 3, 4, 5] * 9)  # Mix of different values
        viewpoint = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        
        try:
            result = renderer.get_text_observation(state, viewpoint)
            assert isinstance(result, str)
            assert len(result) == 9
        except Exception as e:
            pytest.fail(f"Text observation with different values test failed: {e}")
    
    def test_image_observation_basic(self):
        """Test image observation without labels"""
        renderer = FaceViewRenderer(observation_type="image", image_size=(84, 84))
        state = np.zeros(54, dtype=int)
        viewpoint = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        
        try:
            result = renderer.get_image_observation(state, viewpoint)
            assert isinstance(result, np.ndarray)
            assert result.shape == (84, 84, 3)
            assert result.dtype == np.uint8
        except Exception as e:
            pytest.fail(f"Image observation test failed: {e}")
    
    def test_image_observation_with_labels(self):
        """Test image observation with labels (PIL text drawing)"""
        renderer = FaceViewRenderer(observation_type="image", image_size=(84, 84), use_label=True)
        state = np.zeros(54, dtype=int)
        viewpoint = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        
        try:
            result = renderer.get_image_observation(state, viewpoint)
            assert isinstance(result, np.ndarray)
            assert result.shape == (84, 84, 3)
            assert result.dtype == np.uint8
            # dump to image
            Image.fromarray(result).save("test_image_observation_with_labels.png")
            # The result should be different from non-labeled version due to text overlay
        except Exception as e:
            pytest.fail(f"Image observation with labels test failed: {e}")
    
    def test_image_observation_different_sizes(self):
        """Test image observation with different image sizes"""
        sizes = [(42, 42), (126, 126), (168, 168)]
        for size in sizes:
            renderer = FaceViewRenderer(observation_type="image", image_size=size)
            state = np.zeros(54, dtype=int)
            viewpoint = (0, 1, 2, 3, 4, 5, 6, 7, 8)
            
            try:
                result = renderer.get_image_observation(state, viewpoint)
                assert isinstance(result, np.ndarray)
                assert result.shape == (size[0], size[1], 3)
                assert result.dtype == np.uint8
            except Exception as e:
                pytest.fail(f"Image observation size {size} test failed: {e}")
    
    def test_get_observation_text_mode(self):
        """Test get_observation method in text mode"""
        renderer = FaceViewRenderer(observation_type="text")
        state = np.zeros(54, dtype=int)
        viewpoint = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        
        try:
            result = renderer.get_observation(state, viewpoint)
            assert isinstance(result, str)
            assert len(result) == 9
        except Exception as e:
            pytest.fail(f"get_observation text mode test failed: {e}")
    
    def test_get_observation_image_mode(self):
        """Test get_observation method in image mode"""
        renderer = FaceViewRenderer(observation_type="image", image_size=(84, 84))
        state = np.zeros(54, dtype=int)
        viewpoint = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        
        try:
            result = renderer.get_observation(state, viewpoint)
            assert isinstance(result, np.ndarray)
            assert result.shape == (84, 84, 3)
        except Exception as e:
            pytest.fail(f"get_observation image mode test failed: {e}")
    
    def test_get_observation_invalid_mode(self):
        """Test get_observation with invalid observation type"""
        renderer = FaceViewRenderer(observation_type="invalid")
        state = np.zeros(54, dtype=int)
        viewpoint = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        
        with pytest.raises(ValueError, match="Invalid observation_type"):
            renderer.get_observation(state, viewpoint)


class TestFaceViewEnvironment:
    """Test the FaceViewEnvironment class"""
    
    def test_environment_creation_text_mode(self):
        """Test that environment can be created in text mode"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace(view_actions=FaceViewActions)
            reward_function = get_reward_function(type="dummy")
            renderer = FaceViewRenderer(observation_type="text")
            
            env = FaceViewEnvironment(
                state_manager, action_space_config, reward_function, renderer, 
                max_steps=100, observation_type="text"
            )
            
            assert env is not None
            assert hasattr(env, 'viewpoint')
            assert env.viewpoint == (0, 1, 2, 3, 4, 5, 6, 7, 8)  # Default front face
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Environment creation (text) failed: {e}")
    
    def test_environment_creation_image_mode(self):
        """Test that environment can be created in image mode"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace(view_actions=FaceViewActions)
            reward_function = get_reward_function(type="dummy")
            renderer = FaceViewRenderer(observation_type="image", image_size=(84, 84))
            
            env = FaceViewEnvironment(
                state_manager, action_space_config, reward_function, renderer,
                max_steps=100, observation_type="image", image_size=(84, 84)
            )
            
            assert env is not None
            assert hasattr(env, 'viewpoint')
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Environment creation (image) failed: {e}")
    
    def test_abstract_methods_exist(self):
        """Test that required abstract methods are implemented"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace(view_actions=FaceViewActions)
            reward_function = get_reward_function(type="dummy")
            renderer = FaceViewRenderer(observation_type="text")
            
            env = FaceViewEnvironment(
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
            
            # Test viewpoint updates
            env._update_viewpoint('rotate_view_90')
            env._update_viewpoint('view_left')
            
            observation = env.get_observation()
            assert observation is not None
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Abstract method test failed: {e}")
    
    def test_observation_space_setup_text(self):
        """Test observation space setup for text mode"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace(view_actions=FaceViewActions)
            reward_function = get_reward_function(type="dummy")
            renderer = FaceViewRenderer(observation_type="text")
            
            env = FaceViewEnvironment(
                state_manager, action_space_config, reward_function, renderer,
                observation_type="text"
            )
            
            # Check that observation space is set up correctly
            assert hasattr(env, 'observation_space')
            assert isinstance(env.observation_space, spaces.Text)
            assert env.observation_space.max_length == 9
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Observation space (text) test failed: {e}")
    
    def test_observation_space_setup_image(self):
        """Test observation space setup for image mode"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace(view_actions=FaceViewActions)
            reward_function = get_reward_function(type="dummy")
            renderer = FaceViewRenderer(observation_type="image", image_size=(84, 84))
            
            env = FaceViewEnvironment(
                state_manager, action_space_config, reward_function, renderer,
                observation_type="image", image_size=(84, 84)
            )
            
            # Check that observation space is set up correctly
            assert hasattr(env, 'observation_space')
            assert isinstance(env.observation_space, spaces.Box)
            assert env.observation_space.shape == (84, 84, 3)
            assert env.observation_space.dtype == np.uint8
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Observation space (image) test failed: {e}")
    
    def test_viewpoint_rotation_actions(self):
        """Test viewpoint rotation actions"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace(view_actions=FaceViewActions)
            reward_function = get_reward_function(type="dummy")
            renderer = FaceViewRenderer(observation_type="text")
            
            env = FaceViewEnvironment(
                state_manager, action_space_config, reward_function, renderer
            )
            
            # Test different rotation actions
            original_viewpoint = env.viewpoint
            
            env._update_viewpoint('rotate_view_90')
            viewpoint_90 = env.viewpoint
            assert viewpoint_90 != original_viewpoint
            
            env._reset_viewpoint()
            env._update_viewpoint('rotate_view_180')
            viewpoint_180 = env.viewpoint
            assert viewpoint_180 != original_viewpoint
            
            env._reset_viewpoint()
            env._update_viewpoint('rotate_view_270')
            viewpoint_270 = env.viewpoint
            assert viewpoint_270 != original_viewpoint
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Viewpoint rotation test failed: {e}")
    
    def test_viewpoint_direction_actions(self):
        """Test viewpoint direction actions"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace(view_actions=FaceViewActions)
            reward_function = get_reward_function(type="dummy")
            renderer = FaceViewRenderer(observation_type="text")
            
            env = FaceViewEnvironment(
                state_manager, action_space_config, reward_function, renderer
            )
            
            # Test different direction actions
            original_viewpoint = env.viewpoint
            
            for action in ['view_left', 'view_up', 'view_right', 'view_down']:
                env._reset_viewpoint()
                env._update_viewpoint(action)
                new_viewpoint = env.viewpoint
                assert new_viewpoint != original_viewpoint, f"Viewpoint should change for {action}"
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Viewpoint direction test failed: {e}")
    
    def test_view_transformation_sequence_with_images(self):
        """Test view transformation sequence: right -> up -> left and save images"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace(view_actions=FaceViewActions)
            reward_function = get_reward_function(type="dummy")
            # Use image renderer with labels to visualize changes
            renderer = FaceViewRenderer(observation_type="image", image_size=(84, 84), use_label=True)
            
            env = FaceViewEnvironment(
                state_manager, action_space_config, reward_function, renderer,
                observation_type="image", image_size=(84, 84)
            )
            
            # Set up a more interesting state for visualization
            state_manager.scramble(num_moves=5)  # Scramble the cube a bit
            
            # Test sequence: right -> up -> left
            transformation_sequence = ['view_right', 'view_up', 'view_left']
            
            # Save initial viewpoint image
            env._reset_viewpoint()
            initial_obs = env.get_observation()
            Image.fromarray(initial_obs).save(f"test_image_view_00_initial_front.png")
            print(f"保存初始视图 (front): viewpoint={env.viewpoint}")
            
            # Apply transformations and save images
            for i, action in enumerate(transformation_sequence, 1):
                prev_viewpoint = env.viewpoint
                env._update_viewpoint(action)
                new_viewpoint = env.viewpoint
                
                # Get and save image observation
                obs = env.get_observation()
                Image.fromarray(obs).save(f"test_image_view_{i:02d}_{action}.png")
                
                print(f"变换 {i}: {action}")
                print(f"  前一个viewpoint: {prev_viewpoint}")
                print(f"  新的viewpoint: {new_viewpoint}")
                print(f"  保存图像: test_image_view_{i:02d}_{action}.png")
                
                # Verify viewpoint actually changed
                assert new_viewpoint != prev_viewpoint, f"Viewpoint should change after {action}"
                assert isinstance(obs, np.ndarray), f"Observation should be numpy array"
                assert obs.shape == (84, 84, 3), f"Observation shape should be (84, 84, 3)"
            
            # Test that we can still get observations after all transformations
            final_obs = env.get_observation()
            assert final_obs is not None
            assert isinstance(final_obs, np.ndarray)
            
            print("所有视图变换测试完成!")
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"View transformation sequence test failed: {e}")
    
    def test_all_view_directions_with_images(self):
        """Test all view directions and save images for each"""
        try:
            from cube.core.base_simulator import CubeSimulator as StateManager
            from cube.core.base_action_space import ActionSpace
            from cube.core.base_reward import get_reward_function
            
            state_manager = StateManager()
            action_space_config = ActionSpace(view_actions=FaceViewActions)
            reward_function = get_reward_function(type="dummy")
            # Use image renderer with labels
            renderer = FaceViewRenderer(observation_type="image", image_size=(84, 84), use_label=True)
            
            env = FaceViewEnvironment(
                state_manager, action_space_config, reward_function, renderer,
                observation_type="image", image_size=(84, 84)
            )
            
            # Set up a scrambled state for better visualization
            state_manager.scramble(num_moves=10)
            
            # Test all four basic directions from front face
            directions = ['view_left', 'view_up', 'view_right', 'view_down']
            
            # Save front view first
            env._reset_viewpoint()
            front_obs = env.get_observation()
            Image.fromarray(front_obs).save("test_image_all_views_front.png")
            print(f"保存FRONT视图: viewpoint={env.viewpoint}")
            
            # Test each direction
            for direction in directions:
                env._reset_viewpoint()  # Reset to front
                env._update_viewpoint(direction)
                
                obs = env.get_observation()
                filename = f"test_image_all_views_{direction.replace('view_', '')}.png"
                Image.fromarray(obs).save(filename)
                
                print(f"保存{direction.upper()}视图: viewpoint={env.viewpoint}, 文件={filename}")
                
                # Verify observation
                assert isinstance(obs, np.ndarray)
                assert obs.shape == (84, 84, 3)
            
            print("所有方向视图测试完成!")
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"All view directions test failed: {e}")


class TestMakeFaceViewEnv:
    """Test the make_face_view_env factory function"""
    
    def test_make_face_view_env_default_params(self):
        """Test make_face_view_env with default parameters"""
        try:
            env = make_face_view_env()
            
            assert env is not None
            assert hasattr(env, 'max_steps')
            assert env.max_steps == 1000  # Default value
            assert hasattr(env, 'state_manager')
            assert hasattr(env, 'action_space_config')
            assert hasattr(env, 'reward_function')
            assert hasattr(env, 'renderer')
            assert isinstance(env.renderer, FaceViewRenderer)
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"make_face_view_env default params test failed: {e}")
    
    def test_make_face_view_env_custom_params(self):
        """Test make_face_view_env with custom parameters"""
        try:
            custom_reward_config = {"type": "dummy"}
            env = make_face_view_env(max_steps=500, reward_function_config=custom_reward_config)
            
            assert env is not None
            assert env.max_steps == 500
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"make_face_view_env custom params test failed: {e}")
    
    def test_make_face_view_env_functionality(self):
        """Test that created environment has basic functionality"""
        try:
            env = make_face_view_env()
            
            # Test that it has required Gymnasium interface
            assert hasattr(env, 'reset')
            assert hasattr(env, 'step')
            assert hasattr(env, 'action_space')
            assert hasattr(env, 'observation_space')
            
            # Test reset functionality
            observation, info = env.reset()
            assert observation is not None
            assert isinstance(info, dict)
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"make_face_view_env functionality test failed: {e}")


class TestFaceViewActions:
    """Test the FaceViewActions constant"""
    
    def test_face_view_actions_defined(self):
        """Test that FaceViewActions are properly defined"""
        expected_actions = [
            "view_left", "view_up", "view_right", "view_down", 
            "rotate_view_90", "rotate_view_180", "rotate_view_270"
        ]
        
        assert FaceViewActions == expected_actions
        assert len(FaceViewActions) == 7
        
        # Test that all actions are strings
        for action in FaceViewActions:
            assert isinstance(action, str)


class TestIntegration:
    """Integration tests for the complete face view environment"""
    
    def test_full_environment_cycle_text_mode(self):
        """Test full environment cycle in text mode"""
        try:
            env = make_face_view_env()
            
            # Test reset
            observation, info = env.reset()
            assert isinstance(observation, str)
            assert len(observation) == 9
            assert isinstance(info, dict)
            
            # Test step with cube action
            action = 0  # First action in action space
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(obs, str)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Full environment cycle (text) test failed: {e}")
    
    def test_multiple_steps_with_view_actions(self):
        """Test multiple steps including view actions"""
        try:
            env = make_face_view_env()
            env.reset()
            
            # Test multiple steps
            for _ in range(5):
                action = np.random.randint(0, len(env.action_space_config.actions))
                obs, reward, terminated, truncated, info = env.step(action)
                
                assert obs is not None
                assert isinstance(reward, (int, float))
                assert isinstance(terminated, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)
                
                if terminated or truncated:
                    break
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Multiple steps with view actions test failed: {e}")
    
    def test_reset_consistency(self):
        """Test that reset works consistently"""
        try:
            env = make_face_view_env()
            
            # Multiple resets should work
            for _ in range(3):
                obs, info = env.reset()
                assert obs is not None
                assert isinstance(info, dict)
                assert env.viewpoint == (0, 1, 2, 3, 4, 5, 6, 7, 8)  # Should reset to front face
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Reset consistency test failed: {e}")
    
    def test_action_space_compatibility(self):
        """Test action space compatibility with view actions"""
        try:
            env = make_face_view_env()
            
            # Check that action space includes view actions
            assert hasattr(env.action_space_config, 'view_actions')
            assert env.action_space_config.view_actions == FaceViewActions
            
            # Check that action space is properly sized
            total_actions = len(env.action_space_config.cube_actions) + len(FaceViewActions)
            assert env.action_space.n >= len(FaceViewActions)
            
        except ImportError as e:
            pytest.skip(f"Skipping test due to import error: {e}")
        except Exception as e:
            pytest.fail(f"Action space compatibility test failed: {e}") 