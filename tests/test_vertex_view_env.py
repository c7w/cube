import pytest
import numpy as np
import os
from PIL import Image

from cube.envs.vertex_view_env import make_vertex_view_env, VertexViewEnvironment
from cube.core.base_action_space import ActionType

# Define a temporary directory to save test images
TEST_OUTPUT_DIR = "test_outputs"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


@pytest.fixture
def env():
    """Pytest fixture to create a default VertexViewEnvironment."""
    return make_vertex_view_env()

def test_environment_creation(env):
    """Test that the environment is created successfully."""
    assert isinstance(env, VertexViewEnvironment)
    assert env.observation_space.shape == (84, 84, 3)
    # 12 cube moves (F, F', B, B', etc.) + 2 view actions
    assert env.action_space.n == 12 + 2

def test_reset(env):
    """Test the reset method and the returned observation."""
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (84, 84, 3)
    assert obs.dtype == np.uint8
    assert 'step_count' in info
    assert info['step_count'] == 0

def test_step_cube_action(env):
    """Test stepping with a cube action."""
    env.reset()
    action_idx = env.action_space_config.get_action_idx(ActionType.CUBE, 'F')
    obs, reward, terminated, truncated, info = env.step(action_idx)
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (84, 84, 3)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info['action'] == 'F'

def test_step_view_action(env):
    """Test stepping with a view action (even if it's a no-op)."""
    env.reset()
    action_idx = env.action_space_config.get_action_idx(ActionType.VIEW, 'rotate_vertex_120')
    # This should not crash, even though the function is currently a `pass`
    obs, reward, terminated, truncated, info = env.step(action_idx)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (84, 84, 3)
    assert info['action'] == 'rotate_vertex_120'
    # Since it's a view action, the episode should not terminate
    assert not terminated

def define_eight_canonical_views():
    """Defines 8 viewpoints to see each vertex of the cube."""
    base_zoom = -15
    # Combinations of rotations to show each of the 8 corners
    views = []
    for rot_x in [-35, 35]: # Looking slightly from top or bottom
        for rot_y in [45, 135, 225, 315]: # Cardinal and diagonal directions
            views.append({"rot_x": rot_x, "rot_y": rot_y, "zoom": base_zoom})
    return views

def test_all_vertex_viewpoints_rendering(env):
    """
    Renders an image from each of the 8 canonical viewpoints and saves them.
    This allows for visual verification of the renderer.
    """
    views = define_eight_canonical_views()
    state, _ = env.reset() # Get initial solved state

    for i, viewpoint in enumerate(views):
        # Manually set the environment's viewpoint for rendering
        env.viewpoint = viewpoint
        
        # Get the observation for the current viewpoint
        obs = env.get_observation()
        
        # Basic validation of the observation
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (84, 84, 3)

        # Save the image for visual inspection
        try:
            image = Image.fromarray(obs)
            output_filename = os.path.join(TEST_OUTPUT_DIR, f"vertex_view_test_v{i+1}.png")
            image.save(output_filename)
            print(f"Saved viewpoint {i+1} to {output_filename}")
        except Exception as e:
            pytest.fail(f"Failed to save image for viewpoint {i+1}: {e}")

    # Verify that 8 images were created
    assert len(os.listdir(TEST_OUTPUT_DIR)) >= 8 