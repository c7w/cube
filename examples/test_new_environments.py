#!/usr/bin/env python3
"""
Test script for the new CubeBench environments.
Tests SequenceEnv, VertexViewEnv, and FaceViewEnv.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.envs.sequence_env import SequenceEnv
from environment.envs.vertex_view_env import VertexViewEnv
from environment.envs.face_view_env import FaceViewEnv
from environment.utils.state_utils import serialize_state, deserialize_state
from environment.utils.view_utils import get_vertex_views, get_face_views


def test_sequence_env():
    """Test the sequence environment."""
    print("=" * 50)
    print("Testing SequenceEnv")
    print("=" * 50)
    
    # Create environment
    env = SequenceEnv(scramble_moves=5)
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test action space
    print(f"Action space: {env.action_space}")
    print(f"Action names: {env.get_action_names()}")
    
    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Obs={obs[:10]}...")
        
        if terminated or truncated:
            break
    
    # Test get_next_state_observation
    state_str = env.get_state_string()
    next_state, next_obs = env.get_next_state_observation(state_str, 0)  # F move
    print(f"Next state prediction: {next_state[:10]}...")
    
    print("SequenceEnv test completed successfully!\n")


def test_vertex_view_env():
    """Test the vertex view environment."""
    print("=" * 50)
    print("Testing VertexViewEnv")
    print("=" * 50)
    
    # Create environment
    env = VertexViewEnv(scramble_moves=3, show_face_labels=True)
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test action space
    print(f"Action space: {env.action_space}")
    print(f"Action names: {env.get_action_names()}")
    
    # Test vertex views
    vertex_views = get_vertex_views()
    print(f"Total vertex views: {len(vertex_views)}")
    print(f"First vertex view: {vertex_views[0]}")
    
    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, View={info.get('view_id', 'N/A')}")
        
        if terminated or truncated:
            break
    
    # Test get_next_state_observation
    state_info = env.get_state_info()
    next_state, next_obs = env.get_next_state_observation(
        state_info['state_string'], 
        state_info['view_id'], 
        0  # F move
    )
    print(f"Next state prediction: {next_state[0][:10]}..., view={next_state[1]}")
    print(f"Next observation shape: {next_obs.shape}")
    
    # Clean up
    env.cleanup()
    print("VertexViewEnv test completed successfully!\n")


def test_face_view_env():
    """Test the face view environment."""
    print("=" * 50)
    print("Testing FaceViewEnv")
    print("=" * 50)
    
    # Create environment
    env = FaceViewEnv(scramble_moves=3, show_face_labels=True)
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test action space
    print(f"Action space: {env.action_space}")
    print(f"Action names: {env.get_action_names()}")
    
    # Test face views
    face_views = get_face_views()
    print(f"Total face views: {len(face_views)}")
    print(f"First face view: {face_views[0]}")
    
    # Test a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, View={info.get('view_id', 'N/A')}")
        
        if terminated or truncated:
            break
    
    # Test get_next_state_observation
    state_info = env.get_state_info()
    next_state, next_obs = env.get_next_state_observation(
        state_info['state_string'], 
        state_info['view_id'], 
        0  # F move
    )
    print(f"Next state prediction: {next_state[0][:10]}..., view={next_state[1]}")
    print(f"Next observation shape: {next_obs.shape}")
    
    # Clean up
    env.cleanup()
    print("FaceViewEnv test completed successfully!\n")


def test_state_utils():
    """Test state serialization utilities."""
    print("=" * 50)
    print("Testing State Utilities")
    print("=" * 50)
    
    # Create a test state
    from environment.cube_simulator import CubeSimulator
    cube = CubeSimulator()
    cube.scramble(5)
    state = cube.get_state()
    
    # Test serialization
    state_str = serialize_state(state)
    print(f"Original state shape: {state.shape}")
    print(f"Serialized state: {state_str}")
    
    # Test deserialization
    recovered_state = deserialize_state(state_str)
    print(f"Recovered state shape: {recovered_state.shape}")
    print(f"States match: {np.array_equal(state, recovered_state)}")
    
    print("State utilities test completed successfully!\n")


def save_sample_images():
    """Save sample images from different views."""
    print("=" * 50)
    print("Saving Sample Images")
    print("=" * 50)
    
    try:
        import cv2
        
        # Create environments
        vertex_env = VertexViewEnv(scramble_moves=10, show_face_labels=True)
        face_env = FaceViewEnv(scramble_moves=10, show_face_labels=True)
        
        # Get observations
        vertex_obs, _ = vertex_env.reset()
        face_obs, _ = face_env.reset()
        
        # Save images
        cv2.imwrite('vertex_view_sample.png', cv2.cvtColor(vertex_obs, cv2.COLOR_RGB2BGR))
        cv2.imwrite('face_view_sample.png', cv2.cvtColor(face_obs, cv2.COLOR_RGB2BGR))
        
        print("Sample images saved: vertex_view_sample.png, face_view_sample.png")
        
        # Clean up
        vertex_env.cleanup()
        face_env.cleanup()
        
    except ImportError:
        print("OpenCV not available, skipping image saving")
    except Exception as e:
        print(f"Error saving images: {e}")


if __name__ == "__main__":
    print("Testing CubeBench New Environments")
    print("=" * 60)
    
    try:
        # Test individual components
        test_state_utils()
        test_sequence_env()
        test_vertex_view_env()
        test_face_view_env()
        
        # Save sample images
        save_sample_images()
        
        print("=" * 60)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 