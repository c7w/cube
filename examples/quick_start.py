#!/usr/bin/env python3
"""
Quick start example for CubeBench new environments.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the new environments
from environment import SequenceEnv, VertexViewEnv, FaceViewEnv


def quick_sequence_example():
    """Quick example using SequenceEnv for token modeling."""
    print("=== SequenceEnv Quick Example ===")
    
    # Create environment
    env = SequenceEnv(scramble_moves=5)
    
    # Reset and get initial observation
    obs, info = env.reset()
    print(f"Initial state: {obs}")
    print(f"Actions available: {env.action_space.n}")
    
    # Take a few random actions
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, State={obs[:20]}...")
        
        if done or truncated:
            break
    
    print("SequenceEnv example completed!\n")


def quick_vertex_example():
    """Quick example using VertexViewEnv for 3D perspective learning."""
    print("=== VertexViewEnv Quick Example ===")
    
    # Create environment  
    env = VertexViewEnv(scramble_moves=3, show_face_labels=False)
    
    # Reset and get initial observation
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial view: {info['view_id']}")
    print(f"Actions available: {env.action_space.n}")
    
    # Take a cube move
    cube_action = 0  # F move
    obs, reward, done, truncated, info = env.step(cube_action)
    print(f"After F move: View={info['view_id']}, Reward={reward:.3f}")
    
    # Take a view transition
    view_action = 12  # view_axis1
    obs, reward, done, truncated, info = env.step(view_action)
    print(f"After view transition: View={info['view_id']}, Reward={reward:.3f}")
    
    env.cleanup()
    print("VertexViewEnv example completed!\n")


def quick_face_example():
    """Quick example using FaceViewEnv for face-centered perspectives."""
    print("=== FaceViewEnv Quick Example ===")
    
    # Create environment
    env = FaceViewEnv(scramble_moves=3, show_face_labels=False)
    
    # Reset and get initial observation
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial view: {info['view_id']}")
    print(f"Actions available: {env.action_space.n}")
    
    # Take a cube move
    cube_action = 8  # U move
    obs, reward, done, truncated, info = env.step(cube_action)
    print(f"After U move: View={info['view_id']}, Reward={reward:.3f}")
    
    # Take a view rotation
    view_action = 13  # view_rot90
    obs, reward, done, truncated, info = env.step(view_action)
    print(f"After view rotation: View={info['view_id']}, Reward={reward:.3f}")
    
    env.cleanup()
    print("FaceViewEnv example completed!\n")


def main():
    """Run all quick examples."""
    print("CubeBench Quick Start Examples")
    print("=" * 50)
    print()
    
    try:
        quick_sequence_example()
        quick_vertex_example() 
        quick_face_example()
        
        print("=" * 50)
        print("All examples completed successfully!")
        print()
        print("Next steps:")
        print("1. Check out examples/demo_new_environments.py for detailed demos")
        print("2. Run examples/test_new_environments.py to verify installation")
        print("3. Use get_next_state_observation() for planning and search")
        print("4. Customize reward functions and action spaces as needed")
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 