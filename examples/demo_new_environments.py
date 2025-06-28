#!/usr/bin/env python3
"""
Demonstration script for the new CubeBench environments.
Shows how to use SequenceEnv, VertexViewEnv, and FaceViewEnv.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.envs.sequence_env import SequenceEnv
from environment.envs.vertex_view_env import VertexViewEnv
from environment.envs.face_view_env import FaceViewEnv


def demo_sequence_env():
    """Demonstrate sequence environment for token modeling."""
    print("=" * 60)
    print("SEQUENCE ENVIRONMENT DEMO")
    print("=" * 60)
    print("The SequenceEnv is designed for sequence modeling tasks.")
    print("State and observation are both 54-character token sequences.")
    print("Actions: 12 cube rotations (F, F', B, B', L, L', R, R', U, U', D, D')")
    print()
    
    # Create environment
    env = SequenceEnv(scramble_moves=10)
    
    # Reset and get initial state
    obs, info = env.reset()
    print(f"Initial state: {obs}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Available actions: {env.get_action_names()}")
    print()
    
    # Demonstrate state transitions
    print("Demonstrating state transitions:")
    for i in range(3):
        action = i * 4  # F, B, L moves
        action_name = env.get_action_names()[action]
        
        # Predict next state
        next_state, next_obs = env.get_next_state_observation(obs, action)
        
        print(f"Action {action} ({action_name}):")
        print(f"  Current:  {obs[:20]}...")
        print(f"  Next:     {next_state[:20]}...")
        print(f"  Changed:  {obs != next_state}")
        
        # Actually perform the action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Reward: {reward:.3f}")
        print()
        
        if terminated or truncated:
            break
    
    print("SequenceEnv demo completed!\n")


def demo_vertex_view_env():
    """Demonstrate vertex view environment for 3D perspective learning."""
    print("=" * 60)
    print("VERTEX VIEW ENVIRONMENT DEMO")
    print("=" * 60)
    print("The VertexViewEnv provides 3D cube views from vertex perspectives.")
    print("State: (cube_state_string, view_id)")
    print("Observation: 256x256 RGB image")
    print("Actions: 12 cube rotations + 3 view transitions")
    print()
    
    # Create environment
    env = VertexViewEnv(scramble_moves=5, show_face_labels=True)
    
    # Reset and get initial state
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial view ID: {info['view_id']}")
    print(f"Available view transitions: {info['available_view_transitions']}")
    print(f"Action space size: {env.action_space.n}")
    print()
    
    # Demonstrate view transitions
    print("Demonstrating view transitions:")
    for i in range(3):
        view_action = 12 + i  # view_axis1, view_axis2, view_axis3
        action_name = env.get_action_names()[view_action]
        
        old_view = info['view_id']
        obs, reward, terminated, truncated, info = env.step(view_action)
        new_view = info['view_id']
        
        print(f"Action {view_action} ({action_name}):")
        print(f"  View transition: {old_view} -> {new_view}")
        print(f"  Reward: {reward:.3f}")
        print(f"  New available transitions: {info['available_view_transitions']}")
        print()
    
    # Demonstrate cube move
    print("Demonstrating cube move:")
    cube_action = 0  # F move
    action_name = env.get_action_names()[cube_action]
    
    old_state = info['state_string']
    obs, reward, terminated, truncated, info = env.step(cube_action)
    new_state = info['state_string']
    
    print(f"Action {cube_action} ({action_name}):")
    print(f"  State changed: {old_state[:20]}... -> {new_state[:20]}...")
    print(f"  View unchanged: {info['view_id']}")
    print(f"  Reward: {reward:.3f}")
    print()
    
    # Save sample image
    try:
        import cv2
        cv2.imwrite('vertex_demo_sample.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        print("Sample image saved: vertex_demo_sample.png")
    except ImportError:
        print("OpenCV not available, skipping image save")
    
    env.cleanup()
    print("VertexViewEnv demo completed!\n")


def demo_face_view_env():
    """Demonstrate face view environment for 3D perspective learning."""
    print("=" * 60)
    print("FACE VIEW ENVIRONMENT DEMO")
    print("=" * 60)
    print("The FaceViewEnv provides 3D cube views from face center perspectives.")
    print("State: (cube_state_string, view_id)")
    print("Observation: 256x256 RGB image")
    print("Actions: 12 cube rotations + 4 view transitions")
    print()
    
    # Create environment
    env = FaceViewEnv(scramble_moves=5, show_face_labels=True)
    
    # Reset and get initial state
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial view ID: {info['view_id']}")
    print(f"Available view transitions: {info['available_view_transitions']}")
    print(f"Action space size: {env.action_space.n}")
    print()
    
    # Demonstrate view transitions
    print("Demonstrating view transitions:")
    for i in range(4):
        view_action = 12 + i  # view_rot0, view_rot90, view_rot180, view_rot270
        action_name = env.get_action_names()[view_action]
        
        old_view = info['view_id']
        obs, reward, terminated, truncated, info = env.step(view_action)
        new_view = info['view_id']
        
        print(f"Action {view_action} ({action_name}):")
        print(f"  View transition: {old_view} -> {new_view}")
        print(f"  Reward: {reward:.3f}")
        print()
        
        if i == 1:  # Stop after a couple transitions
            break
    
    # Demonstrate cube move
    print("Demonstrating cube move:")
    cube_action = 8  # U move
    action_name = env.get_action_names()[cube_action]
    
    old_state = info['state_string']
    obs, reward, terminated, truncated, info = env.step(cube_action)
    new_state = info['state_string']
    
    print(f"Action {cube_action} ({action_name}):")
    print(f"  State changed: {old_state[:20]}... -> {new_state[:20]}...")
    print(f"  View unchanged: {info['view_id']}")
    print(f"  Reward: {reward:.3f}")
    print()
    
    # Save sample image
    try:
        import cv2
        cv2.imwrite('face_demo_sample.png', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        print("Sample image saved: face_demo_sample.png")
    except ImportError:
        print("OpenCV not available, skipping image save")
    
    env.cleanup()
    print("FaceViewEnv demo completed!\n")


def demonstrate_get_next_state_observation():
    """Demonstrate the get_next_state_observation function across environments."""
    print("=" * 60)
    print("GET_NEXT_STATE_OBSERVATION DEMO")
    print("=" * 60)
    print("This function allows predicting next state and observation without")
    print("actually stepping the environment. Useful for planning and search.")
    print()
    
    # SequenceEnv
    print("SequenceEnv:")
    seq_env = SequenceEnv(scramble_moves=3)
    obs, _ = seq_env.reset()
    next_state, next_obs = seq_env.get_next_state_observation(obs, 0)  # F move
    print(f"  Current:  {obs[:30]}...")
    print(f"  F move -> {next_state[:30]}...")
    print()
    
    # VertexViewEnv
    print("VertexViewEnv:")
    vertex_env = VertexViewEnv(scramble_moves=3)
    obs, info = vertex_env.reset()
    state_str = info['state_string']
    view_id = info['view_id']
    next_state, next_obs = vertex_env.get_next_state_observation(state_str, view_id, 0)
    print(f"  Current state:  {state_str[:30]}...")
    print(f"  Current view:   {view_id}")
    print(f"  F move -> state: {next_state[0][:30]}...")
    print(f"  F move -> view:  {next_state[1]}")
    print(f"  F move -> obs shape: {next_obs.shape}")
    vertex_env.cleanup()
    print()
    
    # FaceViewEnv
    print("FaceViewEnv:")
    face_env = FaceViewEnv(scramble_moves=3)
    obs, info = face_env.reset()
    state_str = info['state_string']
    view_id = info['view_id']
    next_state, next_obs = face_env.get_next_state_observation(state_str, view_id, 12)  # view transition
    print(f"  Current state:  {state_str[:30]}...")
    print(f"  Current view:   {view_id}")
    print(f"  View transition -> state: {next_state[0][:30]}...")
    print(f"  View transition -> view:  {next_state[1]}")
    print(f"  View transition -> obs shape: {next_obs.shape}")
    face_env.cleanup()
    print()


def main():
    """Run all demonstrations."""
    print("CubeBench New Environments Demonstration")
    print("=" * 80)
    print()
    
    try:
        # Run individual demos
        demo_sequence_env()
        demo_vertex_view_env()
        demo_face_view_env()
        demonstrate_get_next_state_observation()
        
        print("=" * 80)
        print("All demonstrations completed successfully!")
        print()
        print("Key Features:")
        print("- SequenceEnv: Token sequences for language model training")
        print("- VertexViewEnv: 3D vertex perspectives with view transitions")
        print("- FaceViewEnv: 3D face perspectives with rotational views")
        print("- All environments support get_next_state_observation for planning")
        print("- Headless rendering support for server deployment")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 