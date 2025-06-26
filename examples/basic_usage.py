#!/usr/bin/env python3
"""
Basic usage example for CubeBench environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import make_cubebench_env, ActionSpace
from environment.reward import RewardType, RewardFactory

def main():
    """Demonstrate basic environment usage."""
    print("CubeBench Basic Usage Example")
    print("=" * 40)
    
    # Create environment with different configurations
    print("\n1. Creating environment with symbolic observations...")
    env_symbolic = make_cubebench_env(
        observation_mode="symbolic",
        max_steps=50,
        scramble_moves=5
    )
    
    print(f"Action space: {env_symbolic.action_space}")
    print(f"Observation space: {env_symbolic.observation_space}")
    
    # Reset and get initial observation
    obs, info = env_symbolic.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Take some random actions
    print("\n2. Taking random actions...")
    for step in range(10):
        action = env_symbolic.action_space.sample()
        obs, reward, terminated, truncated, info = env_symbolic.step(action)
        
        print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, "
              f"Solved faces={info['solved_faces']}, Terminated={terminated}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env_symbolic.close()
    
    # Test with image observations
    print("\n3. Testing with image observations...")
    env_image = make_cubebench_env(
        observation_mode="image",
        max_steps=20,
        scramble_moves=3
    )
    
    obs, info = env_image.reset()
    print(f"Image observation shape: {obs.shape}")
    
    # Take a few actions
    for step in range(5):
        action = env_image.action_space.sample()
        obs, reward, terminated, truncated, info = env_image.step(action)
        print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}")
    
    env_image.close()
    
    # Test different reward functions
    print("\n4. Testing different reward functions...")
    
    # Sparse reward
    sparse_reward = RewardFactory.create_reward(RewardType.SPARSE)
    env_sparse = make_cubebench_env(
        reward_function=sparse_reward,
        observation_mode="symbolic",
        max_steps=10
    )
    
    obs, info = env_sparse.reset()
    for step in range(5):
        action = env_sparse.action_space.sample()
        obs, reward, terminated, truncated, info = env_sparse.step(action)
        print(f"Sparse reward step {step + 1}: {reward:.2f}")
    
    env_sparse.close()
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 