"""
Base environment class for CubeBench.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List, Protocol
from abc import ABC, abstractmethod

from environment.utils.cube_simulator import CubeSimulator
from environment.action_space import ActionSpace, ActionType
from environment.rewards.reward import RewardFunction, get_reward_function


class BaseCubeEnv(gym.Env, ABC):
    """
    Base environment for CubeBench.
    """
    
    def __init__(self,
                 cube_manager: CubeSimulator,
                 action_space_config: ActionSpace,
                 reward_function: RewardFunction = None,
                 max_steps: int = 1000,
                 scramble_moves: int = 20):
        """
        Initialize the environment.
        
        Args:
            cube_manager: Cube state manager
            action_space_config: Action space configuration
            reward_function: Reward function (defaults to DummyReward if None)
            max_steps: Maximum steps per episode
            scramble_moves: Number of moves to scramble the cube
        """
        super().__init__()
        
        # Core components
        self.cube_manager = cube_manager
        self.action_space_config = action_space_config
        if reward_function is None:
            self.reward_function = get_reward_function()
        else:
            self.reward_function = reward_function
        self.max_steps = max_steps
        self.scramble_moves = scramble_moves
        
        # Environment state
        self.step_count = 0
        self.camera_params = {
            'position': (0.0, 0.0, 5.0),
            'orientation': (0.0, 0.0, 0.0)
        }
        
        # Setup spaces
        self._setup_spaces()
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        self.action_space = self.action_space_config.to_gym_space()
        self._setup_observation_space()
    
    @abstractmethod
    def _setup_observation_space(self):
        """Setup observation space - to be implemented by subclasses"""
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset cube state
        self.cube_manager.reset()
        
        # Scramble if needed
        if self.scramble_moves > 0:
            self.cube_manager.scramble(self.scramble_moves)
        
        # Reset environment state
        self.step_count = 0
        self._reset_camera_params()
        self.reward_function.reset()
        
        # Get observation
        observation = self.get_observation()
        
        # Create info
        info = {
            'step_count': self.step_count,
            'action_history': self.cube_manager.get_action_history()
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        # Save previous state for reward calculation
        previous_state = self.cube_manager.get_state().copy()
        
        # Get action details
        action_type, action_name = self.action_space_config.get_action(action)
        
        # Apply action
        if action_type == ActionType.CUBE:
            success = self.cube_manager.apply_move(action_name)
            if not success:
                terminated = True
                truncated = False
            else:
                terminated = self.cube_manager.is_solved()
                truncated = False
        
        elif action_type == ActionType.VIEW:
            self._update_camera_params(action_name)
            terminated = False
            truncated = False
        
        elif action_type == ActionType.SPECIAL:
            terminated, truncated = self._handle_special_action(action_name)
        
        else:
            terminated = True
            truncated = False
        
        # Update step count
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Get observation
        observation = self.get_observation()
        
        # Calculate reward using reward function with previous state
        current_state = self.cube_manager.get_state()
        reward = self.reward_function.calculate_reward(
            previous_state, current_state, action_name, terminated, self.step_count, self.max_steps
        )
        
        # Create info dict
        info = {
            'step_count': self.step_count,
            'action': action_name,
            'action_type': action_type.value,
            'action_history': self.cube_manager.get_action_history()
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the current state"""
        return self.get_observation()
    
    def close(self):
        """Close the environment"""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete environment state"""
        return {
            'cube_state': self.cube_manager.get_state().copy(),
            'camera_params': self.camera_params.copy(),
            'step_count': self.step_count,
            'action_history': self.cube_manager.get_action_history()
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set environment state"""
        self.cube_manager.set_state(state['cube_state'].copy())
        self.camera_params = state['camera_params'].copy()
        self.step_count = state['step_count']
    
    def is_solved(self) -> bool:
        """Check if cube is solved"""
        return self.cube_manager.is_solved()
    
    def get_action_history(self) -> List[str]:
        """Get action history"""
        return self.cube_manager.get_action_history()
    
    @abstractmethod
    def get_observation(self) -> Any:
        """Get current observation - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _reset_camera_params(self):
        """Reset camera parameters - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _update_camera_params(self, action_name: str):
        """Update camera parameters - to be implemented by subclasses"""
        pass
    
    def _handle_special_action(self, action_name: str) -> Tuple[bool, bool]:
        """Handle special actions"""
        if action_name == "scramble":
            self.cube_manager.scramble(self.scramble_moves)
            return False, False
        elif action_name == "reset":
            self.cube_manager.reset()
            return False, False
        else:
            return False, False