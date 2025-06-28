"""
Sequence-based CubeBench environment for token modeling.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List

from ..base_env import CubeBenchEnv
from ..action_space import ActionSpace, ActionType
from ..reward import RewardFunction
from ..renderers.sequence_renderer import SequenceRenderer
from ..utils.state_utils import serialize_state, deserialize_state


class SequenceEnv(CubeBenchEnv):
    """
    Sequence-based environment for token modeling.
    
    State and observation are both 54-character token sequences.
    Only supports cube rotation actions (12 actions total).
    """
    
    def __init__(self,
                 reward_function: Optional[RewardFunction] = None,
                 max_steps: int = 1000,
                 scramble_moves: int = 20):
        """
        Initialize sequence environment.
        
        Args:
            reward_function: Reward function to use
            max_steps: Maximum steps per episode
            scramble_moves: Number of moves to scramble the cube
        """
        # Initialize sequence-specific attributes first
        self.sequence_renderer = SequenceRenderer()
        self.current_state_str = ""
        
        # Create action space with only cube moves (no view actions)
        action_space_config = ActionSpace(
            include_view_actions=False,
            include_special_actions=False
        )
        
        # Initialize base environment with sequence-specific settings
        super().__init__(
            action_space_config=action_space_config,
            reward_function=reward_function,
            render_mode="symbolic",  # We don't need image rendering
            max_steps=max_steps,
            scramble_moves=scramble_moves,
            observation_mode="symbolic"  # Use symbolic mode as base
        )
        
        # Override observation space for string sequences
        self.observation_space = spaces.Text(54, charset="WYROBG")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        # Call parent reset to get initial state
        _, info = super().reset(seed=seed, options=options)
        
        # Convert state to string representation
        self.current_state_str = serialize_state(self.current_state)
        
        # Update info
        info['state_string'] = self.current_state_str
        info['state_valid'] = self.sequence_renderer.validate_observation(self.current_state_str)
        
        return self.current_state_str, info
    
    def step(self, action: int) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (0-11 for cube moves)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Call parent step to handle cube logic
        _, reward, terminated, truncated, info = super().step(action)
        
        # Convert new state to string representation
        self.current_state_str = serialize_state(self.current_state)
        
        # Update info
        info['state_string'] = self.current_state_str
        info['state_valid'] = self.sequence_renderer.validate_observation(self.current_state_str)
        
        return self.current_state_str, reward, terminated, truncated, info
    
    def get_next_state_observation(self, state_str: str, action: int) -> Tuple[str, str]:
        """
        Get next state and observation given current state and action.
        
        Args:
            state_str: Current state as string
            action: Action index
            
        Returns:
            Tuple of (next_state_str, observation_str)
        """
        # Convert string state to array
        current_state_array = deserialize_state(state_str)
        
        # Temporarily set the cube state
        original_state = self.current_state.copy()
        original_state_str = self.current_state_str
        
        self.current_state = current_state_array
        self.cube_simulator.set_state(current_state_array)
        
        # Apply action
        action_type, action_name = self.action_space_config.get_action(action)
        
        if action_type == ActionType.CUBE:
            success = self.cube_simulator.apply_move(action_name)
            if success:
                next_state_array = self.cube_simulator.get_state()
                next_state_str = serialize_state(next_state_array)
                observation_str = next_state_str  # In sequence env, state == observation
            else:
                # Invalid move - return current state
                next_state_str = state_str
                observation_str = state_str
        else:
            # Invalid action type for sequence env
            next_state_str = state_str
            observation_str = state_str
        
        # Restore original state
        self.current_state = original_state
        self.current_state_str = original_state_str
        self.cube_simulator.set_state(original_state)
        
        return next_state_str, observation_str
    
    def _get_observation(self) -> str:
        """Get current observation (string representation)"""
        return self.current_state_str
    
    def render(self):
        """Render the current state as string"""
        if self.render_mode == "human":
            print(f"Cube state: {self.current_state_str}")
            return self.current_state_str
        else:
            return self.current_state_str
    
    def get_state_string(self) -> str:
        """Get current state as string"""
        return self.current_state_str
    
    def set_state_string(self, state_str: str):
        """Set state from string representation"""
        if not self.sequence_renderer.validate_observation(state_str):
            raise ValueError(f"Invalid state string: {state_str}")
        
        state_array = deserialize_state(state_str)
        self.set_state(state_array)
        self.current_state_str = state_str
    
    def get_action_names(self) -> List[str]:
        """Get list of action names"""
        actions = []
        for i in range(self.action_space.n):
            _, action_name = self.action_space_config.get_action(i)
            actions.append(action_name)
        return actions 