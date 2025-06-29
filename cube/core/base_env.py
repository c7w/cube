"""
Base environment class for CubeBench.
Env = State + Transition (CubeSimulator) + ActionSpace + RewardFunction + Renderer (for observation)
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple, List, Protocol
from abc import ABC, abstractmethod

from cube.core.base_simulator import State, CubeSimulator as StateManager
from cube.core.base_action_space import ActionSpace, ActionType
from cube.core.base_reward import RewardFunction


class BaseRenderer(ABC):
    
    @property
    def FACE_TO_COLOR(self):
        return {"FRONT": "R", "BACK": "G", "LEFT": "B", "RIGHT": "Y", "UP": "O", "DOWN": "W"}
    
    @property
    def COLOR_TO_FACE(self):
        return {v: k for k, v in self.FACE_TO_COLOR.items()}
    
    @property
    def STATE_TO_COLOR(self):
        return {k: self.FACE_TO_COLOR[StateManager.STATE_TO_FACE[k]] for k in StateManager.STATE_TO_FACE.keys()}
    
    @property
    def COLOR_TO_STATE(self):
        return {v: k for k, v in self.STATE_TO_COLOR.items()}
    
    @property
    def COLOR_TO_RGB(self):
        return {
            "R": (255, 0, 0),
            "G": (0, 255, 0),
            "B": (0, 0, 255),
            "Y": (255, 255, 0),
            "O": (255, 165, 0),
            "W": (255, 255, 255)
        }
    
    @property
    def STATE_TO_RGB(self):
        return {k: self.COLOR_TO_RGB[self.STATE_TO_COLOR[k]] for k in self.STATE_TO_COLOR.keys()}
    
    def __init__(self):
        pass
    
    """Abstract base class for renderers"""
    @abstractmethod
    def get_observation(self, state: State, viewpoint: Any) -> Any:
        """Get observation from state"""
        pass


class BaseCubeEnvironment(gym.Env, ABC):
    """
    Base environment for CubeBench.
    """
    
    def __init__(self,
                 state_manager: StateManager,
                 action_space_config: ActionSpace,
                 reward_function: RewardFunction,
                 renderer: BaseRenderer,
                 max_steps: int = 1000):
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
        self.state_manager = state_manager
        self.action_space_config = action_space_config
        self.reward_function = reward_function
        self.renderer = renderer
        self.max_steps = max_steps
        
        # Environment state
        self.step_count = 0
        
        # Setup spaces
        self._setup_spaces()
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        self.action_space = self.action_space_config.to_gym_space()
        self._setup_observation_space()
    
    def _setup_observation_space(self):
        """Setup observation space - to be implemented by subclasses"""
        self.viewpoint_space = []
        raise NotImplementedError("Subclasses must implement this method")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset cube state
        self.state_manager.reset()
        
        # Reset environment state
        self.step_count = 0
        self._reset_viewpoint()
        self.reward_function.reset()
        
        # Get observation
        observation = self.get_observation()
        
        # Create info
        info = {
            'step_count': self.step_count,
            'action_history': self.state_manager.get_action_history()
        }
        
        return observation, info
    
    def scramble(self, num_moves: int = 20):
        """Scramble the cube"""
        self.state_manager.scramble(num_moves)
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment"""
        # Save previous state for reward calculation
        previous_state = self.state_manager.get_state().copy()
        
        # Get action details
        action_type, action_name = self.action_space_config.get_action(action)
        
        # Apply action
        if action_type == ActionType.CUBE:
            success = self.state_manager.apply_move(action_name)
            if not success:
                terminated = True
                truncated = False
            else:
                terminated = self.state_manager.is_solved()
                truncated = False
        
        elif action_type == ActionType.VIEW:
            self._update_viewpoint(action_name)
            terminated = False
            truncated = False
        
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
        current_state = self.state_manager.get_state()
        reward = self.reward_function.calculate_reward(
            previous_state, current_state, action_name, terminated, self.step_count, self.max_steps
        )
        
        # Create info dict
        info = {
            'step_count': self.step_count,
            'action': action_name,
            'action_type': action_type.value,
            'action_history': self.state_manager.get_action_history()
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
            'cube_state': self.state_manager.get_state().copy(),
            'viewpoint': self.viewpoint.copy(),
            'step_count': self.step_count,
            'action_history': self.state_manager.get_action_history()
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set environment state"""
        self.state_manager.set_state(state['cube_state'].copy())
        self.viewpoint = state['viewpoint'].copy()
        self.step_count = state['step_count']
    
    def is_solved(self) -> bool:
        """Check if cube is solved"""
        return self.state_manager.is_solved()
    
    def get_action_history(self) -> List[str]:
        """Get action history"""
        return self.state_manager.get_action_history()
    
    def get_observation(self) -> Any:
        """Get current observation"""
        return self.renderer.get_observation(self.state_manager.get_state(), self.viewpoint)
    
    def _reset_viewpoint(self):
        """Reset viewpoint - to be implemented by subclasses"""
        self.viewpoint = None
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def _update_viewpoint(self, action_name: str):
        """Update viewpoint - to be implemented by subclasses"""
        pass


