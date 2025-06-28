"""
Face view CubeBench environment for 3D perspective learning.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List

from ..base_env import CubeBenchEnv
from ..action_space import ActionSpace, ActionType
from ..reward import RewardFunction
from ..renderers.face_renderer import FaceViewRenderer
from ..utils.state_utils import serialize_state, deserialize_state
from ..utils.view_utils import ViewType, get_view_neighbors


class FaceViewEnv(CubeBenchEnv):
    """
    Face view environment for 3D perspective learning.
    
    State: (cube_state_string, view_id)
    Observation: 256x256 RGB image from current face view
    Actions: 12 cube moves + 4 view transitions
    """
    
    def __init__(self,
                 reward_function: Optional[RewardFunction] = None,
                 max_steps: int = 1000,
                 scramble_moves: int = 20,
                 show_face_labels: bool = False):
        """
        Initialize face view environment.
        
        Args:
            reward_function: Reward function to use
            max_steps: Maximum steps per episode
            scramble_moves: Number of moves to scramble the cube
            show_face_labels: Whether to show face labels in rendered images
        """
        # Initialize face-specific attributes first
        self.face_renderer = FaceViewRenderer(
            image_size=256,
            show_face_labels=show_face_labels
        )
        self.current_view_id = 0  # Start at view 0
        self.view_neighbors = get_view_neighbors(ViewType.FACE)
        self.current_state_str = ""
        
        # Create action space with cube moves + view transitions
        action_space_config = ActionSpace(
            include_view_actions=True,
            include_special_actions=False,
            custom_view_actions=['view_rot0', 'view_rot90', 'view_rot180', 'view_rot270']
        )
        
        # Initialize base environment
        super().__init__(
            action_space_config=action_space_config,
            reward_function=reward_function,
            render_mode="rgb_array",
            max_steps=max_steps,
            scramble_moves=scramble_moves,
            observation_mode="image"
        )
        
        # Override observation space for 256x256 RGB images
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        
        # Reset view to 0
        self.current_view_id = 0
        
        # Convert state to string representation
        self.current_state_str = serialize_state(self.current_state)
        
        # Get observation from current view
        observation = self.face_renderer.get_observation(self.current_state, self.current_view_id)
        
        # Update info
        info['state_string'] = self.current_state_str
        info['view_id'] = self.current_view_id
        info['view_type'] = 'face'
        info['available_view_transitions'] = self.view_neighbors[self.current_view_id]
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index (0-11 for cube moves, 12-15 for view transitions)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get action details
        action_type, action_name = self.action_space_config.get_action(action)
        
        # Save current state for reward calculation
        current_state = self.current_state.copy()
        
        if action_type == ActionType.CUBE:
            # Apply cube move using parent logic
            _, reward, terminated, truncated, info = super().step(action)
            
            # Update string representation
            self.current_state_str = serialize_state(self.current_state)
            
        elif action_type == ActionType.VIEW:
            # Apply view transition
            old_view_id = self.current_view_id
            success = self._apply_view_transition(action_name)
            
            if success:
                reward = -0.01  # Small penalty for view changes
                terminated = False
                truncated = False
            else:
                reward = -0.1  # Penalty for invalid view transition
                terminated = False
                truncated = False
            
            # Increment step count
            self.step_count += 1
            
            # Check for truncation
            if self.step_count >= self.max_steps:
                truncated = True
            
            # Create info dict
            info = {
                'step_count': self.step_count,
                'action': action_name,
                'action_type': action_type.value,
                'solved_faces': self.cube_simulator.get_solved_faces_count(),
                'is_solved': self.cube_simulator.is_solved(),
                'action_history': self.cube_simulator.get_action_history(),
                'old_view_id': old_view_id,
                'new_view_id': self.current_view_id,
                'view_transition_success': success
            }
        
        else:
            # Invalid action type
            reward = -1.0
            terminated = False
            truncated = False
            info = {'error': 'Invalid action type'}
        
        # Get observation from current view
        observation = self.face_renderer.get_observation(self.current_state, self.current_view_id)
        
        # Update info
        info['state_string'] = self.current_state_str
        info['view_id'] = self.current_view_id
        info['view_type'] = 'face'
        info['available_view_transitions'] = self.view_neighbors[self.current_view_id]
        
        return observation, reward, terminated, truncated, info
    
    def get_next_state_observation(self, state_str: str, view_id: int, action: int) -> Tuple[Tuple[str, int], np.ndarray]:
        """
        Get next state and observation given current state and action.
        
        Args:
            state_str: Current cube state as string
            view_id: Current view ID
            action: Action index
            
        Returns:
            Tuple of ((next_state_str, next_view_id), observation_array)
        """
        # Convert string state to array
        current_state_array = deserialize_state(state_str)
        
        # Temporarily set the environment state
        original_state = self.current_state.copy()
        original_state_str = self.current_state_str
        original_view_id = self.current_view_id
        
        self.current_state = current_state_array
        self.current_state_str = state_str
        self.current_view_id = view_id
        self.cube_simulator.set_state(current_state_array)
        
        # Apply action
        action_type, action_name = self.action_space_config.get_action(action)
        
        if action_type == ActionType.CUBE:
            # Apply cube move
            success = self.cube_simulator.apply_move(action_name)
            if success:
                next_state_array = self.cube_simulator.get_state()
                next_state_str = serialize_state(next_state_array)
                next_view_id = view_id  # View doesn't change for cube moves
            else:
                # Invalid move
                next_state_str = state_str
                next_view_id = view_id
        
        elif action_type == ActionType.VIEW:
            # Apply view transition
            success = self._apply_view_transition(action_name)
            next_state_str = state_str  # Cube state doesn't change for view moves
            next_view_id = self.current_view_id
        
        else:
            # Invalid action
            next_state_str = state_str
            next_view_id = view_id
        
        # Get observation from new state/view
        next_state_array = deserialize_state(next_state_str)
        observation = self.face_renderer.get_observation(next_state_array, next_view_id)
        
        # Restore original state
        self.current_state = original_state
        self.current_state_str = original_state_str
        self.current_view_id = original_view_id
        self.cube_simulator.set_state(original_state)
        
        return (next_state_str, next_view_id), observation
    
    def _apply_view_transition(self, action_name: str) -> bool:
        """
        Apply view transition action.
        
        Args:
            action_name: View action name ('view_rot0', 'view_rot90', 'view_rot180', 'view_rot270')
            
        Returns:
            True if transition was successful
        """
        # Map action name to neighbor index
        action_to_neighbor = {
            'view_rot0': 0,
            'view_rot90': 1,
            'view_rot180': 2,
            'view_rot270': 3
        }
        
        if action_name not in action_to_neighbor:
            return False
        
        neighbor_idx = action_to_neighbor[action_name]
        available_neighbors = self.view_neighbors[self.current_view_id]
        
        if neighbor_idx < len(available_neighbors):
            self.current_view_id = available_neighbors[neighbor_idx]
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (rendered image)"""
        return self.face_renderer.get_observation(self.current_state, self.current_view_id)
    
    def render(self):
        """Render the current state"""
        if self.render_mode == "human":
            # For human rendering, return the image
            return self.face_renderer.get_observation(self.current_state, self.current_view_id)
        elif self.render_mode == "rgb_array":
            return self.face_renderer.get_observation(self.current_state, self.current_view_id)
        else:
            return None
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            'state_string': self.current_state_str,
            'view_id': self.current_view_id,
            'view_type': 'face',
            'available_view_transitions': self.view_neighbors[self.current_view_id]
        }
    
    def set_state_info(self, state_str: str, view_id: int):
        """Set state from string representation and view ID"""
        state_array = deserialize_state(state_str)
        self.set_state(state_array)
        self.current_state_str = state_str
        
        if 0 <= view_id < 24:
            self.current_view_id = view_id
        else:
            raise ValueError(f"View ID must be 0-23, got {view_id}")
    
    def get_action_names(self) -> List[str]:
        """Get list of action names"""
        actions = []
        for i in range(self.action_space.n):
            _, action_name = self.action_space_config.get_action(i)
            actions.append(action_name)
        return actions
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self.face_renderer, 'cleanup'):
            self.face_renderer.cleanup() 