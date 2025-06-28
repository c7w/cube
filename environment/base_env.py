"""
Base environment class for CubeBench.
Compatible with Gymnasium API for RL training.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod

from .cube_simulator import CubeSimulator
from .action_space import ActionSpace, ActionType
from .renderer import CubeRenderer, RenderMode
from .reward import RewardFunction, RewardType, RewardFactory


class CubeBenchEnv(gym.Env):
    """
    Base environment for CubeBench.
    Implements the Gymnasium interface for RL training.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "symbolic"],
        "render_fps": 4,
    }
    
    def __init__(self,
                 action_space_config: Optional[ActionSpace] = None,
                 reward_function: Optional[RewardFunction] = None,
                 render_mode: str = "rgb_array",
                 max_steps: int = 1000,
                 scramble_moves: int = 20,
                 observation_mode: str = "image"):
        """
        Initialize the environment.
        
        Args:
            action_space_config: Action space configuration
            reward_function: Reward function to use
            render_mode: Rendering mode ("human", "rgb_array", "symbolic")
            max_steps: Maximum steps per episode
            scramble_moves: Number of moves to scramble the cube
            observation_mode: Observation mode ("image", "symbolic", "both")
        """
        super().__init__()
        
        # Initialize components
        self.cube_simulator = CubeSimulator()
        self.action_space_config = action_space_config or ActionSpace()
        self.max_steps = max_steps
        self.scramble_moves = scramble_moves
        self.observation_mode = observation_mode
        
        # Initialize reward function
        if reward_function is None:
            reward_function = RewardFactory.create_reward(RewardType.HYBRID)
        self.reward_function = reward_function
        
        # Initialize renderer
        self.render_mode = render_mode
        self.renderer = CubeRenderer(
            mode=RenderMode.BOTH if observation_mode == "both" else RenderMode.IMAGE,
            image_size=400
        )
        
        # Environment state
        self.step_count = 0
        self.current_state = None
        self.view_angle = (0.0, 0.0, 0.0)  # (x, y, z) in degrees
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Reset environment
        self.reset()
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Action space
        self.action_space = self.action_space_config.to_gym_space()
        
        # Observation space
        if self.observation_mode == "image":
            # RGB image observation
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(400, 400, 3), dtype=np.uint8
            )
        elif self.observation_mode == "symbolic":
            # Symbolic state observation (6 faces * 9 squares)
            self.observation_space = spaces.Box(
                low=0, high=5, shape=(6, 9), dtype=np.int8
            )
        elif self.observation_mode == "both":
            # Combined observation space
            self.observation_space = spaces.Dict({
                'image': spaces.Box(low=0, high=255, shape=(400, 400, 3), dtype=np.uint8),
                'symbolic': spaces.Box(low=0, high=5, shape=(6, 9), dtype=np.int8)
            })
        else:
            raise ValueError(f"Unknown observation mode: {self.observation_mode}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset cube simulator
        self.cube_simulator.reset()
        
        # Scramble the cube
        if self.scramble_moves > 0:
            self.cube_simulator.scramble(self.scramble_moves)
        
        # Reset environment state
        self.step_count = 0
        self.current_state = self.cube_simulator.get_state()
        self.view_angle = (0.0, 0.0, 0.0)
        
        # Reset reward function
        self.reward_function.reset()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Create info dict
        info = {
            'step_count': self.step_count,
            'solved_faces': self.cube_simulator.get_solved_faces_count(),
            'is_solved': self.cube_simulator.is_solved(),
            'action_history': self.cube_simulator.get_action_history()
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action index
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get action details
        action_type, action_name = self.action_space_config.get_action(action)
        
        # Save current state for reward calculation
        current_state = self.current_state.copy()
        
        # Apply action
        if action_type == ActionType.CUBE:
            # Apply cube move
            success = self.cube_simulator.apply_move(action_name)
            if not success:
                # Invalid move - penalize heavily
                reward = -100.0
                terminated = True
                truncated = False
            else:
                # Update current state
                self.current_state = self.cube_simulator.get_state()
                reward = self._calculate_reward(current_state, action_name, self.current_state)
                terminated = self.cube_simulator.is_solved()
                truncated = False
        
        elif action_type == ActionType.VIEW:
            # Apply view transformation
            self._apply_view_action(action_name)
            reward = -0.01  # Small penalty for view changes
            terminated = False
            truncated = False
        
        elif action_type == ActionType.SPECIAL:
            # Handle special actions
            reward, terminated, truncated = self._handle_special_action(action_name)
        
        else:
            # Unknown action type
            reward = -100.0
            terminated = True
            truncated = False
        
        # Increment step count
        self.step_count += 1
        
        # Check for truncation
        if self.step_count >= self.max_steps:
            truncated = True
        
        # Get observation
        observation = self._get_observation()
        
        # Create info dict
        info = {
            'step_count': self.step_count,
            'action': action_name,
            'action_type': action_type.value,
            'solved_faces': self.cube_simulator.get_solved_faces_count(),
            'is_solved': self.cube_simulator.is_solved(),
            'action_history': self.cube_simulator.get_action_history(),
            'view_angle': self.view_angle
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Any:
        """Get current observation based on observation mode"""
        if self.observation_mode == "image":
            return self.renderer._render_image(self.current_state, self.view_angle)
        elif self.observation_mode == "symbolic":
            return self.current_state
        elif self.observation_mode == "both":
            result = self.renderer.render(self.current_state, self.view_angle)
            return {
                'image': result['image'],
                'symbolic': self.current_state
            }
        else:
            raise ValueError(f"Unknown observation mode: {self.observation_mode}")
    
    def _calculate_reward(self, current_state: np.ndarray, action: str, next_state: np.ndarray) -> float:
        """Calculate reward for the transition"""
        is_done = self.cube_simulator.is_solved()
        
        # Convert states to 6x9 format for reward function compatibility
        def convert_to_6x9(state_54):
            """Convert 54-element state to 6x9 format"""
            faces = np.zeros((6, 9), dtype=int)
            face_order = ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN']
            face_positions = {
                'FRONT': 0, 'BACK': 9, 'LEFT': 18, 'RIGHT': 27, 'UP': 36, 'DOWN': 45
            }
            
            for i, face_name in enumerate(face_order):
                start_pos = face_positions[face_name]
                faces[i] = state_54[start_pos:start_pos+9]
            
            return faces
        
        current_state_6x9 = convert_to_6x9(current_state)
        next_state_6x9 = convert_to_6x9(next_state)
        
        return self.reward_function.calculate_reward(
            current_state_6x9, action, next_state_6x9, is_done, self.step_count, self.max_steps
        )
    
    def _apply_view_action(self, action_name: str):
        """Apply view transformation action"""
        if action_name == "rotate_x":
            self.view_angle = (self.view_angle[0] + 90, self.view_angle[1], self.view_angle[2])
        elif action_name == "rotate_y":
            self.view_angle = (self.view_angle[0], self.view_angle[1] + 90, self.view_angle[2])
        elif action_name == "rotate_z":
            self.view_angle = (self.view_angle[0], self.view_angle[1], self.view_angle[2] + 90)
        elif action_name == "reset_view":
            self.view_angle = (0.0, 0.0, 0.0)
        
        # Normalize angles to [0, 360)
        self.view_angle = tuple(angle % 360 for angle in self.view_angle)
    
    def _handle_special_action(self, action_name: str) -> Tuple[float, bool, bool]:
        """Handle special actions"""
        if action_name == "solve":
            # Solve the cube (for evaluation)
            # This would typically use a solving algorithm
            reward = 1000.0 if self.cube_simulator.is_solved() else -100.0
            return reward, True, False
        
        elif action_name == "scramble":
            # Scramble the cube
            self.cube_simulator.scramble(self.scramble_moves)
            self.current_state = self.cube_simulator.get_state()
            return -10.0, False, False
        
        elif action_name == "undo":
            # Undo last move
            success = self.cube_simulator.undo_last_move()
            if success:
                self.current_state = self.cube_simulator.get_state()
                return -1.0, False, False
            else:
                return -10.0, False, False
        
        else:
            # Unknown special action
            return -100.0, True, False
    
    def render(self):
        """Render the current state"""
        if self.render_mode == "human":
            # For human rendering, we would display the image
            # This is a placeholder - in practice you might use cv2.imshow or similar
            return self.renderer._render_image(self.current_state, self.view_angle)
        elif self.render_mode == "rgb_array":
            return self.renderer._render_image(self.current_state, self.view_angle)
        elif self.render_mode == "symbolic":
            return self.renderer._render_symbolic(self.current_state)
        else:
            raise ValueError(f"Unknown render mode: {self.render_mode}")
    
    def close(self):
        """Close the environment"""
        pass
    
    def get_state(self) -> np.ndarray:
        """Get current cube state"""
        return self.current_state.copy()
    
    def set_state(self, state: np.ndarray):
        """Set cube state (for debugging/testing)"""
        self.current_state = state.copy()
        self.cube_simulator.set_state(state)
    
    def get_action_history(self) -> List[str]:
        """Get action history"""
        return self.cube_simulator.get_action_history()
    
    def get_solved_faces_count(self) -> int:
        """Get number of solved faces"""
        return self.cube_simulator.get_solved_faces_count()
    
    def is_solved(self) -> bool:
        """Check if cube is solved"""
        return self.cube_simulator.is_solved()
    
    def get_view_angle(self) -> Tuple[float, float, float]:
        """Get current view angle"""
        return self.view_angle
    
    def set_view_angle(self, angle: Tuple[float, float, float]):
        """Set view angle"""
        self.view_angle = tuple(angle % 360 for angle in angle)


# Convenience function to create environment
def make_cubebench_env(action_space_config: Optional[ActionSpace] = None,
                      reward_function: Optional[RewardFunction] = None,
                      render_mode: str = "rgb_array",
                      max_steps: int = 1000,
                      scramble_moves: int = 20,
                      observation_mode: str = "image") -> CubeBenchEnv:
    """
    Create a CubeBench environment with specified configuration.
    
    Args:
        action_space_config: Action space configuration
        reward_function: Reward function to use
        render_mode: Rendering mode
        max_steps: Maximum steps per episode
        scramble_moves: Number of moves to scramble the cube
        observation_mode: Observation mode
        
    Returns:
        Configured CubeBench environment
    """
    return CubeBenchEnv(
        action_space_config=action_space_config,
        reward_function=reward_function,
        render_mode=render_mode,
        max_steps=max_steps,
        scramble_moves=scramble_moves,
        observation_mode=observation_mode
    ) 