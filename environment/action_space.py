"""
Action space definitions for the CubeBench environment.
"""

from enum import Enum
from typing import List, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CubeAction(Enum):
    """Cube rotation actions (18 basic moves)"""
    # Face rotations
    F = "F"      # Front face clockwise
    F_PRIME = "F'"  # Front face counter-clockwise
    B = "B"      # Back face clockwise
    B_PRIME = "B'"  # Back face counter-clockwise
    L = "L"      # Left face clockwise
    L_PRIME = "L'"  # Left face counter-clockwise
    R = "R"      # Right face clockwise
    R_PRIME = "R'"  # Right face counter-clockwise
    U = "U"      # Up face clockwise
    U_PRIME = "U'"  # Up face counter-clockwise
    D = "D"      # Down face clockwise
    D_PRIME = "D'"  # Down face counter-clockwise


class ViewAction(Enum):
    """View manipulation actions"""
    ROTATE_X = "rotate_x"      # Rotate view around X axis
    ROTATE_Y = "rotate_y"      # Rotate view around Y axis
    ROTATE_Z = "rotate_z"      # Rotate view around Z axis
    RESET_VIEW = "reset_view"  # Reset to default view


class SpecialAction(Enum):
    """Special actions"""
    SOLVE = "solve"        # Solve the cube (for evaluation)
    SCRAMBLE = "scramble"  # Scramble the cube
    UNDO = "undo"          # Undo last action


class ActionType(Enum):
    """Action type categories"""
    CUBE = "cube"
    VIEW = "view"
    SPECIAL = "special"


class ActionSpace:
    """Action space for the CubeBench environment"""
    
    def __init__(self, include_view_actions: bool = True, include_special_actions: bool = False, 
                 custom_view_actions: List[str] = None):
        """
        Initialize action space.
        
        Args:
            include_view_actions: Whether to include view manipulation actions
            include_special_actions: Whether to include special actions (solve, scramble, undo)
            custom_view_actions: Custom list of view actions (overrides default view actions)
        """
        self.include_view_actions = include_view_actions
        self.include_special_actions = include_special_actions
        self.custom_view_actions = custom_view_actions
        
        # Build action list
        self.actions = []
        self.action_to_idx = {}
        self.idx_to_action = {}
        
        # Add cube actions (always included)
        for action in CubeAction:
            self.actions.append((ActionType.CUBE, action.value))
        
        # Add view actions if enabled
        if include_view_actions:
            if custom_view_actions:
                # Use custom view actions
                for action_name in custom_view_actions:
                    self.actions.append((ActionType.VIEW, action_name))
            else:
                # Use default view actions
                for action in ViewAction:
                    self.actions.append((ActionType.VIEW, action.value))
        
        # Add special actions if enabled
        if include_special_actions:
            for action in SpecialAction:
                self.actions.append((ActionType.SPECIAL, action.value))
        
        # Create mappings
        for idx, (action_type, action_name) in enumerate(self.actions):
            self.action_to_idx[(action_type, action_name)] = idx
            self.idx_to_action[idx] = (action_type, action_name)
        
        self.n_actions = len(self.actions)
    
    def get_action(self, action_idx: int) -> tuple[ActionType, str]:
        """Get action from index"""
        if action_idx < 0 or action_idx >= self.n_actions:
            raise ValueError(f"Action index {action_idx} out of range [0, {self.n_actions})")
        return self.idx_to_action[action_idx]
    
    def get_action_idx(self, action_type: ActionType, action_name: str) -> int:
        """Get action index from action type and name"""
        if (action_type, action_name) not in self.action_to_idx:
            raise ValueError(f"Action {(action_type, action_name)} not found in action space")
        return self.action_to_idx[(action_type, action_name)]
    
    def get_cube_actions(self) -> List[str]:
        """Get list of cube rotation actions"""
        return [action_name for action_type, action_name in self.actions if action_type == ActionType.CUBE]
    
    def get_view_actions(self) -> List[str]:
        """Get list of view manipulation actions"""
        return [action_name for action_type, action_name in self.actions if action_type == ActionType.VIEW]
    
    def get_special_actions(self) -> List[str]:
        """Get list of special actions"""
        return [action_name for action_type, action_name in self.actions if action_type == ActionType.SPECIAL]
    
    def to_gym_space(self) -> gym.Space:
        """Convert to Gymnasium space"""
        return spaces.Discrete(self.n_actions)
    
    def __len__(self) -> int:
        return self.n_actions
    
    def __repr__(self) -> str:
        return f"ActionSpace(n_actions={self.n_actions}, include_view={self.include_view_actions}, include_special={self.include_special_actions})"


# Predefined action spaces for common use cases
CUBE_ONLY_ACTION_SPACE = ActionSpace(include_view_actions=False, include_special_actions=False)
FULL_ACTION_SPACE = ActionSpace(include_view_actions=True, include_special_actions=True) 