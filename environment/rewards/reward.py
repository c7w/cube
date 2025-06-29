"""
Reward functions for the CubeBench environment.
Defines various reward schemes for reinforcement learning.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable
from enum import Enum
from abc import ABC, abstractmethod


class RewardType(Enum):
    """Types of reward functions"""
    SPARSE = "sparse"           # Only reward for completion
    DENSE = "dense"             # Reward for progress
    HYBRID = "hybrid"           # Combination of sparse and dense
    CUSTOM = "custom"           # Custom reward function


class RewardFunction(ABC):
    """Abstract base class for reward functions"""
    
    @abstractmethod
    def calculate_reward(self, 
                        current_state: np.ndarray,
                        action: str,
                        next_state: np.ndarray,
                        is_done: bool,
                        step_count: int,
                        max_steps: int) -> float:
        """
        Calculate reward for a transition.
        
        Args:
            current_state: Current cube state (6x9 array)
            action: Action taken
            next_state: Next cube state (6x9 array)
            is_done: Whether episode is done
            step_count: Current step number
            max_steps: Maximum steps allowed
            
        Returns:
            Reward value
        """
        pass
    
    def reset(self):
        """Reset any internal state"""
        pass


class DummyReward(RewardFunction):
    """
    Dummy reward function.
    Returns 0 reward.
    """
    def calculate_reward(self,
                         current_state: np.ndarray,
                         action: str,
                         next_state: np.ndarray,
                         is_done: bool,
                         step_count: int,
                         max_steps: int) -> float:
        return 0.0
    
    def reset(self):
        pass


class SparseReward(RewardFunction):
    """
    Sparse reward function.
    Only gives reward when the cube is solved.
    """
    
    def __init__(self, completion_reward: float = 1000.0, step_penalty: float = 0.0):
        """
        Initialize sparse reward function.
        
        Args:
            completion_reward: Reward for solving the cube
            step_penalty: Penalty per step (usually 0 for sparse)
        """
        self.completion_reward = completion_reward
        self.step_penalty = step_penalty
    
    def calculate_reward(self, 
                        current_state: np.ndarray,
                        action: str,
                        next_state: np.ndarray,
                        is_done: bool,
                        step_count: int,
                        max_steps: int) -> float:
        """Calculate sparse reward"""
        reward = -self.step_penalty  # Small step penalty
        
        if is_done and self._is_solved(next_state):
            reward += self.completion_reward
        
        return reward
    
    def _is_solved(self, state: np.ndarray) -> bool:
        """Check if cube is solved"""
        for face_idx in range(6):
            center_color = state[face_idx, 4]
            if not np.all(state[face_idx] == center_color):
                return False
        return True


class DenseReward(RewardFunction):
    """
    Dense reward function.
    Gives rewards for progress towards solving the cube.
    """
    
    def __init__(self, 
                 completion_reward: float = 1000.0,
                 step_penalty: float = 0.1,
                 face_reward: float = 10.0,
                 progress_reward: float = 5.0):
        """
        Initialize dense reward function.
        
        Args:
            completion_reward: Reward for solving the cube
            step_penalty: Penalty per step
            face_reward: Reward for each solved face
            progress_reward: Reward for progress (more solved squares)
        """
        self.completion_reward = completion_reward
        self.step_penalty = step_penalty
        self.face_reward = face_reward
        self.progress_reward = progress_reward
        self.previous_solved_faces = 0
        self.previous_solved_squares = 0
    
    def calculate_reward(self, 
                        current_state: np.ndarray,
                        action: str,
                        next_state: np.ndarray,
                        is_done: bool,
                        step_count: int,
                        max_steps: int) -> float:
        """Calculate dense reward"""
        reward = -self.step_penalty
        
        # Count solved faces and squares
        current_solved_faces = self._count_solved_faces(current_state)
        current_solved_squares = self._count_solved_squares(current_state)
        next_solved_faces = self._count_solved_faces(next_state)
        next_solved_squares = self._count_solved_squares(next_state)
        
        # Reward for new solved faces
        new_solved_faces = next_solved_faces - current_solved_faces
        reward += new_solved_faces * self.face_reward
        
        # Reward for progress (more solved squares)
        progress = next_solved_squares - current_solved_squares
        reward += progress * self.progress_reward
        
        # Completion reward
        if is_done and self._is_solved(next_state):
            reward += self.completion_reward
        
        return reward
    
    def _count_solved_faces(self, state: np.ndarray) -> int:
        """Count number of solved faces"""
        solved_count = 0
        for face_idx in range(6):
            center_color = state[face_idx, 4]
            if np.all(state[face_idx] == center_color):
                solved_count += 1
        return solved_count
    
    def _count_solved_squares(self, state: np.ndarray) -> int:
        """Count total number of correctly colored squares"""
        correct_squares = 0
        for face_idx in range(6):
            center_color = state[face_idx, 4]
            correct_squares += np.sum(state[face_idx] == center_color)
        return correct_squares
    
    def _is_solved(self, state: np.ndarray) -> bool:
        """Check if cube is solved"""
        return self._count_solved_faces(state) == 6


class HybridReward(RewardFunction):
    """
    Hybrid reward function.
    Combines sparse and dense rewards with configurable weights.
    """
    
    def __init__(self,
                 completion_reward: float = 1000.0,
                 step_penalty: float = 0.05,
                 face_reward: float = 5.0,
                 progress_reward: float = 2.0,
                 sparse_weight: float = 0.3,
                 dense_weight: float = 0.7):
        """
        Initialize hybrid reward function.
        
        Args:
            completion_reward: Reward for solving the cube
            step_penalty: Penalty per step
            face_reward: Reward for each solved face
            progress_reward: Reward for progress
            sparse_weight: Weight for sparse component
            dense_weight: Weight for dense component
        """
        self.completion_reward = completion_reward
        self.step_penalty = step_penalty
        self.face_reward = face_reward
        self.progress_reward = progress_reward
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        
        # Initialize sub-reward functions
        self.sparse_reward = SparseReward(completion_reward, 0.0)
        self.dense_reward = DenseReward(0.0, step_penalty, face_reward, progress_reward)
    
    def calculate_reward(self, 
                        current_state: np.ndarray,
                        action: str,
                        next_state: np.ndarray,
                        is_done: bool,
                        step_count: int,
                        max_steps: int) -> float:
        """Calculate hybrid reward"""
        # Calculate sparse and dense rewards
        sparse_r = self.sparse_reward.calculate_reward(
            current_state, action, next_state, is_done, step_count, max_steps)
        dense_r = self.dense_reward.calculate_reward(
            current_state, action, next_state, is_done, step_count, max_steps)
        
        # Combine rewards
        reward = self.sparse_weight * sparse_r + self.dense_weight * dense_r
        
        return reward


class CustomReward(RewardFunction):
    """
    Custom reward function.
    Allows user-defined reward calculation.
    """
    
    def __init__(self, reward_func: Callable):
        """
        Initialize custom reward function.
        
        Args:
            reward_func: Custom reward function with signature:
                reward_func(current_state, action, next_state, is_done, step_count, max_steps) -> float
        """
        self.reward_func = reward_func
    
    def calculate_reward(self, 
                        current_state: np.ndarray,
                        action: str,
                        next_state: np.ndarray,
                        is_done: bool,
                        step_count: int,
                        max_steps: int) -> float:
        """Calculate custom reward"""
        return self.reward_func(current_state, action, next_state, is_done, step_count, max_steps)


class RewardFactory:
    """Factory for creating reward functions"""
    
    @staticmethod
    def create_reward(reward_type: RewardType, **kwargs) -> RewardFunction:
        """
        Create a reward function of the specified type.
        
        Args:
            reward_type: Type of reward function
            **kwargs: Additional arguments for the reward function
            
        Returns:
            Reward function instance
        """
        if reward_type == RewardType.SPARSE:
            return SparseReward(**kwargs)
        elif reward_type == RewardType.DENSE:
            return DenseReward(**kwargs)
        elif reward_type == RewardType.HYBRID:
            return HybridReward(**kwargs)
        elif reward_type == RewardType.CUSTOM:
            if 'reward_func' not in kwargs:
                raise ValueError("Custom reward function requires 'reward_func' parameter")
            return CustomReward(kwargs['reward_func'])
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")


# Predefined reward functions for common use cases
SPARSE_REWARD = SparseReward()
DENSE_REWARD = DenseReward()
HYBRID_REWARD = HybridReward()


def create_efficiency_reward() -> RewardFunction:
    """
    Create a reward function that encourages efficient solving.
    Penalizes long solution sequences.
    """
    def efficiency_reward(current_state, action, next_state, is_done, step_count, max_steps):
        reward = -0.1  # Small step penalty
        
        # Bonus for solving
        if is_done and _is_solved(next_state):
            # Bonus decreases with step count
            efficiency_bonus = max(100, 1000 - step_count * 10)
            reward += efficiency_bonus
        
        return reward
    
    def _is_solved(state):
        for face_idx in range(6):
            center_color = state[face_idx, 4]
            if not np.all(state[face_idx] == center_color):
                return False
        return True
    
    return CustomReward(efficiency_reward)


def create_exploration_reward() -> RewardFunction:
    """
    Create a reward function that encourages exploration.
    Rewards for discovering new states.
    """
    class ExplorationReward(RewardFunction):
        def __init__(self):
            self.visited_states = set()
            self.completion_reward = 1000.0
            self.exploration_reward = 1.0
            self.step_penalty = 0.05
        
        def calculate_reward(self, current_state, action, next_state, is_done, step_count, max_steps):
            reward = -self.step_penalty
            
            # Convert state to hashable format
            state_hash = hash(next_state.tobytes())
            
            # Reward for exploring new states
            if state_hash not in self.visited_states:
                reward += self.exploration_reward
                self.visited_states.add(state_hash)
            
            # Completion reward
            if is_done and self._is_solved(next_state):
                reward += self.completion_reward
            
            return reward
        
        def _is_solved(self, state):
            for face_idx in range(6):
                center_color = state[face_idx, 4]
                if not np.all(state[face_idx] == center_color):
                    return False
            return True
        
        def reset(self):
            self.visited_states.clear()
    
    return ExplorationReward()


def get_reward_function(type: str = "dummy", **kwargs) -> RewardFunction:
    if type == "dummy":
        return DummyReward()
    elif type == "sparse":
        return SparseReward(**kwargs)
    elif type == "dense":
        return DenseReward(**kwargs)
    elif type == "hybrid":
        return HybridReward(**kwargs)
    elif type == "custom":
        return CustomReward(**kwargs)