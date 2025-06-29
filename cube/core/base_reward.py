from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np
from cube.rewards import SolvedReward, FaceSolvedReward, StickerReward
from cube.rewards.solved_reward import SolvedReward
from cube.rewards.face_solved_reward import FaceSolvedReward
from cube.rewards.sticker_reward import StickerReward
from cube.rewards.heuristic_reward import HeuristicReward

class RewardFunction(ABC):
    """Abstract base class for reward functions"""
    
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def calculate_reward(self, 
                        previous_state: np.ndarray,
                        action: str,
                        current_state: np.ndarray,
                        is_done: bool,
                        step_count: int,
                        max_steps: int) -> float:
        """
        Calculate reward for a transition.
        
        Args:
            previous_state: Cube state before the action
            action: Action taken
            current_state: Cube state after the action
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


class DummyRewardFunction(RewardFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def calculate_reward(self, 
                        previous_state: np.ndarray,
                        action: str,
                        current_state: np.ndarray,
                        is_done: bool,
                        step_count: int,
                        max_steps: int) -> float:
        return 0.0
    
    def reset(self):
        pass

# Use registry to get reward function
REWARD_FUNCTION_REGISTRY = {
    "dummy": DummyRewardFunction,
    "solved": SolvedReward,
    "face_solved": FaceSolvedReward,
    "sticker": StickerReward,
    "heuristic": HeuristicReward
}

def get_reward_function(**reward_function_config: Dict[str, Any]) -> RewardFunction:
    reward_type = reward_function_config.get("type", "dummy")
    if reward_type not in REWARD_FUNCTION_REGISTRY:
        raise ValueError(f"Unknown reward function type: {reward_type}")
    return REWARD_FUNCTION_REGISTRY[reward_type](**reward_function_config)