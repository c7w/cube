import numpy as np
from cube.core.base_reward import RewardFunction
from cube.utils import heuristic_utils

class HeuristicReward(RewardFunction):
    """
    A reward function that gives rewards based on progress through the 
    beginner's method for solving a Rubik's cube.
    """
    def __init__(self,
                 step_weights: list = [1, 2, 3, 4, 5, 6, 7],
                 solved_bonus: float = 100.0,
                 step_penalty: float = -0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.step_weights = step_weights
        self.solved_bonus = solved_bonus
        self.step_penalty = step_penalty
        self.heuristic_functions = [
            heuristic_utils.step_1_the_cross,
            heuristic_utils.step_2_the_corners,
            heuristic_utils.step_3_the_second_layer,
            heuristic_utils.step_4_the_last_layer_cross,
            heuristic_utils.step_5_the_last_layer_edges,
            heuristic_utils.step_6_the_last_layer_corners_placed,
            heuristic_utils.step_7_the_last_layer_corners_oriented,
        ]

    def _calculate_heuristic_score(self, state: np.ndarray) -> float:
        """Calculates the total heuristic score for a given state."""
        score = 0.0
        for i, func in enumerate(self.heuristic_functions):
            score += self.step_weights[i] * func(state)
        return score

    def calculate_reward(self, previous_state: np.ndarray, action: str, current_state: np.ndarray, is_done: bool, step_count: int, max_steps: int) -> float:
        if is_done:
            return self.solved_bonus

        previous_score = self._calculate_heuristic_score(previous_state)
        current_score = self._calculate_heuristic_score(current_state)
        
        # Reward is the change in heuristic score
        reward = current_score - previous_score
        
        return reward + self.step_penalty
