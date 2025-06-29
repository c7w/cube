import numpy as np
from cube.core.base_reward import RewardFunction

class SolvedReward(RewardFunction):
    """
    Gives a large positive reward only when the cube is solved,
    and a small penalty for every step taken.
    """
    def __init__(self, solved_bonus: float = 10.0, step_penalty: float = -0.01, **kwargs):
        super().__init__(**kwargs)
        self.solved_bonus = solved_bonus
        self.step_penalty = step_penalty

    def calculate_reward(self, previous_state: np.ndarray, action: str, current_state: np.ndarray, is_done: bool, step_count: int, max_steps: int) -> float:
        if is_done:
            return self.solved_bonus
        return self.step_penalty 