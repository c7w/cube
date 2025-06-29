import numpy as np
from cube.core.base_reward import RewardFunction

class StickerReward(RewardFunction):
    """
    Provides a dense reward based on heuristic progress, measured by the
    number of stickers that are in their correct face.
    """
    def __init__(self, solved_bonus: float = 10.0, sticker_progress_bonus: float = 0.1, step_penalty: float = -0.01, **kwargs):
        super().__init__(**kwargs)
        self.solved_bonus = solved_bonus
        self.sticker_progress_bonus = sticker_progress_bonus
        self.step_penalty = step_penalty

    def _count_correct_stickers(self, state: np.ndarray) -> int:
        """Counts how many stickers are on their correct destination face."""
        # A sticker at index i is on its correct face if its color is floor(i/9)
        solved_state_colors = np.floor(np.arange(54) / 9)
        return np.sum(state == solved_state_colors)

    def calculate_reward(self, previous_state: np.ndarray, action: str, current_state: np.ndarray, is_done: bool, step_count: int, max_steps: int) -> float:
        if is_done:
            return self.solved_bonus

        prev_correct = self._count_correct_stickers(previous_state)
        current_correct = self._count_correct_stickers(current_state)
        
        reward = (current_correct - prev_correct) * self.sticker_progress_bonus
        
        return reward + self.step_penalty 