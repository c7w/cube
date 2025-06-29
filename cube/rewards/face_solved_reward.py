import numpy as np
from cube.core.base_reward import RewardFunction

class FaceSolvedReward(RewardFunction):
    """
    Rewards the agent for increasing the number of solved faces.
    This encourages intermediate progress.
    """
    def __init__(self, solved_bonus: float = 10.0, face_solved_bonus: float = 1.0, step_penalty: float = -0.01, **kwargs):
        super().__init__(**kwargs)
        self.solved_bonus = solved_bonus
        self.face_solved_bonus = face_solved_bonus
        self.step_penalty = step_penalty

    def _count_solved_faces(self, state: np.ndarray) -> int:
        """Counts how many of the 6 faces are solved."""
        count = 0
        for i in range(6):
            face_color = i
            if np.all(state[i*9:(i+1)*9] == face_color):
                count += 1
        return count

    def calculate_reward(self, previous_state: np.ndarray, action: str, current_state: np.ndarray, is_done: bool, step_count: int, max_steps: int) -> float:
        if is_done:
            return self.solved_bonus

        prev_solved_faces = self._count_solved_faces(previous_state)
        current_solved_faces = self._count_solved_faces(current_state)
        
        reward = (current_solved_faces - prev_solved_faces) * self.face_solved_bonus
        
        return reward + self.step_penalty