# Makes the 'rewards' directory a package.

from .solved_reward import SolvedReward
from .face_solved_reward import FaceSolvedReward
from .sticker_reward import StickerReward

__all__ = ["SolvedReward", "FaceSolvedReward", "StickerReward"]