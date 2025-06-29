import numpy as np
from typing import Any, Dict, Optional
from gymnasium import spaces
from environment.envs.base_env import BaseCubeEnv
from environment.utils.cube_simulator import CubeSimulator
from environment.action_space import ActionSpace, CUBE_ONLY_ACTION_SPACE
from environment.rewards.reward import RewardFunction, get_reward_function
from environment.utils.state_utils import serialize_state as state_to_color_string, COLOR_TO_LETTER

def make_sequence_env(max_steps: int = 1000, scramble_moves: int = 20, reward_function_config: Dict[str, Any] = {}):
    cube_manager = CubeSimulator()
    action_space_config = CUBE_ONLY_ACTION_SPACE
    reward_function = get_reward_function(**reward_function_config)
    return SequenceCubeEnv(cube_manager, action_space_config, max_steps, scramble_moves, reward_function)

class SequenceRenderer:
    """
    Renderer for sequence-based cube representation.
    Handles conversion between state arrays and token sequences.
    """
    def get_observation(self, state: np.ndarray) -> str:
        return state_to_color_string(state)

# --- SequenceEnv implementation ---
class SequenceCubeEnv(BaseCubeEnv):
    """
    Sequence-based CubeBench environment.
    Only implements the required abstract methods with minimal logic.
    """
    def __init__(self, cube_manager: CubeSimulator, action_space_config: ActionSpace, 
                 max_steps: int = 1000, scramble_moves: int = 20, 
                 reward_function: RewardFunction = None):
        # Initialize renderer first
        self.renderer = SequenceRenderer()
        
        # Call parent constructor with dependencies
        super().__init__(
            cube_manager=cube_manager,
            action_space_config=action_space_config,
            reward_function=reward_function,
            max_steps=max_steps,
            scramble_moves=scramble_moves
        )
        
        # Now reset after everything is initialized
        self.reset()
    
    def _setup_observation_space(self):
        # Observation is a 54-char string with color chars representing cube faces
        self.observation_space = spaces.Text(54, charset=list(COLOR_TO_LETTER.values()))

    def get_observation(self) -> Any:
        # Use the renderer for observation
        return self.renderer.get_observation(self.cube_manager.get_state())

    def _reset_camera_params(self):
        # Sequence env does not use camera/view
        self.camera_params = None

    def _update_camera_params(self, action_name: str):
        # Sequence env does not use camera/view
        pass