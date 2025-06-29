import numpy as np
from typing import Any, Dict, Optional
from gymnasium import spaces
from environment.envs.base_env import BaseCubeEnv
from environment.utils.cube_simulator import CubeSimulator
from environment.action_space import CUBE_ONLY_ACTION_SPACE, ActionSpace
from environment.rewards.reward import RewardFunction, get_reward_function

COLOR_CHARS = "WYROBG"  # 0:W, 1:Y, 2:R, 3:O, 4:B, 5:G

def serialize_state(state: np.ndarray) -> str:
    return ''.join(COLOR_CHARS[x] for x in state)

def deserialize_state(sequence: str) -> np.ndarray:
    color_to_idx = {c: i for i, c in enumerate(COLOR_CHARS)}
    return np.array([color_to_idx[c] for c in sequence], dtype=int)

def validate_state_string(sequence: str) -> bool:
    return len(sequence) == 54 and all(c in COLOR_CHARS for c in sequence)

class SequenceRenderer:
    """
    Renderer for sequence-based cube representation.
    Handles conversion between state arrays and token sequences.
    """
    def render_state_to_sequence(self, state: np.ndarray) -> str:
        return serialize_state(state)
    def render_sequence_to_state(self, sequence: str) -> np.ndarray:
        return deserialize_state(sequence)
    def render(self, state: np.ndarray) -> Dict[str, Any]:
        sequence = self.render_state_to_sequence(state)
        return {
            'sequence': sequence,
            'length': len(sequence),
            'valid': validate_state_string(sequence)
        }
    def get_observation(self, state: np.ndarray) -> str:
        return self.render_state_to_sequence(state)
    def validate_observation(self, observation: str) -> bool:
        return validate_state_string(observation)

# --- SequenceEnv implementation ---
class SequenceCubeEnv(BaseCubeEnv):
    """
    Sequence-based CubeBench environment.
    Only implements the required abstract methods with minimal logic.
    """
    def __init__(self, cube_manager: CubeSimulator, action_space_config: ActionSpace, max_steps: int = 1000, scramble_moves: int = 20, reward_function: RewardFunction = None):
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
        # Observation is a 54-char string with digits 0-5 representing cube faces
        self.observation_space = spaces.Text(54, charset=COLOR_CHARS)

    def get_observation(self) -> Any:
        # Use the renderer for observation
        return self.renderer.get_observation(self.cube_manager.get_state())

    def _reset_camera_params(self):
        # Sequence env does not use camera/view
        self.camera_params = None

    def _update_camera_params(self, action_name: str):
        # Sequence env does not use camera/view
        pass

def make_sequence_env(max_steps: int = 1000, scramble_moves: int = 20, reward_function_config: Dict[str, Any] = {}):
    cube_manager = CubeSimulator()
    action_space_config = CUBE_ONLY_ACTION_SPACE
    reward_function = get_reward_function(**reward_function_config)
    return SequenceCubeEnv(cube_manager, action_space_config, max_steps, scramble_moves, reward_function)