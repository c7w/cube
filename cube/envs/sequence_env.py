import numpy as np
from typing import Any, Dict, Optional
from gymnasium import spaces
from cube.core.base_env import BaseCubeEnvironment, BaseRenderer
from cube.core.base_simulator import State, CubeSimulator as StateManager
from cube.core.base_action_space import ActionSpace
from cube.core.base_reward import RewardFunction, get_reward_function

def make_sequence_env(max_steps: int = 1000, reward_function_config: Dict[str, Any] = {"type": "dummy"}):
    state_manager = StateManager()
    action_space_config = ActionSpace()
    reward_function = get_reward_function(**reward_function_config)
    renderer = SequenceRenderer()
    return SequenceCubeEnvironment(state_manager, action_space_config, reward_function, renderer, max_steps)

class SequenceRenderer(BaseRenderer):
    """
    Renderer for sequence-based cube representation.
    Handles conversion between state arrays and token sequences.
    """
    def __init__(self):
        super().__init__()
    
    def get_observation(self, state: State, viewpoint: Any) -> str:
        return "".join([self.STATE_TO_COLOR[state[i]] for i in range(54)])

# --- SequenceEnv implementation ---
class SequenceCubeEnvironment(BaseCubeEnvironment):
    """
    Sequence-based CubeBench environment.
    Only implements the required abstract methods with minimal logic.
    """
    def __init__(self, 
                 state_manager: StateManager, 
                 action_space_config: ActionSpace, 
                 reward_function: RewardFunction, 
                 renderer: BaseRenderer, 
                 max_steps: int = 1000):
        # Initialize renderer first
        # Call parent constructor with dependencies
        super().__init__(
            state_manager=state_manager,
            action_space_config=action_space_config,
            reward_function=reward_function,
            renderer=renderer,
            max_steps=max_steps
        )
        
        # Now reset after everything is initialized
        self.reset()
    
    def _setup_observation_space(self):
        # Observation is a 54-char string with color chars representing cube faces
        self.observation_space = spaces.Text(54, charset=list(self.renderer.STATE_TO_COLOR.values()))

    def get_observation(self) -> Any:
        # Use the renderer for observation
        return self.renderer.get_observation(self.state_manager.get_state(), self.viewpoint)

    def _reset_viewpoint(self):
        self.viewpoint = None
    
    def _update_viewpoint(self, action_name: str):
        pass