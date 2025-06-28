"""
CubeBench: A framework for evaluating and training Vision Language Models (VLM) 
on embodied reasoning tasks using an interactive Rubik's cube environment.
"""

from .base_env import CubeBenchEnv
from .cube_simulator import CubeSimulator
from .action_space import ActionSpace, ActionType, CubeAction, ViewAction, SpecialAction
from .reward import RewardFunction, SparseReward, DenseReward, HybridReward, RewardType
from .renderer import CubeRenderer, RenderMode

# New specialized environments
from .envs.sequence_env import SequenceEnv
from .envs.vertex_view_env import VertexViewEnv
from .envs.face_view_env import FaceViewEnv

# Utility modules
from .utils.state_utils import serialize_state, deserialize_state
from .utils.view_utils import ViewType, get_vertex_views, get_face_views

# Specialized renderers
from .renderers.sequence_renderer import SequenceRenderer
from .renderers.vertex_renderer import VertexViewRenderer
from .renderers.face_renderer import FaceViewRenderer

__version__ = "1.0.0"

__all__ = [
    # Core components
    'CubeBenchEnv',
    'CubeSimulator', 
    'ActionSpace',
    'ActionType',
    'CubeAction',
    'ViewAction', 
    'SpecialAction',
    'RewardFunction',
    'SparseReward',
    'DenseReward', 
    'HybridReward',
    'RewardType',
    'CubeRenderer',
    'RenderMode',
    
    # New environments
    'SequenceEnv',
    'VertexViewEnv',
    'FaceViewEnv',
    
    # Utilities
    'serialize_state',
    'deserialize_state',
    'ViewType',
    'get_vertex_views',
    'get_face_views',
    
    # Specialized renderers
    'SequenceRenderer',
    'VertexViewRenderer',
    'FaceViewRenderer',
] 