"""
CubeBench Environment Package

A comprehensive framework for evaluating and training VLM embodied reasoning capabilities.
"""

from .base_env import CubeBenchEnv
from .cube_simulator import CubeSimulator, Color
from .renderer import CubeRenderer, RenderMode
from .action_space import ActionSpace
from .reward import RewardFunction

__version__ = "0.1.0"
__all__ = [
    'CubeBenchEnv',
    'CubeSimulator', 
    'Color',
    'CubeRenderer',
    'RenderMode', 
    'ActionSpace',
    'RewardFunction'
] 