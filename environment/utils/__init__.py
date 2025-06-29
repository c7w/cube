"""
Utility functions and classes for CubeBench.
"""

from .cube_simulator import CubeSimulator
from .state_utils import (
    serialize_state as state_to_color_string,
    deserialize_state as color_string_to_state,
    validate_state_string as validate_color_string,
    COLOR_TO_LETTER
)

# Get COLOR_CHARS from COLOR_TO_LETTER mapping
COLOR_CHARS = list(COLOR_TO_LETTER.values())

__all__ = [
    'CubeSimulator',
    'COLOR_CHARS',
    'state_to_color_string',
    'color_string_to_state',
    'validate_color_string',
]