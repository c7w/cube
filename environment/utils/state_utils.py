"""
State serialization and deserialization utilities.
"""

import numpy as np
from typing import Dict


# Color ID to letter mapping for better readability
COLOR_TO_LETTER: Dict[int, str] = {
    0: 'W',  # White
    1: 'Y',  # Yellow
    2: 'R',  # Red
    3: 'O',  # Orange
    4: 'B',  # Blue
    5: 'G'   # Green
}

# Reverse mapping
LETTER_TO_COLOR: Dict[str, int] = {v: k for k, v in COLOR_TO_LETTER.items()}


def serialize_state(state: np.ndarray) -> str:
    """
    Convert 54-element state array to string representation.
    
    Args:
        state: 54-element numpy array with color IDs (0-5)
        
    Returns:
        54-character string like "WWWYYYRRR..."
    """
    if len(state) != 54:
        raise ValueError(f"State must have 54 elements, got {len(state)}")
    
    return ''.join(COLOR_TO_LETTER[int(color_id)] for color_id in state)


def deserialize_state(state_str: str) -> np.ndarray:
    """
    Convert string representation back to 54-element state array.
    
    Args:
        state_str: 54-character string like "WWWYYYRRR..."
        
    Returns:
        54-element numpy array with color IDs (0-5)
    """
    if len(state_str) != 54:
        raise ValueError(f"State string must have 54 characters, got {len(state_str)}")
    
    try:
        state = np.array([LETTER_TO_COLOR[char] for char in state_str], dtype=int)
        return state
    except KeyError as e:
        raise ValueError(f"Invalid character in state string: {e}")


def validate_state_string(state_str: str) -> bool:
    """
    Validate if a state string is properly formatted.
    
    Args:
        state_str: String to validate
        
    Returns:
        True if valid, False otherwise
    """
    if len(state_str) != 54:
        return False
    
    return all(char in LETTER_TO_COLOR for char in state_str)


def get_color_counts(state_str: str) -> Dict[str, int]:
    """
    Count occurrences of each color in the state string.
    
    Args:
        state_str: 54-character state string
        
    Returns:
        Dictionary with color counts
    """
    if not validate_state_string(state_str):
        raise ValueError("Invalid state string")
    
    counts = {color: 0 for color in COLOR_TO_LETTER.values()}
    for char in state_str:
        counts[char] += 1
    
    return counts 