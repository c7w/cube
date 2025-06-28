"""
Sequence renderer for token-based cube representation.
"""

import numpy as np
from typing import Dict, Any, Optional
from ..utils.state_utils import serialize_state, deserialize_state, validate_state_string


class SequenceRenderer:
    """
    Renderer for sequence-based cube representation.
    Handles conversion between state arrays and token sequences.
    """
    
    def __init__(self):
        """Initialize sequence renderer."""
        pass
    
    def render_state_to_sequence(self, state: np.ndarray) -> str:
        """
        Render cube state as token sequence.
        
        Args:
            state: 54-element numpy array with color IDs
            
        Returns:
            54-character token sequence
        """
        return serialize_state(state)
    
    def render_sequence_to_state(self, sequence: str) -> np.ndarray:
        """
        Convert token sequence back to state array.
        
        Args:
            sequence: 54-character token sequence
            
        Returns:
            54-element numpy array with color IDs
        """
        return deserialize_state(sequence)
    
    def render(self, state: np.ndarray) -> Dict[str, Any]:
        """
        Main render function for compatibility with other renderers.
        
        Args:
            state: 54-element numpy array with color IDs
            
        Returns:
            Dictionary with sequence representation
        """
        sequence = self.render_state_to_sequence(state)
        
        return {
            'sequence': sequence,
            'length': len(sequence),
            'valid': validate_state_string(sequence)
        }
    
    def get_observation(self, state: np.ndarray) -> str:
        """
        Get observation for the environment.
        
        Args:
            state: 54-element numpy array with color IDs
            
        Returns:
            Token sequence observation
        """
        return self.render_state_to_sequence(state)
    
    def validate_observation(self, observation: str) -> bool:
        """
        Validate if an observation is properly formatted.
        
        Args:
            observation: Token sequence to validate
            
        Returns:
            True if valid, False otherwise
        """
        return validate_state_string(observation) 