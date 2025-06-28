"""
Unit tests for SequenceRenderer.
"""

import numpy as np
import pytest
from environment.renderers.sequence_renderer import SequenceRenderer
from environment.utils.state_utils import serialize_state, deserialize_state


class TestSequenceRenderer:
    """Test cases for SequenceRenderer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = SequenceRenderer()
        
        # Create a solved cube state
        self.solved_state = np.array([
            # FRONT (RED)
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            # BACK (ORANGE)  
            3, 3, 3, 3, 3, 3, 3, 3, 3,
            # LEFT (BLUE)
            4, 4, 4, 4, 4, 4, 4, 4, 4,
            # RIGHT (GREEN)
            5, 5, 5, 5, 5, 5, 5, 5, 5,
            # UP (YELLOW)
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            # DOWN (WHITE)
            0, 0, 0, 0, 0, 0, 0, 0, 0
        ], dtype=int)
        
        # Expected token sequence for solved state
        self.expected_sequence = "RRRRRRRRROOOOOOOOOBBBBBBBBBGGGGGGGGGYYYYYYYYYWWWWWWWWW"
    
    def test_init(self):
        """Test renderer initialization."""
        renderer = SequenceRenderer()
        assert renderer is not None
    
    def test_render_state_to_sequence(self):
        """Test converting state array to token sequence."""
        sequence = self.renderer.render_state_to_sequence(self.solved_state)
        assert isinstance(sequence, str)
        assert len(sequence) == 54
        assert sequence == self.expected_sequence
    
    def test_render_sequence_to_state(self):
        """Test converting token sequence back to state array."""
        state = self.renderer.render_sequence_to_state(self.expected_sequence)
        assert isinstance(state, np.ndarray)
        assert state.shape == (54,)
        assert state.dtype == int
        np.testing.assert_array_equal(state, self.solved_state)
    
    def test_render_roundtrip(self):
        """Test that state -> sequence -> state is lossless."""
        # Test with solved state
        sequence = self.renderer.render_state_to_sequence(self.solved_state)
        recovered_state = self.renderer.render_sequence_to_state(sequence)
        np.testing.assert_array_equal(recovered_state, self.solved_state)
        
        # Test with random state
        random_state = np.random.randint(0, 6, 54)
        sequence = self.renderer.render_state_to_sequence(random_state)
        recovered_state = self.renderer.render_sequence_to_state(sequence)
        np.testing.assert_array_equal(recovered_state, random_state)
    
    def test_render_main_function(self):
        """Test main render function."""
        result = self.renderer.render(self.solved_state)
        
        assert isinstance(result, dict)
        assert 'sequence' in result
        assert 'length' in result
        assert 'valid' in result
        
        assert result['sequence'] == self.expected_sequence
        assert result['length'] == 54
        assert result['valid'] == True
    
    def test_get_observation(self):
        """Test get_observation method."""
        observation = self.renderer.get_observation(self.solved_state)
        assert isinstance(observation, str)
        assert len(observation) == 54
        assert observation == self.expected_sequence
    
    def test_validate_observation(self):
        """Test observation validation."""
        # Valid observation
        assert self.renderer.validate_observation(self.expected_sequence) == True
        
        # Invalid observations
        assert self.renderer.validate_observation("") == False
        assert self.renderer.validate_observation("INVALID") == False
        assert self.renderer.validate_observation("R" * 53) == False  # Too short
        assert self.renderer.validate_observation("R" * 55) == False  # Too long
        assert self.renderer.validate_observation("X" * 54) == False  # Invalid characters
    
    def test_invalid_state_input(self):
        """Test handling of invalid state inputs."""
        # Wrong shape
        with pytest.raises((ValueError, IndexError)):
            self.renderer.render_state_to_sequence(np.array([1, 2, 3]))
        
        # Wrong dtype (should still work but convert)
        float_state = self.solved_state.astype(float)
        sequence = self.renderer.render_state_to_sequence(float_state)
        assert isinstance(sequence, str)
        assert len(sequence) == 54
    
    def test_invalid_sequence_input(self):
        """Test handling of invalid sequence inputs."""
        # Invalid length
        with pytest.raises(ValueError):
            self.renderer.render_sequence_to_state("RRRR")
        
        # Invalid characters
        with pytest.raises(ValueError):
            self.renderer.render_sequence_to_state("X" * 54)
    
    def test_different_cube_states(self):
        """Test rendering different cube configurations."""
        # All same color (impossible but should work)
        uniform_state = np.full(54, 2, dtype=int)  # All red
        sequence = self.renderer.render_state_to_sequence(uniform_state)
        assert sequence == "R" * 54
        
        # Alternating pattern
        alternating_state = np.array([i % 6 for i in range(54)], dtype=int)
        sequence = self.renderer.render_state_to_sequence(alternating_state)
        assert len(sequence) == 54
        
        # Verify roundtrip
        recovered = self.renderer.render_sequence_to_state(sequence)
        np.testing.assert_array_equal(recovered, alternating_state)


if __name__ == "__main__":
    pytest.main([__file__]) 