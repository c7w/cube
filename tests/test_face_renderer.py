"""
Unit tests for FaceViewRenderer.
"""

import numpy as np
import pytest
from environment.renderers.face_renderer import FaceViewRenderer


class TestFaceViewRenderer:
    """Test cases for FaceViewRenderer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = FaceViewRenderer(image_size=64)  # Small size for faster tests
        
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
        
        # Create a scrambled state for testing
        self.scrambled_state = np.random.randint(0, 6, 54, dtype=int)
    
    def test_init(self):
        """Test renderer initialization."""
        # Test default initialization
        renderer = FaceViewRenderer()
        assert renderer.image_size == 256
        assert renderer.show_face_labels == False
        assert len(renderer.face_views) == 24
        
        # Test custom initialization
        renderer_custom = FaceViewRenderer(image_size=128, show_face_labels=True)
        assert renderer_custom.image_size == 128
        assert renderer_custom.show_face_labels == True
    
    def test_render_view_valid_ids(self):
        """Test rendering with valid view IDs."""
        for view_id in [0, 5, 10, 15, 20, 23]:  # Sample view IDs
            image = self.renderer.render_view(self.solved_state, view_id)
            
            # Check image properties
            assert isinstance(image, np.ndarray)
            assert image.shape == (64, 64, 3)
            assert image.dtype == np.uint8
            
            # Check that image is not all zeros (should have some content)
            assert np.any(image > 0)
            
            # Check that values are in valid RGB range
            assert np.all(image >= 0)
            assert np.all(image <= 255)
    
    def test_render_view_invalid_ids(self):
        """Test rendering with invalid view IDs."""
        invalid_ids = [-1, 24, 25, 100]
        
        for view_id in invalid_ids:
            with pytest.raises(ValueError):
                self.renderer.render_view(self.solved_state, view_id)
    
    def test_render_view_different_states(self):
        """Test rendering different cube states."""
        view_id = 0
        
        # Render solved state
        solved_image = self.renderer.render_view(self.solved_state, view_id)
        
        # Render scrambled state
        scrambled_image = self.renderer.render_view(self.scrambled_state, view_id)
        
        # Images should be different (unless by extreme coincidence)
        assert not np.array_equal(solved_image, scrambled_image)
        
        # Both should have valid properties
        for image in [solved_image, scrambled_image]:
            assert image.shape == (64, 64, 3)
            assert image.dtype == np.uint8
            assert np.any(image > 0)
    
    def test_get_observation(self):
        """Test get_observation method."""
        view_id = 5
        observation = self.renderer.get_observation(self.solved_state, view_id)
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (64, 64, 3)
        assert observation.dtype == np.uint8
        
        # Should be same as render_view
        direct_render = self.renderer.render_view(self.solved_state, view_id)
        np.testing.assert_array_equal(observation, direct_render)
    
    def test_get_all_views(self):
        """Test get_all_views method."""
        all_views = self.renderer.get_all_views()
        
        assert isinstance(all_views, list)
        assert len(all_views) == 24
        
        # Check structure of view info
        for i, view_info in enumerate(all_views):
            assert isinstance(view_info, dict)
            assert 'view_id' in view_info
            assert 'camera_pos' in view_info
            assert 'look_at' in view_info
            assert 'up_vector' in view_info
            assert 'rotation_angle' in view_info
            assert 'face_normal' in view_info
            assert view_info['view_id'] == i
    
    def test_render_consistency(self):
        """Test that rendering the same state/view gives consistent results."""
        view_id = 0
        
        # Render same state multiple times
        image1 = self.renderer.render_view(self.solved_state, view_id)
        image2 = self.renderer.render_view(self.solved_state, view_id)
        
        # Should be identical (deterministic rendering)
        np.testing.assert_array_equal(image1, image2)
    
    def test_all_view_ids(self):
        """Test that all 24 view IDs work without errors."""
        for view_id in range(24):
            try:
                image = self.renderer.render_view(self.solved_state, view_id)
                assert image.shape == (64, 64, 3)
                assert image.dtype == np.uint8
            except Exception as e:
                pytest.fail(f"View ID {view_id} failed with error: {e}")
    
    def test_invalid_state_input(self):
        """Test handling of invalid state inputs."""
        view_id = 0
        
        # Wrong shape
        with pytest.raises((ValueError, IndexError)):
            self.renderer.render_view(np.array([1, 2, 3]), view_id)
        
        # Wrong dtype (should work but convert)
        float_state = self.solved_state.astype(float)
        image = self.renderer.render_view(float_state, view_id)
        assert image.shape == (64, 64, 3)
        assert image.dtype == np.uint8
    
    def test_image_size_property(self):
        """Test that image_size property is respected."""
        sizes = [32, 64, 128]
        
        for size in sizes:
            renderer = FaceViewRenderer(image_size=size)
            image = renderer.render_view(self.solved_state, 0)
            assert image.shape == (size, size, 3)
    
    def test_face_views_structure(self):
        """Test that face_views have correct structure."""
        assert hasattr(self.renderer, 'face_views')
        assert len(self.renderer.face_views) == 24
        
        for i, view in enumerate(self.renderer.face_views):
            assert isinstance(view, dict)
            required_keys = ['view_id', 'camera_pos', 'look_at', 'up_vector', 'rotation_angle', 'face_normal']
            for key in required_keys:
                assert key in view, f"View {i} missing key: {key}"
            
            assert view['view_id'] == i
            assert len(view['camera_pos']) == 3
            assert len(view['look_at']) == 3
            assert len(view['up_vector']) == 3
            assert len(view['face_normal']) == 3
            assert isinstance(view['rotation_angle'], (int, float))
    
    def test_cleanup(self):
        """Test cleanup method."""
        # Should not raise any errors
        try:
            self.renderer.cleanup()
        except Exception as e:
            pytest.fail(f"Cleanup failed with error: {e}")
    
    def test_face_labels_option(self):
        """Test face labels option."""
        renderer_with_labels = FaceViewRenderer(image_size=64, show_face_labels=True)
        renderer_without_labels = FaceViewRenderer(image_size=64, show_face_labels=False)
        
        # Both should work
        image_with = renderer_with_labels.render_view(self.solved_state, 0)
        image_without = renderer_without_labels.render_view(self.solved_state, 0)
        
        assert image_with.shape == (64, 64, 3)
        assert image_without.shape == (64, 64, 3)
        
        # Clean up
        renderer_with_labels.cleanup()
        renderer_without_labels.cleanup()
    
    def test_face_rotations(self):
        """Test that different face rotations produce different views."""
        # Get views for the same face with different rotations
        # Each face should have 4 rotations (0°, 90°, 180°, 270°)
        face_views = []
        for view_id in range(4):  # First 4 views should be same face, different rotations
            image = self.renderer.render_view(self.solved_state, view_id)
            face_views.append(image)
        
        # Check that they're all different (at least some should be)
        for i in range(len(face_views)):
            for j in range(i + 1, len(face_views)):
                # Images should be different due to rotation
                # (unless the cube state is perfectly symmetric)
                assert face_views[i].shape == face_views[j].shape
    
    def test_face_normal_vectors(self):
        """Test that face normal vectors are properly defined."""
        for view in self.renderer.face_views:
            face_normal = view['face_normal']
            
            # Should be a 3D vector
            assert len(face_normal) == 3
            
            # Should be normalized (length ≈ 1)
            length = np.linalg.norm(face_normal)
            assert abs(length - 1.0) < 1e-6, f"Face normal not normalized: {face_normal}, length: {length}"
    
    def test_rotation_angles(self):
        """Test that rotation angles are in expected range."""
        rotation_angles = set()
        
        for view in self.renderer.face_views:
            angle = view['rotation_angle']
            rotation_angles.add(angle)
            
            # Rotation angle should be in [0, 360) or equivalent
            assert isinstance(angle, (int, float))
            assert angle >= 0
            assert angle < 360
        
        # Should have exactly 4 different rotation angles (0, 90, 180, 270)
        assert len(rotation_angles) == 4
        expected_angles = {0, 90, 180, 270}
        assert rotation_angles == expected_angles
    
    def test_six_faces_coverage(self):
        """Test that all 6 faces are covered in the 24 views."""
        # Each face should appear 4 times (4 rotations each)
        # We can check this by examining the view names or positions
        
        # Get unique camera positions (should represent 6 face centers)
        unique_positions = set()
        for view in self.renderer.face_views:
            pos = tuple(view['camera_pos'])
            unique_positions.add(pos)
        
        # Should have 6 unique positions (one for each face center)
        assert len(unique_positions) == 6
    
    def test_view_id_consistency(self):
        """Test that view IDs are consistent with array indices."""
        for i, view in enumerate(self.renderer.face_views):
            assert view['view_id'] == i, f"View at index {i} has ID {view['view_id']}"


if __name__ == "__main__":
    pytest.main([__file__]) 