"""
Face view renderer for 3D cube visualization from face center perspectives.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use EGL for headless rendering

try:
    from OpenGL.GL import *
    from OpenGL.arrays import vbo
    import OpenGL.GL.shaders as shaders
    try:
        from OpenGL.EGL import *
        EGL_AVAILABLE = True
    except (ImportError, AttributeError):
        EGL_AVAILABLE = False
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    EGL_AVAILABLE = False
    print("Warning: OpenGL not available, falling back to matplotlib")

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

from ..utils.view_utils import ViewType, get_face_views
from ..renderer import CubeRenderer


class FaceViewRenderer:
    """
    Renderer for face-perspective 3D cube visualization.
    Supports headless rendering using OpenGL EGL or matplotlib fallback.
    """
    
    def __init__(self, image_size: int = 256, show_face_labels: bool = False):
        """
        Initialize face view renderer.
        
        Args:
            image_size: Output image size (width=height)
            show_face_labels: Whether to show face labels on rendered cube
        """
        self.image_size = image_size
        self.show_face_labels = show_face_labels
        self.face_views = get_face_views()
        self.use_opengl = OPENGL_AVAILABLE
        
        # Initialize base renderer for color mapping
        self.base_renderer = CubeRenderer()
        
        # EGL context for headless rendering
        self.egl_display = None
        self.egl_context = None
        
        if self.use_opengl and EGL_AVAILABLE:
            self._init_opengl()
        else:
            self.use_opengl = False
            print("Using matplotlib fallback for rendering")
    
    def _init_opengl(self):
        """Initialize OpenGL EGL context for headless rendering."""
        try:
            # Initialize EGL
            self.egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
            eglInitialize(self.egl_display, None, None)
            
            # Choose config
            config_attribs = [
                EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                EGL_BLUE_SIZE, 8,
                EGL_GREEN_SIZE, 8,
                EGL_RED_SIZE, 8,
                EGL_DEPTH_SIZE, 24,
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_NONE
            ]
            
            configs = eglChooseConfig(self.egl_display, config_attribs, 1)
            config = configs[0]
            
            # Create context
            context_attribs = [EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE]
            self.egl_context = eglCreateContext(self.egl_display, config, EGL_NO_CONTEXT, context_attribs)
            
            # Create pbuffer surface
            pbuffer_attribs = [
                EGL_WIDTH, self.image_size,
                EGL_HEIGHT, self.image_size,
                EGL_NONE
            ]
            self.egl_surface = eglCreatePbufferSurface(self.egl_display, config, pbuffer_attribs)
            
            # Make context current
            eglMakeCurrent(self.egl_display, self.egl_surface, self.egl_surface, self.egl_context)
            
        except Exception as e:
            print(f"Failed to initialize OpenGL EGL: {e}")
            self.use_opengl = False
    
    def render_view(self, state: np.ndarray, view_id: int) -> np.ndarray:
        """
        Render cube from a specific face view.
        
        Args:
            state: 54-element cube state array
            view_id: View ID (0-23)
            
        Returns:
            RGB image as numpy array (256x256x3)
        """
        if view_id < 0 or view_id >= 24:
            raise ValueError(f"View ID must be 0-23, got {view_id}")
        
        view_info = self.face_views[view_id]
        
        if self.use_opengl:
            return self._render_opengl(state, view_info)
        else:
            return self._render_matplotlib(state, view_info)
    
    def _render_opengl(self, state: np.ndarray, view_info: Dict) -> np.ndarray:
        """Render using OpenGL."""
        # Set up viewport
        glViewport(0, 0, self.image_size, self.image_size)
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        
        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        # Perspective projection
        fov = 45.0
        aspect = 1.0
        near = 0.1
        far = 100.0
        
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        projection = np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        glLoadMatrixf(projection.T)
        
        # Set up modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Position camera based on view info
        camera_pos = view_info['camera_pos']
        look_at = view_info['look_at']
        up_vector = view_info['up_vector']
        
        # Simple lookAt implementation
        forward = np.array(look_at) - np.array(camera_pos)
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Create view matrix
        view_matrix = np.eye(4)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up
        view_matrix[2, :3] = -forward
        view_matrix[:3, 3] = -np.array(camera_pos)
        
        glLoadMatrixf(view_matrix.T)
        
        # Apply face-specific rotation
        rotation_angle = view_info['rotation_angle']
        face_normal = view_info['face_normal']
        glRotatef(rotation_angle, *face_normal)
        
        # Render cube
        self._draw_cube_opengl(state)
        
        # Read pixels
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, self.image_size, self.image_size, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.image_size, self.image_size, 3)
        
        # Flip vertically (OpenGL coordinates)
        image = np.flipud(image)
        
        return image
    
    def _render_matplotlib(self, state: np.ndarray, view_info: Dict) -> np.ndarray:
        """Fallback rendering using matplotlib."""
        fig, ax = plt.subplots(1, 1, figsize=(self.image_size/100, self.image_size/100), dpi=100)
        
        # Simple 2D projection of 3D cube based on face view
        # This is a simplified fallback - not true 3D rendering
        
        # Get visible faces based on view direction
        visible_faces = self._get_visible_faces(view_info)
        
        # Draw visible faces with face-specific rotation
        face_size = 0.8
        face_spacing = 0.1
        rotation_angle = view_info['rotation_angle']
        
        for i, face_name in enumerate(visible_faces):
            face_data = self._get_face_data(state, face_name)
            x_offset = (i % 2) * (face_size + face_spacing) - face_size/2
            y_offset = (i // 2) * (face_size + face_spacing) - face_size/2
            
            self._draw_face_matplotlib(ax, face_data, x_offset, y_offset, face_size, 
                                     face_name, rotation_angle)
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Convert to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, :3]  # Remove alpha channel
        
        plt.close(fig)
        
        return buf
    
    def _draw_cube_opengl(self, state: np.ndarray):
        """Draw cube using OpenGL primitives."""
        # Define cube vertices
        vertices = np.array([
            # Front face
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
            # Back face
            [-1, -1, -1], [-1,  1, -1], [ 1,  1, -1], [ 1, -1, -1],
            # Top face
            [-1,  1, -1], [-1,  1,  1], [ 1,  1,  1], [ 1,  1, -1],
            # Bottom face
            [-1, -1, -1], [ 1, -1, -1], [ 1, -1,  1], [-1, -1,  1],
            # Right face
            [ 1, -1, -1], [ 1,  1, -1], [ 1,  1,  1], [ 1, -1,  1],
            # Left face
            [-1, -1, -1], [-1, -1,  1], [-1,  1,  1], [-1,  1, -1]
        ], dtype=np.float32)
        
        # Define face indices
        faces = [
            [0, 1, 2, 3],    # Front
            [4, 5, 6, 7],    # Back
            [8, 9, 10, 11],  # Top
            [12, 13, 14, 15], # Bottom
            [16, 17, 18, 19], # Right
            [20, 21, 22, 23]  # Left
        ]
        
        face_names = ['FRONT', 'BACK', 'UP', 'DOWN', 'RIGHT', 'LEFT']
        
        # Draw each face
        for face_idx, face_indices in enumerate(faces):
            face_name = face_names[face_idx]
            face_data = self._get_face_data(state, face_name)
            
            # Draw 9 squares for this face
            for i in range(3):
                for j in range(3):
                    color_id = face_data[i, j]
                    color = np.array(self.base_renderer.COLOR_MAP[color_id]) / 255.0
                    
                    glColor3fv(color)
                    
                    # Calculate square position within the face
                    # This is simplified - proper implementation would need texture mapping
                    glBegin(GL_QUADS)
                    for vertex_idx in face_indices:
                        glVertex3fv(vertices[vertex_idx])
                    glEnd()
    
    def _get_visible_faces(self, view_info: Dict) -> List[str]:
        """Determine which faces are visible from the current face view."""
        # For face views, the primary face being viewed should be most prominent
        face_id = view_info['face_id']
        face_names = ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN']
        primary_face = face_names[face_id]
        
        # Return primary face and two adjacent faces
        if primary_face == 'FRONT':
            return ['FRONT', 'RIGHT', 'UP']
        elif primary_face == 'BACK':
            return ['BACK', 'LEFT', 'UP']
        elif primary_face == 'LEFT':
            return ['LEFT', 'FRONT', 'UP']
        elif primary_face == 'RIGHT':
            return ['RIGHT', 'BACK', 'UP']
        elif primary_face == 'UP':
            return ['UP', 'FRONT', 'RIGHT']
        elif primary_face == 'DOWN':
            return ['DOWN', 'FRONT', 'RIGHT']
        else:
            return ['FRONT', 'RIGHT', 'UP']
    
    def _get_face_data(self, state: np.ndarray, face_name: str) -> np.ndarray:
        """Get face data from cube state."""
        return self.base_renderer._get_face_data(state, face_name)
    
    def _draw_face_matplotlib(self, ax, face_data: np.ndarray, x_offset: float, y_offset: float, 
                            face_size: float, face_name: str, rotation_angle: float = 0):
        """Draw a single face using matplotlib with rotation."""
        square_size = face_size / 3
        
        # Apply rotation to the face data if needed
        if rotation_angle != 0:
            # Rotate face data by the specified angle
            rotations = int(rotation_angle // 90) % 4
            for _ in range(rotations):
                face_data = np.rot90(face_data)
        
        for i in range(3):
            for j in range(3):
                color_id = face_data[i, j]
                color = np.array(self.base_renderer.COLOR_MAP[color_id]) / 255.0
                
                x = x_offset + j * square_size
                y = y_offset + (2-i) * square_size  # Flip Y coordinate
                
                rect = Rectangle((x, y), square_size, square_size, 
                               facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
        
        # Add face label if requested
        if self.show_face_labels:
            ax.text(x_offset + face_size/2, y_offset + face_size/2, face_name,
                   ha='center', va='center', fontsize=8, color='white')
    
    def get_observation(self, state: np.ndarray, view_id: int) -> np.ndarray:
        """
        Get observation for the environment.
        
        Args:
            state: 54-element cube state array
            view_id: Current view ID
            
        Returns:
            RGB image observation
        """
        return self.render_view(state, view_id)
    
    def get_all_views(self) -> List[Dict]:
        """Get information about all available views."""
        return self.face_views.copy()
    
    def cleanup(self):
        """Clean up OpenGL resources."""
        if self.use_opengl and self.egl_context:
            eglDestroyContext(self.egl_display, self.egl_context)
            eglTerminate(self.egl_display) 