"""
Renderer for the CubeBench environment.
Supports both image and symbolic state output modes.
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors


class RenderMode(Enum):
    """Rendering modes"""
    IMAGE = "image"           # Visual representation
    SYMBOLIC = "symbolic"     # Symbolic state representation
    BOTH = "both"            # Both modes


class CubeRenderer:
    """
    Renderer for the 3x3x3 Rubik's cube.
    
    Supports multiple rendering modes:
    - Image mode: Visual representation of the cube
    - Symbolic mode: Numerical state representation
    """
    
    # Color mapping for visualization
    COLOR_MAP = {
        0: (255, 255, 255),  # White
        1: (0, 255, 255),    # Yellow
        2: (0, 0, 255),      # Red
        3: (0, 165, 255),    # Orange
        4: (255, 0, 0),      # Blue
        5: (0, 255, 0)       # Green
    }
    
    # Color names for symbolic representation
    COLOR_NAMES = {
        0: "WHITE",
        1: "YELLOW", 
        2: "RED",
        3: "ORANGE",
        4: "BLUE",
        5: "GREEN"
    }
    
    # Face names for symbolic representation
    FACE_NAMES = ["FRONT", "BACK", "LEFT", "RIGHT", "UP", "DOWN"]
    
    # Face starting positions in 54-element state
    FACE_POSITIONS = {
        'FRONT': 0,   # 0-8
        'BACK': 9,    # 9-17
        'LEFT': 18,   # 18-26
        'RIGHT': 27,  # 27-35
        'UP': 36,     # 36-44
        'DOWN': 45    # 45-53
    }
    
    def __init__(self, mode: RenderMode = RenderMode.IMAGE, 
                 image_size: int = 400, 
                 show_face_labels: bool = True,
                 show_numbers: bool = False):
        """
        Initialize renderer.
        
        Args:
            mode: Rendering mode
            image_size: Size of output image (width=height)
            show_face_labels: Whether to show face labels in image mode
            show_numbers: Whether to show position numbers (0-53) on each square
        """
        self.mode = mode
        self.image_size = image_size
        self.show_face_labels = show_face_labels
        self.show_numbers = show_numbers
        
        # Calculate face size based on image size
        self.face_size = image_size // 4  # 4 faces visible in cross layout
        
        # Face positions in cross layout (relative to center)
        # These coordinates map to (grid_x, grid_y) for drawing
        # The cross layout is:
        #      UP
        #   LEFT FRONT RIGHT BACK
        #      DOWN
        # Consistent with your provided image_b816da.png
        self.face_positions = {
            'UP': (1, 0),       # UP face at (1,0) of the 4x4 grid
            'LEFT': (0, 1),     # LEFT face at (0,1)
            'FRONT': (1, 1),    # FRONT face at (1,1)
            'RIGHT': (2, 1),    # RIGHT face at (2,1)
            'DOWN': (1, 2),     # DOWN face at (1,2)
            'BACK': (1, 3)      # BACK face at (1,3) - consistent with your image layout
        }
    
    def render(self, cube_state: np.ndarray, 
               view_angle: Optional[Tuple[float, float, float]] = None) -> Dict[str, Any]:
        """
        Render the cube state.
        
        Args:
            cube_state: 54-element array or 6x9 array representing cube state
            view_angle: Optional view angle (x, y, z) in degrees
            
        Returns:
            Dictionary containing rendered output based on mode
        """
        # Convert to 54-element format if needed
        if cube_state.shape == (6, 9):
            state_54 = self._convert_6x9_to_54(cube_state)
        elif len(cube_state) == 54:
            state_54 = cube_state
        else:
            raise ValueError(f"Invalid cube state shape: {cube_state.shape}")
        
        result = {}
        
        if self.mode in [RenderMode.IMAGE, RenderMode.BOTH]:
            result['image'] = self._render_image(state_54, view_angle)
        
        if self.mode in [RenderMode.SYMBOLIC, RenderMode.BOTH]:
            result['symbolic'] = self._render_symbolic(state_54)
        
        return result
    
    def _convert_6x9_to_54(self, faces_6x9: np.ndarray) -> np.ndarray:
        """Convert 6x9 face format to 54-element format for compatibility"""
        state_54 = np.zeros(54, dtype=int)
        face_order = ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN']
        
        for i, face_name in enumerate(face_order):
            start_pos = self.FACE_POSITIONS[face_name]
            state_54[start_pos:start_pos+9] = faces_6x9[i]
        
        return state_54
    
    def _render_image(self, cube_state: np.ndarray, 
                     view_angle: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Render cube as an image.
        
        Args:
            cube_state: 54-element array representing cube state
            view_angle: Optional view angle (x, y, z) in degrees
            
        Returns:
            RGB image as numpy array
        """
        # Create blank image
        # The total grid size is 4x4 faces, so total width/height is 4 * face_size
        total_image_size = self.face_size * 4
        img = np.ones((total_image_size, total_image_size, 3), dtype=np.uint8) * 50
        
        # Render each face
        # Explicitly iterate for drawing order to match the cross layout visually
        for face_name in ['UP', 'LEFT', 'FRONT', 'RIGHT', 'DOWN', 'BACK']: 
            face_data = self._get_face_data(cube_state, face_name)
            self._draw_face(img, face_data, face_name)
        
        # Apply view transformation if specified
        if view_angle is not None:
            img = self._apply_view_transform(img, view_angle)
        
        return img
    
    def _get_face_data(self, cube_state: np.ndarray, face_name: str) -> np.ndarray:
        """Get face data as 3x3 array from 54-element state"""
        start_pos = self.FACE_POSITIONS[face_name]
        face_data = cube_state[start_pos:start_pos+9]
        return face_data.reshape(3, 3)
    
    def _draw_face(self, img: np.ndarray, face_data: np.ndarray, face_name: str):
        """Draw a single face on the image"""
        # Get grid coordinates for this face
        grid_x, grid_y = self.face_positions[face_name]
        
        # Calculate pixel starting position for this face
        start_x_pixel = grid_x * self.face_size
        start_y_pixel = grid_y * self.face_size
        
        # Draw each square in the face
        square_size = self.face_size // 3
        
        for i in range(3): # row in face
            for j in range(3): # column in face
                color_value = face_data[i, j]
                
                # Use default color for out-of-range values (when using position numbers)
                if color_value in self.COLOR_MAP:
                    color_bgr = self.COLOR_MAP[color_value]
                else:
                    # Generate a distinct color if the value is not in COLOR_MAP
                    # This happens if show_numbers is true and cube_state contains numbers directly
                    # For a Rubik's cube, color_value should always be 0-5.
                    # This fallback helps visualize when cube_state itself is showing indices
                    
                    # Convert to HSV, then BGR for a distinct, deterministic color
                    # Hue from 0 to 179 (OpenCV range), Saturation and Value fixed
                    color_hue = (int(color_value) * 10) % 180 
                    color_bgr = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[color_hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0,0])

                # Calculate square position in pixels
                square_x_pixel = start_x_pixel + j * square_size
                square_y_pixel = start_y_pixel + i * square_size
                
                # Draw square
                cv2.rectangle(img, 
                            (square_x_pixel, square_y_pixel), 
                            (square_x_pixel + square_size, square_y_pixel + square_size),
                            color_bgr, -1)
                
                # Draw border
                cv2.rectangle(img, 
                            (square_x_pixel, square_y_pixel), 
                            (square_x_pixel + square_size, square_y_pixel + square_size),
                            (0, 0, 0), 2)
                
                # Add position numbers if enabled (0-53 global numbering)
                if self.show_numbers:
                    start_pos = self.FACE_POSITIONS[face_name]
                    # Calculate the global position number for this square
                    # The numbers within a face are 0-8, so i*3+j converts 2D (i,j) to 1D index
                    position_number = start_pos + i * 3 + j 
                    
                    # Text positioning
                    text_x = square_x_pixel + square_size // 2 - 10 # Adjust for centering
                    text_y = square_y_pixel + square_size // 2 + 5  # Adjust for centering
                    cv2.putText(img, str(position_number), 
                               (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add face label if enabled
        if self.show_face_labels:
            label_x = start_x_pixel + self.face_size // 2
            label_y = start_y_pixel + self.face_size // 2
            cv2.putText(img, face_name, 
                       (label_x - 30, label_y), # Adjust position to roughly center text
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _apply_view_transform(self, img: np.ndarray, 
                            view_angle: Tuple[float, float, float]) -> np.ndarray:
        """
        Apply view transformation to the image.
        This is a simplified implementation - in a full 3D renderer,
        you would use proper 3D transformations.
        """
        # For now, just rotate the image based on view angle
        # This is a placeholder for proper 3D rendering
        x_angle, y_angle, z_angle = view_angle
        
        # Simple 2D rotation as placeholder
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        
        # Apply rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, y_angle, 1.0)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
        
        return rotated_img
    
    def _render_symbolic(self, cube_state: np.ndarray) -> Dict[str, Any]:
        """
        Render cube as symbolic representation.
        
        Args:
            cube_state: 54-element array representing cube state
            
        Returns:
            Dictionary containing symbolic representation
        """
        symbolic_state = {}
        
        # Add face information
        for face_name in self.FACE_NAMES:
            face_data = self._get_face_data(cube_state, face_name)
            
            # Convert to color names
            face_colors = []
            for color_val in face_data.flatten():
                if color_val in self.COLOR_NAMES:
                    face_colors.append(self.COLOR_NAMES[color_val])
                else:
                    face_colors.append(f"COLOR_{color_val}")
            
            # Reshape to 3x3 grid
            face_grid = np.array(face_colors).reshape(3, 3)
            
            # Get center color
            center_val = face_data[1, 1]
            center_color = self.COLOR_NAMES.get(center_val, f"COLOR_{center_val}")
            
            symbolic_state[face_name] = {
                'grid': face_grid.tolist(),
                'center_color': center_color,
                'solved': np.all(face_data == center_val)
            }
        
        # Add overall state information
        solved_faces = sum(1 for face_name in self.FACE_NAMES 
                          if symbolic_state[face_name]['solved'])
        
        symbolic_state['overall'] = {
            'solved_faces': solved_faces,
            'total_faces': 6,
            'is_solved': solved_faces == 6
        }
        
        return symbolic_state
    
    def render_to_string(self, cube_state: np.ndarray) -> str:
        """
        Render cube state as a string representation.
        Useful for debugging and logging.
        """
        symbolic = self._render_symbolic(cube_state)
        
        result = []
        result.append("Cube State (Symbolic):")
        result.append(f"Solved: {symbolic['overall']['is_solved']}")
        result.append(f"Solved faces: {symbolic['overall']['solved_faces']}/6")
        result.append("\n") # Add a newline for better spacing
        
        # Revert to simpler face by face string representation as the aligned textual grid is complex
        for face_name in self.FACE_NAMES:
            face_data = symbolic[face_name]
            result.append(f"--- {face_name} (center: {face_data['center_color']}) ---")
            
            for row in face_data['grid']:
                result.append("  " + " ".join([f"{color:<8}" for color in row])) # Pad for alignment
            result.append("\n")
        
        return "\n".join(result)
    
    def save_image(self, cube_state: np.ndarray, filename: str, 
                  view_angle: Optional[Tuple[float, float, float]] = None):
        """
        Save rendered image to file.
        
        Args:
            cube_state: 54-element array representing cube state
            filename: Output filename
            view_angle: Optional view angle
        """
        img = self._render_image(cube_state, view_angle)
        cv2.imwrite(filename, img)
    
    def create_animation_frames(self, cube_state: np.ndarray, 
                              rotation_angles: List[Tuple[float, float, float]]) -> List[np.ndarray]:
        """
        Create animation frames by rotating the cube view.
        
        Args:
            cube_state: 54-element array representing cube state
            rotation_angles: List of view angles for each frame
            
        Returns:
            List of rendered images
        """
        frames = []
        for angle in rotation_angles:
            frame = self._render_image(cube_state, angle)
            frames.append(frame)
        return frames


def render_cube_image(cube_state: np.ndarray, 
                     image_size: int = 400, 
                     view_angle: Optional[Tuple[float, float, float]] = None,
                     show_face_labels: bool = True,
                     show_numbers: bool = False) -> np.ndarray:
    """Quick function to render cube as image"""
    renderer = CubeRenderer(RenderMode.IMAGE, image_size, show_face_labels, show_numbers)
    result = renderer.render(cube_state, view_angle)
    return result['image']


def render_cube_symbolic(cube_state: np.ndarray) -> Dict[str, Any]:
    """Quick function to render cube as symbolic state"""
    renderer = CubeRenderer(RenderMode.SYMBOLIC)
    result = renderer.render(cube_state)
    return result['symbolic']


def render_cube_both(cube_state: np.ndarray, 
                    image_size: int = 400,
                    view_angle: Optional[Tuple[float, float, float]] = None,
                    show_face_labels: bool = True,
                    show_numbers: bool = False) -> Dict[str, Any]:
    """Quick function to render cube in both modes"""
    renderer = CubeRenderer(RenderMode.BOTH, image_size, show_face_labels, show_numbers)
    return renderer.render(cube_state, view_angle)