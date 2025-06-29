import numpy as np
from typing import Any, Dict, Optional, Tuple
from gymnasium import spaces
from cube.core.base_env import BaseCubeEnvironment, BaseRenderer
from cube.core.base_simulator import State, CubeSimulator as StateManager
from cube.core.base_action_space import ActionSpace, ActionType, CubeAction
from cube.core.base_reward import RewardFunction, get_reward_function
from cube.utils.face_utils import FULL_NEIGHBOR_MAP, Face, rotate_face_once
from PIL import Image, ImageDraw, ImageFont

def make_face_view_env(max_steps: int = 1000, reward_function_config: Dict[str, Any] = {"type": "dummy"}):
    state_manager = StateManager()
    action_space_config = ActionSpace(view_actions=FaceViewActions)
    reward_function = get_reward_function(**reward_function_config)
    renderer = FaceViewRenderer()
    return FaceViewEnvironment(state_manager, action_space_config, reward_function, renderer, max_steps)

FaceViewActions = [
    "view_left", "view_up", "view_right", "view_down", "rotate_view_90", "rotate_view_180", "rotate_view_270"
]

class FaceViewRenderer(BaseRenderer):
    """
    Renderer interface for face-based cube observation.
    This is an interface class - implementations should be provided elsewhere.
    """
    
    def __init__(self, observation_type: str = "text", image_size: Tuple[int, int] = (84, 84), use_label: bool = False, border_size: int = 3):
        self.observation_type = observation_type
        self.image_size = image_size
        self.use_label = use_label
        self.border_size = border_size
        
        if self.use_label:
            assert self.observation_type == "image", "Label is only supported for image observation"
        
    def get_text_observation(self, state: State, viewpoint: Face) -> str:
        """Get text-based face observation for current view"""
        # Get face colors (9 values) and convert to string
        face_colors = state[list(viewpoint)]
        return ''.join(self.STATE_TO_COLOR[color_id] for color_id in face_colors)
    
    def get_image_observation(self, state: State, viewpoint: Face) -> np.ndarray:
        """Get image-based face observation for current view"""
        # Get face colors (9 values)
        face_colors = state[list(viewpoint)]
        
        # Create 3x3 grid of RGB colors
        grid = np.zeros((3, 3, 3), dtype=np.uint8)
        for i in range(3):
            for j in range(3):
                color_id = face_colors[i * 3 + j]
                grid[i, j] = self.STATE_TO_RGB[color_id]
        
        # Resize to desired image size using simple upsampling
        scale_h, scale_w = self.image_size[0] // 3, self.image_size[1] // 3
        image = np.kron(grid, np.ones((scale_h, scale_w, 1), dtype=np.uint8))
        
        # Add black borders between grid cells
        border_width = max(1, min(scale_h, scale_w) // 20)  # Dynamic border width
        
        # Create image with borders
        bordered_image = image.copy()
        
        # Add vertical borders
        for i in range(1, 3):  # Between columns
            x_pos = i * scale_w
            start_x = max(0, x_pos - border_width // 2)
            end_x = min(self.image_size[1], x_pos + border_width // 2 + 1)
            bordered_image[:, start_x:end_x] = [0, 0, 0]  # Black color
        
        # Add horizontal borders
        for i in range(1, 3):  # Between rows
            y_pos = i * scale_h
            start_y = max(0, y_pos - border_width // 2)
            end_y = min(self.image_size[0], y_pos + border_width // 2 + 1)
            bordered_image[start_y:end_y, :] = [0, 0, 0]  # Black color
        
        # Add outer border around the entire image
        outer_border_width = max(2, border_width)
        bordered_image[:outer_border_width, :] = [0, 0, 0]  # Top
        bordered_image[-outer_border_width:, :] = [0, 0, 0]  # Bottom
        bordered_image[:, :outer_border_width] = [0, 0, 0]  # Left
        bordered_image[:, -outer_border_width:] = [0, 0, 0]  # Right
        
        image = bordered_image
        if self.use_label:
            label = Image.fromarray(image)
            # overlay state numbers as label to the image
            label_image = Image.new("RGB", (self.image_size[0], self.image_size[1]), (255, 255, 255))
            label_image.paste(label, (0, 0))
            # write state numbers as label to the image
            draw = ImageDraw.Draw(label_image)
            try:
                # Try to use a default font, fallback to default if not available
                font = ImageFont.truetype("Arial.ttf", size=max(12, min(scale_h, scale_w) // 2))
            except (OSError, IOError):
                # Use default font if truetype font is not available
                font = ImageFont.load_default()
            
            for i in range(3):
                for j in range(3):
                    # Calculate position to center text in each grid cell
                    x = j * scale_w + scale_w // 2
                    y = i * scale_h + scale_h // 2
                    text = str(viewpoint[i * 3 + j])
                    
                    # Get text bounding box for centering
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Center the text
                    text_x = x - text_width // 2
                    text_y = y - text_height // 2
                    
                    # Draw text with black color and white outline for better visibility
                    draw.text((text_x-1, text_y-1), text, font=font, fill="white")
                    draw.text((text_x+1, text_y-1), text, font=font, fill="white")
                    draw.text((text_x-1, text_y+1), text, font=font, fill="white")
                    draw.text((text_x+1, text_y+1), text, font=font, fill="white")
                    draw.text((text_x, text_y), text, font=font, fill="black")
            
            return np.array(label_image)
        else:
            return image
    
    def get_observation(self, state: State, viewpoint: Face) -> Any:
        """Get observation based on current observation type"""
        if self.observation_type == "text":
            return self.get_text_observation(state, viewpoint)
        elif self.observation_type == "image":
            return self.get_image_observation(state, viewpoint)
        else:
            raise ValueError(f"Invalid observation_type: {self.observation_type}. Must be 'text' or 'image'.")


class FaceViewEnvironment(BaseCubeEnvironment):
    def __init__(self, 
                 state_manager: StateManager, 
                 action_space_config: ActionSpace, 
                 reward_function: RewardFunction, 
                 renderer: BaseRenderer, 
                 max_steps: int = 1000,
                 observation_type: str = "text",
                 image_size: Tuple[int, int] = (84, 84),
                 border_size: int = 3,
                 use_label: bool = False):
        
        self.observation_type = observation_type
        self.image_size = image_size
        self.use_label = use_label

        # Initialize face view parameters
        self.viewpoint: Face = (0, 1, 2, 3, 4, 5, 6, 7, 8)  # Default to FRONT face
        
        # Initialize renderer (interface only)
        self.renderer = FaceViewRenderer(observation_type=observation_type, 
                                         image_size=image_size, 
                                         use_label=use_label, 
                                         border_size=border_size)
        
        # Call parent constructor
        super().__init__(
            state_manager=state_manager,
            action_space_config=action_space_config,
            reward_function=reward_function,
            renderer=renderer,
            max_steps=max_steps
        )
    
    def _setup_observation_space(self):
        """
        [ABSTRACT METHOD IMPLEMENTATION]
        Setup observation space for face observation
        """
        if self.observation_type == "text":
            # Observation is a 9-char string representing the current face
            self.observation_space = spaces.Text(9, charset=list(self.renderer.STATE_TO_COLOR.values()))
        elif self.observation_type == "image":
            # Observation is an RGB image of the face
            # Shape: (height, width, channels)
            self.observation_space = spaces.Box(
                low=0, 
                high=255, 
                shape=(self.image_size[0], self.image_size[1], 3),
                dtype=np.uint8
            )
        else:
            raise ValueError(f"Invalid observation_type: {self.observation_type}. Must be 'text' or 'image'.")
    
    def get_observation(self) -> Any:
        """
        [ABSTRACT METHOD IMPLEMENTATION]
        Get current face observation through renderer
        """
        # Get observation from renderer
        state = self.state_manager.get_state()
        return self.renderer.get_observation(state, self.viewpoint)
    
    def _reset_viewpoint(self):
        """
        [ABSTRACT METHOD IMPLEMENTATION]
        Reset camera parameters - set to default face view
        """
        self.viewpoint : Face = (0, 1, 2, 3, 4, 5, 6, 7, 8)  # Reset to FRONT face
    
    def _update_viewpoint(self, action_name: str):
        """
        [ABSTRACT METHOD IMPLEMENTATION]
        Update camera parameters based on view actions
        """
        if action_name == "rotate_view_90":
            self.viewpoint = rotate_face_once(self.viewpoint)
        elif action_name == "rotate_view_180":
            rotated = rotate_face_once(self.viewpoint)
            self.viewpoint = rotate_face_once(rotated)
        elif action_name == "rotate_view_270":
            rotated = rotate_face_once(self.viewpoint)
            rotated = rotate_face_once(rotated)
            self.viewpoint = rotate_face_once(rotated)
        elif action_name.startswith("view_"):
            # Handle face switching: view_left, view_up, view_right, view_down
            neighbor_name = action_name[5:]  # Remove "view_" prefix
            # read the face_name from FULL_NEIGHBOR_MAP. [left, up, right, down]
            neighbor_idx = ['left', 'up', 'right', 'down'].index(neighbor_name)
            self.viewpoint = FULL_NEIGHBOR_MAP[self.viewpoint][neighbor_idx]
