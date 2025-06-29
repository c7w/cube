import numpy as np
from typing import Any, Dict, Optional, Tuple
from gymnasium import spaces
from cube.core.base_env import BaseCubeEnvironment, BaseRenderer
from cube.core.base_simulator import State, CubeSimulator as StateManager
from cube.core.base_action_space import ActionSpace, ActionType, CubeAction
from cube.core.base_reward import RewardFunction, get_reward_function
from cube.utils.face_utils import FULL_NEIGHBOR_MAP, Face, rotate_face_once
from PIL import Image, ImageDraw, ImageFont
import os

# --- 3D Cube Geometry (extracted from interactive_3d_cube.py) ---
try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

VERTICES = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)

EDGES = (
    (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7),
    (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7)
)

SURFACES = {
    'U': (1, 2, 7, 5),  # UP (Y=1)
    'D': (0, 3, 6, 4),  # DOWN (Y=-1)
    'R': (0, 1, 5, 4),  # RIGHT (X=1)
    'L': (2, 3, 6, 7),  # LEFT (X=-1)
    'F': (4, 5, 7, 6),  # FRONT (Z=1)
    'B': (0, 1, 2, 3)   # BACK (Z=-1)
}

# Map face names to sticker indices for a 3x3 face grid
FACE_GRID_INDICES = {
    'F': list(range(0, 9)),
    'B': list(range(9, 18)),
    'L': list(range(18, 27)),
    'R': list(range(27, 36)),
    'U': list(range(36, 45)),
    'D': list(range(45, 54)),
}


def make_vertex_view_env(max_steps: int = 1000, reward_function_config: Dict[str, Any] = {"type": "dummy"}):
    state_manager = StateManager()
    action_space_config = ActionSpace(view_actions=VertexViewActions)
    reward_function = get_reward_function(**reward_function_config)
    renderer = VertexViewRenderer()
    return VertexViewEnvironment(state_manager, action_space_config, reward_function, renderer, max_steps)

VertexViewActions = [
    "rotate_vertex_120", "rotate_vertex_240",
]

class VertexViewRenderer(BaseRenderer):
    """
    Renderer for vertex-based cube representation.
    Handles conversion between state arrays and 3D vertex-based observations.
    """
    def __init__(self, image_size: Tuple[int, int] = (84, 84), use_label: bool = False, border_size: int = 3):
        super().__init__()
        self.image_size = image_size
        self.use_label = use_label
        self.border_size = border_size
        self.sticker_map = self._create_sticker_map()

    def _create_sticker_map(self):
        """Creates a map from (x,y,z) cubelet position to its sticker indices."""
        sticker_map = {}
        # Iterate through each of the 27 cubelets by position
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    if x == 0 and y == 0 and z == 0:
                        continue # Skip center invisible cubelet
                    
                    pos = (x,y,z)
                    sticker_map[pos] = {}
                    
                    if y == 1: # UP face
                        row, col = 1 + z, 1 + x
                        idx = row * 3 + col
                        sticker_map[pos]['U'] = FACE_GRID_INDICES['U'][idx]
                        
                    if y == -1: # DOWN face  
                        row, col = 1 - z, 1 + x
                        idx = row * 3 + col
                        sticker_map[pos]['D'] = FACE_GRID_INDICES['D'][idx]
                        
                    if z == 1: # FRONT face
                        row, col = 1 - y, 1 + x
                        idx = row * 3 + col
                        sticker_map[pos]['F'] = FACE_GRID_INDICES['F'][idx]
                        
                    if z == -1: # BACK face
                        row, col = 1 - y, 1 + x
                        idx = row * 3 + col
                        sticker_map[pos]['B'] = FACE_GRID_INDICES['B'][idx]
                        
                    if x == -1: # LEFT face
                        row, col = 1 - y, 1 + z
                        idx = row * 3 + col
                        sticker_map[pos]['L'] = FACE_GRID_INDICES['L'][idx]
                        
                    if x == 1: # RIGHT face
                        row, col = 1 - y, 1 - z
                        idx = row * 3 + col
                        sticker_map[pos]['R'] = FACE_GRID_INDICES['R'][idx]
        return sticker_map

    def _draw_cubelet(self, pos, state):
        """Draws a single cubelet at the given position with the correct colors."""
        glPushMatrix()
        glTranslatef(pos[0] * 2.1, pos[1] * 2.1, pos[2] * 2.1)
        
        face_colors = self.sticker_map.get(pos, {})
        
        glBegin(GL_QUADS)
        for face_name, surface_indices in SURFACES.items():
            sticker_index = face_colors.get(face_name)
            if sticker_index is not None:
                color_id = state[sticker_index]
                # Use color mapping from BaseRenderer
                color = tuple(c/255.0 for c in self.STATE_TO_RGB[color_id])
            else:
                color = (0.1, 0.1, 0.1) # Inner part color
            
            glColor3fv(color)
            for vertex_index in surface_indices:
                glVertex3fv(VERTICES[vertex_index])
        glEnd()
        
        glColor3fv((0, 0, 0))
        glBegin(GL_LINES)
        for edge in EDGES:
            for vertex in edge:
                glVertex3fv(VERTICES[vertex])
        glEnd()
        glPopMatrix()

    def draw_cube(self, state, rot_x, rot_y, zoom, show_numbers=False, show_face_labels=False, font=None, font_small=None, window_size=(800,600)):
        """Draws the entire cube in the current Pygame OpenGL context."""
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame and PyOpenGL are required for interactive rendering.")

        # --- 3D Drawing ---
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, zoom)
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        
        for pos in self.sticker_map.keys():
            self._draw_cubelet(pos, state)

        # --- 2D Overlay Drawing (Pygame specific) ---
        self._draw_overlays(show_numbers, show_face_labels, font, font_small, window_size)

    def _render_text(self, text, pos, font, color, window_size):
        """Renders text on the screen at a given 2D position."""
        width, height = window_size
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)

        # We need to switch to 2D ortho mode to render text
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, width, 0, height)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw text bg for readability
        bg_surface = pygame.Surface((text_rect.width + 4, text_rect.height + 4), SRCALPHA)
        bg_surface.fill((255, 255, 255, 180))
        bg_data = pygame.image.tostring(bg_surface, "RGBA", True)
        glWindowPos2i(text_rect.left - 2, height - text_rect.bottom - 2)
        glDrawPixels(bg_surface.get_width(), bg_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
        
        # Draw text
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glWindowPos2i(text_rect.left, height - text_rect.bottom)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _draw_overlays(self, show_numbers, show_face_labels, font, font_small, window_size):
        """Draws 2D text labels (numbers, face names) over the 3D view."""
        if not (show_numbers or show_face_labels) or not font:
            return

        # Get matrices and viewport
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)
        width, height = window_size

        # Iterate through cubelets to find face centers
        for pos, faces in self.sticker_map.items():
            for face_name, sticker_index in faces.items():
                # Get center of the face in 3D
                face_center_local = np.mean([VERTICES[i] for i in SURFACES[face_name]], axis=0)
                
                # Check face visibility by transforming the normal vector
                normal = np.array(face_center_local)
                if np.linalg.norm(normal) > 0:
                    normal = normal / np.linalg.norm(normal)
                
                # Transform normal by modelview matrix (rotation part only)
                rotated_normal = modelview[:3, :3].T @ normal
                
                # If normal's z component is negative, it's facing away from camera
                if rotated_normal[2] < 0.1:
                    continue

                face_center_world = (
                    face_center_local[0] + pos[0] * 2.1,
                    face_center_local[1] + pos[1] * 2.1,
                    face_center_local[2] + pos[2] * 2.1
                )
                
                # Project to 2D screen coordinates
                screen_coords = gluProject(*face_center_world, modelview, projection, viewport)
                
                if screen_coords[2] > 1.0: # Check if behind camera
                    continue
                
                x, y = int(screen_coords[0]), int(height - screen_coords[1])

                # Draw the text
                if show_numbers:
                    self._render_text(f"{sticker_index}", (x, y), font_small, (0, 0, 0), window_size)
                
                if show_face_labels and np.all(np.array(pos) == (0,0,0,)): # Center pieces
                     self._render_text(face_name, (x, y), font, (20, 20, 20), window_size)

    def get_image_observation(self, state: State, viewpoint: Any) -> np.ndarray:
        """
        Render the cube state from a given viewpoint to an image array using offscreen rendering.
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame, PyOpenGL, and Pillow are required for image observation.")

        # --- Setup Pygame for offscreen rendering ---
        pygame.init()
        # Using a dummy display is necessary for Pygame to handle fonts and other resources.
        # It's created once and reused.
        if not pygame.display.get_init() or pygame.display.get_surface() is None:
             pygame.display.set_mode(self.image_size, DOUBLEBUF | OPENGL | pygame.HIDDEN)
        
        font_small = pygame.font.Font(None, 18) if self.use_label else None

        # --- Setup OpenGL and FBO ---
        fbo = glGenFramebuffers(1)
        try:
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)

            # Create texture to render to
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.image_size[0], self.image_size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

            # Create depth buffer
            rbo = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, rbo)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, self.image_size[0], self.image_size[1])
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo)

            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError("Framebuffer is not complete!")

            # --- Render Scene ---
            glEnable(GL_DEPTH_TEST)
            glViewport(0, 0, self.image_size[0], self.image_size[1])
            glClearColor(0.8, 0.8, 0.8, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, (self.image_size[0] / self.image_size[1]), 0.1, 100.0)

            # Draw the cube using the Pygame-compatible method (with potential overlays)
            self.draw_cube(state, viewpoint['rot_x'], viewpoint['rot_y'], viewpoint['zoom'],
                           show_numbers=self.use_label, font_small=font_small, window_size=self.image_size)

            # --- Read pixels ---
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            pixels = glReadPixels(0, 0, self.image_size[0], self.image_size[1], GL_RGB, GL_UNSIGNED_BYTE)
            
            image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.image_size[1], self.image_size[0], 3)
            image = np.flipud(image)
        finally:
            # Cleanup
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glDeleteFramebuffers(1, [fbo])
            # The rest of the cleanup for texture and rbo should be here
            glDeleteTextures(1, [texture])
            glDeleteRenderbuffers(1, [rbo])

        return image

    def get_observation(self, state: State, viewpoint: Any) -> Any:
        return self.get_image_observation(state, viewpoint)


class VertexViewEnvironment(BaseCubeEnvironment):
    def __init__(self, 
                 state_manager: StateManager, 
                 action_space_config: ActionSpace, 
                 reward_function: RewardFunction, 
                 renderer: BaseRenderer, 
                 max_steps: int = 1000,
                 image_size: Tuple[int, int] = (84, 84),
                 border_size: int = 3,
                 use_label: bool = False):
        self.image_size = image_size
        self.use_label = use_label
        self.renderer = VertexViewRenderer(image_size=image_size, use_label=use_label, border_size=border_size)
        super().__init__(
            state_manager=state_manager,
            action_space_config=action_space_config,
            reward_function=reward_function,
            renderer=self.renderer,
            max_steps=max_steps
        )

    def _setup_observation_space(self):
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.image_size[0], self.image_size[1], 3),
            dtype=np.uint8
        )


    def get_observation(self) -> Any:
        state = self.state_manager.get_state()
        return self.renderer.get_observation(state, self.viewpoint)

    def _reset_viewpoint(self):
        # TODO: Initialize vertex viewpoint
        # For a 3D renderer, this might be camera angles
        self.viewpoint = {"rot_x": -30, "rot_y": 45, "zoom": -15}

    def _update_viewpoint(self, action_name: str):
        # TODO: Update vertex viewpoint based on actions like "rotate_vertex_120"
        pass
