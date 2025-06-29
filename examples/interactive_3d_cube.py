#!/usr/bin/env python3
"""
Interactive 3D Rubik's Cube Visualization.
Uses Pygame and PyOpenGL to render the cube in 3D.
"""

import sys
import os
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# Add parent directory to path to import environment modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environment.utils.cube_simulator import CubeSimulator
from environment.renderer import CubeRenderer # For color map

# --- 3D Cube Geometry ---
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
    'F': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'B': [9, 10, 11, 12, 13, 14, 15, 16, 17],
    'L': [18, 19, 20, 21, 22, 23, 24, 25, 26],
    'R': [27, 28, 29, 30, 31, 32, 33, 34, 35],
    'U': [36, 37, 38, 39, 40, 41, 42, 43, 44],
    'D': [45, 46, 47, 48, 49, 50, 51, 52, 53],
}

# --- Main Application Class ---
class Cube3DApp:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.cube = CubeSimulator()
        self.renderer = CubeRenderer()
        self.sticker_map = self._create_sticker_map()
        
        self.rot_x = -30
        self.rot_y = 45
        self.zoom = -40
        self.mouse_down = False
        self.last_mouse_pos = (0, 0)
        
        # --- Debugging Flags ---
        self.show_numbers = True
        self.show_face_labels = True
        self.font = None
        self.font_small = None

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
                    
                    # --- Corrected Sticker Mapping ---
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
                        
                    if z == -1: # BACK face (需要特殊处理方向)
                        row, col = 1 + y, 1 + x  # y和x方向都翻转，相当于旋转180度
                        idx = row * 3 + col
                        sticker_map[pos]['B'] = FACE_GRID_INDICES['B'][idx]
                        
                    if x == -1: # LEFT face
                        row, col = 1 - y, 1 + z
                        idx = row * 3 + col
                        sticker_map[pos]['L'] = FACE_GRID_INDICES['L'][idx]
                        
                    if x == 1: # RIGHT face
                        row, col = 1 - y, 1 - z  # 注意z方向翻转
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
                color = tuple(c/255.0 for c in self.renderer.COLOR_MAP[color_id])
            else:
                color = (0.1, 0.1, 0.1)
            
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

    def _draw_overlays(self):
        """Draws 2D text labels (numbers, face names) over the 3D view."""
        if not self.show_numbers and not self.show_face_labels:
            return

        # Get matrices and viewport
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport = glGetIntegerv(GL_VIEWPORT)

        # Iterate through cubelets to find face centers
        for pos, faces in self.sticker_map.items():
            for face_name, sticker_index in faces.items():
                # Get center of the face in 3D
                face_center_local = np.mean([VERTICES[i] for i in SURFACES[face_name]], axis=0)
                
                # Check face visibility by transforming the normal vector
                normal = np.array(face_center_local)
                normal = normal / np.linalg.norm(normal)
                
                # Transform normal by modelview matrix (rotation part only)
                rotated_normal = modelview[:3, :3].T @ normal
                
                # If normal's z component is negative, it's facing away from camera
                if rotated_normal[2] < 0:
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
                
                x, y = int(screen_coords[0]), int(self.height - screen_coords[1])

                # Draw the text
                if self.show_numbers:
                    self.render_text(f"{sticker_index}", (x, y))
                
                if self.show_face_labels and np.all(np.array(pos) == (0,0,0)): # Center pieces
                     self.render_text(face_name, (x, y), is_face_label=True)


    def render_text(self, text, pos, is_face_label=False):
        """Renders text on the screen at a given 2D position."""
        font = self.font if is_face_label else self.font_small
        color = (255, 255, 0) if is_face_label else (0, 0, 0)

        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        
        # We need to switch to 2D ortho mode to render text
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.width, 0, self.height)
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
        glWindowPos2i(text_rect.left - 2, self.height - text_rect.bottom - 2)
        glDrawPixels(bg_surface.get_width(), bg_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, bg_data)
        
        # Draw text
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glWindowPos2i(text_rect.left, self.height - text_rect.bottom)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()


    def _draw_scene(self):
        """Draws the entire scene (3D cube and 2D overlays)."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # --- 3D Drawing ---
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, self.zoom)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        
        current_state = self.cube.get_state()
        for pos in self.sticker_map.keys():
            self._draw_cubelet(pos, current_state)

        # --- 2D Overlay Drawing ---
        self._draw_overlays()


    def _handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return False
                
                # Toggle debug views
                if event.key == pygame.K_n:
                    self.show_numbers = not self.show_numbers
                    print(f"Show numbers: {self.show_numbers}")
                if event.key == pygame.K_t:
                    self.show_face_labels = not self.show_face_labels
                    print(f"Show face labels: {self.show_face_labels}")

                # Move mapping
                key_map = {
                    pygame.K_f: "F", pygame.K_b: "B", pygame.K_l: "L",
                    pygame.K_r: "R", pygame.K_u: "U", pygame.K_d: "D"
                }
                move = key_map.get(event.key)
                if move:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        self.cube.apply_move(f"{move}'")
                        print(f"Applied {move}'")
                    else:
                        self.cube.apply_move(move)
                        print(f"Applied {move}")

                if event.key == pygame.K_SPACE:
                    self.cube.scramble(20)
                    print("Scrambled")
                if event.key == pygame.K_BACKSPACE:
                    self.cube.reset()
                    print("Reset")

            if event.type == MOUSEBUTTONDOWN:
                if event.button == 4: self.zoom += 1
                if event.button == 5: self.zoom -= 1
                if event.button == 1:
                    self.mouse_down = True
                    self.last_mouse_pos = event.pos

            if event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False

            if event.type == MOUSEMOTION and self.mouse_down:
                dx, dy = event.pos[0] - self.last_mouse_pos[0], event.pos[1] - self.last_mouse_pos[1]
                self.rot_y += dx * 0.2
                self.rot_x += dy * 0.2
                self.last_mouse_pos = event.pos
        return True

    def run(self):
        pygame.init()
        self.font = pygame.font.Font(None, 30) # Font for face labels
        self.font_small = pygame.font.Font(None, 18) # Font for sticker numbers
        display = (self.width, self.height)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Interactive 3D Cube - Debug Mode")

        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
        
        print("--- Interactive 3D Cube (Debug Mode) ---")
        print("Controls:")
        print("  F/B/L/R/U/D: Rotate face clockwise")
        print("  Shift + Key: Rotate face counter-clockwise")
        print("  N: Toggle sticker numbers")
        print("  T: Toggle face labels")
        print("  Space: Scramble")
        print("  Backspace: Reset")
        print("  Mouse Drag: Rotate view")
        print("  Mouse Wheel: Zoom")
        print("  Q/Esc: Quit")

        running = True
        while running:
            running = self._handle_input()
            self._draw_scene()
            pygame.display.flip()
            pygame.time.wait(10)
        
        pygame.quit()


if __name__ == '__main__':
    app = Cube3DApp()
    app.run() 