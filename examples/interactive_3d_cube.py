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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cube.core.base_simulator import CubeSimulator
from cube.envs.vertex_view_env import VertexViewRenderer


# --- Main Application Class ---
class Cube3DApp:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.cube = CubeSimulator()
        self.renderer = VertexViewRenderer()

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
        
        # Color mapping is now handled by the renderer, which inherits them from BaseRenderer

    def _draw_scene(self):
        """Draws the entire scene (3D cube and 2D overlays)."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        current_state = self.cube.get_state()
        self.renderer.draw_cube(
            current_state, 
            self.rot_x, 
            self.rot_y, 
            self.zoom,
            show_numbers=self.show_numbers,
            show_face_labels=self.show_face_labels,
            font=self.font,
            font_small=self.font_small,
            window_size=(self.width, self.height)
        )

        # Display rotation info
        info_text = f"rot_x: {self.rot_x:.1f}, rot_y: {self.rot_y:.1f}"
        self._render_ui_text(info_text, (10, 10), self.font_small)

    def _render_ui_text(self, text, position, font, text_color=(255, 255, 255), bg_color=(0, 0, 0, 150)):
        """Renders UI text with a background panel at a given screen position."""
        text_surface = font.render(text, True, text_color)
        text_rect = text_surface.get_rect(topleft=position)

        # Create background surface
        bg_surface = pygame.Surface((text_rect.width + 10, text_rect.height + 6), SRCALPHA)
        bg_surface.fill(bg_color)
        bg_data = pygame.image.tostring(bg_surface, "RGBA", True)

        # Switch to 2D ortho mode
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
        
        # Draw background
        # Note: glWindowPos2i's y-coordinate is from the bottom-left.
        bg_x = text_rect.left - 5
        bg_y = self.height - text_rect.bottom - 3
        glWindowPos2i(bg_x, bg_y)
        glDrawPixels(bg_surface.get_width(), bg_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, bg_data)

        # Draw text
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        text_x = text_rect.left
        text_y = self.height - text_rect.bottom
        glWindowPos2i(text_x, text_y)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Restore 3D projection
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

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
                    # print(f"Show numbers: {self.show_numbers}")
                if event.key == pygame.K_t:
                    self.show_face_labels = not self.show_face_labels
                    # print(f"Show face labels: {self.show_face_labels}")

                # Move mapping
                key_map = {
                    pygame.K_f: "F", pygame.K_b: "B", pygame.K_l: "L",
                    pygame.K_r: "R", pygame.K_u: "U", pygame.K_d: "D"
                }
                move = key_map.get(event.key)
                if move:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        self.cube.apply_move(f"{move}'")
                        # print(f"Applied {move}'")
                    else:
                        self.cube.apply_move(move)
                        # print(f"Applied {move}")

                if event.key == pygame.K_SPACE:
                    self.cube.scramble(20)
                    # print("Scrambled")
                if event.key == pygame.K_BACKSPACE:
                    self.cube.reset()
                    # print("Reset")

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