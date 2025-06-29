#!/usr/bin/env python3
"""
Numbered 2D Rubik's Cube visualization for debugging edge exchanges.
Shows position numbers 1-54 on each square to help diagnose rotation problems.
"""

import sys
import os
import cv2
import numpy as np

# ======= Simple Renderer Begin =======
from enum import Enum
from typing import Dict, Any, Optional, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cube.core.base_simulator import CubeSimulator

class RenderMode(Enum):
    IMAGE = "image"
    BOTH = "both"

class Color:
    WHITE = 0
    YELLOW = 1
    RED = 2
    ORANGE = 3
    BLUE = 4
    GREEN = 5

class CubeRenderer:
    """
    Simple 2D renderer for Rubik's cube visualization.
    Creates a 2D unfolded view of the cube with optional numbering.
    """
    def __init__(self, 
                 mode: RenderMode = RenderMode.IMAGE,
                 image_size: int = 600,
                 show_face_labels: bool = True,
                 show_numbers: bool = False):
        self.mode = mode
        self.base_image_size = image_size  # Store the requested size
        self.show_face_labels = show_face_labels
        self.show_numbers = show_numbers
        # Strictly follow base_env.py mapping
        self.STATE_TO_FACE = {0: "FRONT", 1: "BACK", 2: "LEFT", 3: "RIGHT", 4: "UP", 5: "DOWN"}
        self.FACE_TO_COLOR = {"FRONT": "R", "BACK": "O", "LEFT": "B", "RIGHT": "G", "UP": "Y", "DOWN": "W"}
        self.COLOR_TO_RGB = {
            "R": (255, 0, 0),
            "O": (255, 165, 0),
            "B": (0, 0, 255),
            "G": (0, 255, 0),
            "Y": (255, 255, 0),
            "W": (255, 255, 255)
        }
        self.face_positions = {
            'UP': (1, 0),      # Top
            'LEFT': (0, 1),    # Left
            'FRONT': (1, 1),   # Center
            'RIGHT': (2, 1),   # Right
            'DOWN': (1, 2),    # Bottom
            'BACK': (1, 3)     # Far bottom (below DOWN)
        }
        self.face_starts = {
            'FRONT': 0,   # 0-8
            'BACK': 9,    # 9-17
            'LEFT': 18,   # 18-26
            'RIGHT': 27,  # 27-35
            'UP': 36,     # 36-44
            'DOWN': 45    # 45-53
        }
        self.standard_faces = {
            "FRONT": (0, 1, 2, 3, 4, 5, 6, 7, 8),
            "BACK": (17, 16, 15, 14, 13, 12, 11, 10, 9),  # BACK face is reversed
            "LEFT": (18, 19, 20, 21, 22, 23, 24, 25, 26),
            "RIGHT": (27, 28, 29, 30, 31, 32, 33, 34, 35),
            "UP": (36, 37, 38, 39, 40, 41, 42, 43, 44),
            "DOWN": (45, 46, 47, 48, 49, 50, 51, 52, 53)
        }
    def render(self, state: np.ndarray) -> Dict[str, Any]:
        face_size = self.base_image_size // 4  # 4 faces in height (UP, middle row, DOWN, BACK)
        square_size = face_size // 3
        total_image_size = face_size * 4
        img = np.zeros((total_image_size, total_image_size, 3), dtype=np.uint8)
        img.fill(50)  # Dark background
        for face_name in ['UP', 'LEFT', 'FRONT', 'RIGHT', 'DOWN', 'BACK']:
            grid_x, grid_y = self.face_positions[face_name]
            face_indices = self.standard_faces[face_name]
            face_x = grid_x * face_size
            face_y = grid_y * face_size
            for i in range(3):
                for j in range(3):
                    square_idx = face_indices[i * 3 + j]
                    color_id = state[square_idx]
                    # base_env.py风格三步映射
                    face_label = self.STATE_TO_FACE[color_id]
                    color_letter = self.FACE_TO_COLOR[face_label]
                    color = self.COLOR_TO_RGB[color_letter][::-1]
                    x = face_x + j * square_size
                    y = face_y + i * square_size
                    cv2.rectangle(img, (x, y), (x + square_size - 2, y + square_size - 2), color, -1)
                    cv2.rectangle(img, (x, y), (x + square_size - 2, y + square_size - 2), (0, 0, 0), 2)
                    if self.show_numbers:
                        text_x = x + square_size // 2 - 8
                        text_y = y + square_size // 2 + 5
                        cv2.putText(img, str(square_idx), (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            if self.show_face_labels:
                label_x = face_x + face_size // 2 - 25
                label_y = face_y + face_size + 15
                cv2.putText(img, face_name, (label_x, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return {
            'image': img,
            'mode': self.mode.value
        }
    def save_image(self, state: np.ndarray, filename: str):
        result = self.render(state)
        cv2.imwrite(filename, result['image'])
# ======= Simple Renderer End =======



HELP_TEXT = """
Controls:
  F/B/L/R/U/D: Rotate face clockwise
  f/b/l/r/u/d: Rotate face counter-clockwise (Shift+F etc.)
  Space: Scramble
  R: Reset
  N: Toggle position numbers
  Q/Esc: Quit
"""

def main():
    print(HELP_TEXT)
    
    # 初始化立方体和渲染器
    cube = CubeSimulator()
    renderer = CubeRenderer(
        mode=RenderMode.IMAGE,
        image_size=600,
        show_face_labels=True,
        show_numbers=True  # 默认显示编号
    )
    
    # 创建窗口
    cv2.namedWindow('Numbered Cube - Debug Mode', cv2.WINDOW_AUTOSIZE)
    
    print("Initial cube with position numbers 1-54:")
    print("Face 0 (FRONT):   1-9")
    print("Face 1 (BACK):    10-18") 
    print("Face 2 (LEFT):    19-27")
    print("Face 3 (RIGHT):   28-36")
    print("Face 4 (UP):      37-45")
    print("Face 5 (DOWN):    46-54")
    print()
    
    while True:
        # 渲染立方体
        cube_state = cube.get_state()
        rendered = renderer.render(cube_state)
        img = rendered['image']
        
        # 显示图像
        cv2.imshow('Numbered Cube - Debug Mode', img)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q or Esc
            break
        elif key == ord('f'):
            cube.apply_move('F')
            print("Applied F move")
        elif key == ord('F'):  # Shift+F (实际是大写F)
            cube.apply_move("F'")
            print("Applied F' move")
        elif key == ord('b'):
            cube.apply_move('B')
            print("Applied B move")
        elif key == ord('B'):
            cube.apply_move("B'")
            print("Applied B' move")
        elif key == ord('l'):
            cube.apply_move('L')
            print("Applied L move")
        elif key == ord('L'):
            cube.apply_move("L'")
            print("Applied L' move")
        elif key == ord('r'):
            cube.apply_move('R')
            print("Applied R move")
        elif key == ord('R'):
            cube.apply_move("R'")
            print("Applied R' move")
        elif key == ord('u'):
            cube.apply_move('U')
            print("Applied U move")
        elif key == ord('U'):
            cube.apply_move("U'")
            print("Applied U' move")
        elif key == ord('d'):
            cube.apply_move('D')
            print("Applied D move")
        elif key == ord('D'):
            cube.apply_move("D'")
            print("Applied D' move")
        elif key == ord(' '):  # Space
            cube.scramble(10)
            print("Scrambled cube")
        elif key == ord('r') or key == ord('R'):
            cube.reset()
            print("Reset cube")
        elif key == ord('n') or key == ord('N'):
            renderer.show_numbers = not renderer.show_numbers
            status = "enabled" if renderer.show_numbers else "disabled"
            print(f"Position numbers {status}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 