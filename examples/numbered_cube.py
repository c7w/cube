#!/usr/bin/env python3
"""
Numbered 2D Rubik's Cube visualization for debugging edge exchanges.
Shows position numbers 1-54 on each square to help diagnose rotation problems.
"""

import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.utils.cube_simulator import CubeSimulator, Color
from environment.renderer import CubeRenderer, RenderMode

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