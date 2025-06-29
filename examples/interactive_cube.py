#!/usr/bin/env python3

"""
Interactive Rubik's Cube Demo
Uses the new 54-element state system with explicit permutations.
"""

import sys
import os

# Add parent directory to path to import environment modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from environment.utils.cube_simulator import CubeSimulator
from environment.renderer import CubeRenderer, RenderMode
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def create_2d_debug_view(cube_state):
    """Create 2D numbered layout view for debugging"""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Define face positions in cross layout
    face_positions = {
        'UP': (3, 6),      # Top
        'LEFT': (0, 3),    # Left
        'FRONT': (3, 3),   # Center
        'RIGHT': (6, 3),   # Right
        'BACK': (9, 3),    # Far right
        'DOWN': (3, 0)     # Bottom
    }
    
    # Face starting positions in 54-element state
    face_starts = {
        'FRONT': 0,   # 0-8
        'BACK': 9,    # 9-17
        'LEFT': 18,   # 18-26
        'RIGHT': 27,  # 27-35
        'UP': 36,     # 36-44
        'DOWN': 45    # 45-53
    }
    
    # Color mapping
    colors = ['white', 'yellow', 'red', 'orange', 'blue', 'green']
    
    for face_name, (face_x, face_y) in face_positions.items():
        start_pos = face_starts[face_name]
        
        # Draw each square in the face
        for i in range(3):
            for j in range(3):
                x = face_x + j
                y = face_y + (2 - i)  # Flip y to match cube orientation
                
                # Get color and position number
                pos_idx = start_pos + i * 3 + j
                color_val = cube_state[pos_idx]
                
                # Use color if it's a valid color, otherwise use gray
                if 0 <= color_val < len(colors):
                    face_color = colors[color_val]
                else:
                    face_color = 'lightgray'
                
                # Draw square
                rect = Rectangle((x, y), 1, 1, linewidth=2, edgecolor='black', 
                               facecolor=face_color)
                ax.add_patch(rect)
                
                # Add position number
                ax.text(x + 0.5, y + 0.5, str(pos_idx), 
                       ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add face label
        ax.text(face_x + 1.5, face_y + 3.5, face_name, 
               ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.title("Rubik's Cube 2D Debug View (Position Numbers 0-53)", fontsize=14)
    return fig


def test_all_moves(cube):
    """Test all moves and their inverses"""
    print("\n=== Testing All Moves ===")
    moves = ['F', 'B', 'R', 'L', 'U', 'D']
    
    for move in moves:
        print(f"\nTesting {move} and {move}'...")
        
        # Save initial state
        initial_state = cube.get_state().copy()
        
        # Apply move
        cube.apply_move(move)
        after_move = cube.get_state().copy()
        
        # Apply inverse
        cube.apply_move(move + "'")
        after_inverse = cube.get_state().copy()
        
        # Check if back to initial
        back_to_initial = np.array_equal(initial_state, after_inverse)
        print(f"  {move} + {move}' returns to initial: {back_to_initial}")
        
        if not back_to_initial:
            diff_positions = np.where(initial_state != after_inverse)[0]
            print(f"  Differences at positions: {diff_positions[:10]}...")  # Show first 10


def analyze_move(cube, move):
    """Analyze what a specific move does"""
    print(f"\n=== Analyzing move {move} ===")
    
    # Set each position to its index number
    for i in range(54):
        cube.state[i] = i
    
    print("Before move:")
    state_before = cube.get_state().copy()
    
    # Apply move
    cube.apply_move(move)
    state_after = cube.get_state().copy()
    
    print("After move:")
    print("Position changes:")
    for i in range(54):
        if state_before[i] != state_after[i]:
            print(f"  Position {i}: {state_before[i]} -> {state_after[i]}")


def main():
    """Main interactive demo"""
    print("Interactive Rubik's Cube Demo")
    print("=============================")
    print("New 54-element state system with explicit permutations")
    print()
    print("Commands:")
    print("  F, F', B, B', R, R', L, L', U, U', D, D' - Apply moves")
    print("  reset - Reset to solved state")
    print("  scramble [n] - Scramble with n moves (default 20)")
    print("  undo - Undo last move")
    print("  show - Show current state info")
    print("  save <filename> - Save current state as image")
    print("  debug - Show 2D debug view")
    print("  test - Test all moves and their inverses")
    print("  analyze <move> - Analyze what a specific move does")
    print("  help - Show this help")
    print("  quit - Exit")
    print()
    
    # Initialize cube and renderer
    cube = CubeSimulator()
    renderer = CubeRenderer(mode=RenderMode.BOTH, show_numbers=False)
    
    print("Cube initialized in solved state.")
    print("Type 'help' to see available commands.")
    print()
    
    while True:
        try:
            command = input("cube> ").strip()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'quit' or cmd == 'exit' or cmd == 'q':
                print("Goodbye!")
                break
            
            elif cmd == 'help' or cmd == 'h':
                print("Commands:")
                print("  F, F', B, B', R, R', L, L', U, U', D, D' - Apply moves")
                print("  reset - Reset to solved state")
                print("  scramble [n] - Scramble with n moves (default 20)")
                print("  undo - Undo last move")
                print("  show - Show current state info")
                print("  save <filename> - Save current state as image")
                print("  debug - Show 2D debug view")
                print("  test - Test all moves and their inverses")
                print("  analyze <move> - Analyze what a specific move does")
                print("  help - Show this help")
                print("  quit - Exit")
                
            elif cmd == 'reset':
                cube.reset()
                print("Cube reset to solved state.")
                
            elif cmd == 'scramble':
                n_moves = 20
                if len(parts) > 1:
                    try:
                        n_moves = int(parts[1])
                    except ValueError:
                        print("Invalid number of moves. Using default 20.")
                
                cube.scramble(n_moves)
                print(f"Cube scrambled with {n_moves} moves.")
                
            elif cmd == 'undo':
                if cube.undo_last_move():
                    print("Last move undone.")
                else:
                    print("Nothing to undo.")
                    
            elif cmd == 'show':
                result = renderer.render(cube.get_state())
                symbolic = result['symbolic']
                print(f"Solved: {symbolic['overall']['is_solved']}")
                print(f"Solved faces: {symbolic['overall']['solved_faces']}/6")
                print(f"Move count: {cube.get_move_count()}")
                if cube.get_action_history():
                    print(f"Recent moves: {' '.join(cube.get_action_history()[-10:])}")
                    
            elif cmd == 'save':
                if len(parts) < 2:
                    filename = 'cube_state.png'
                else:
                    filename = parts[1]
                    if not filename.endswith('.png'):
                        filename += '.png'
                
                renderer.save_image(cube.get_state(), filename)
                print(f"Cube state saved as {filename}")
                
            elif cmd == 'debug':
                print("Creating 2D debug view...")
                fig = create_2d_debug_view(cube.get_state())
                plt.show()
                plt.close(fig)
                
            elif cmd == 'test':
                test_all_moves(cube)
                cube.reset()  # Reset after testing
                
            elif cmd == 'analyze':
                if len(parts) < 2:
                    print("Usage: analyze <move>")
                else:
                    move = parts[1].upper()
                    cube.reset()  # Start from solved state
                    analyze_move(cube, move)
                    cube.reset()  # Reset after analysis
                
            elif cmd in ['f', 'f\'', 'b', 'b\'', 'r', 'r\'', 'l', 'l\'', 
                        'u', 'u\'', 'd', 'd\'']:
                move = cmd.upper()
                if cube.apply_move(move):
                    result = renderer.render(cube.get_state())
                    symbolic = result['symbolic']
                    print(f"Applied {move}. Solved faces: {symbolic['overall']['solved_faces']}/6")
                else:
                    print(f"Invalid move: {move}")
                    
            else:
                # Try to parse as a move
                move = command.upper()
                if cube.apply_move(move):
                    result = renderer.render(cube.get_state())
                    symbolic = result['symbolic']
                    print(f"Applied {move}. Solved faces: {symbolic['overall']['solved_faces']}/6")
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands.")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main() 