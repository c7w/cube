"""
Cube simulator for the CubeBench environment.
Implements a 3x3x3 Rubik's cube with state management and move validation.
Uses a 54-element state array with explicit permutation mappings.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import random


class Color(Enum):
    """Cube color enumeration"""
    WHITE = 0
    YELLOW = 1
    RED = 2
    ORANGE = 3
    BLUE = 4
    GREEN = 5


class CubeSimulator:
    """
    3x3x3 Rubik's cube simulator using 54-element state representation.
    
    State representation:
    - state: 54-element array where each element represents a square's color
    - Position mapping:
      * Face 0 (FRONT):  positions 0-8
      * Face 1 (BACK):   positions 9-17
      * Face 2 (LEFT):   positions 18-26
      * Face 3 (RIGHT):  positions 27-35
      * Face 4 (UP):     positions 36-44
      * Face 5 (DOWN):   positions 45-53
    
    Each face is arranged as:
    0 1 2
    3 4 5
    6 7 8
    """
    
    # Face starting positions
    FACE_POSITIONS = {
        'FRONT': 0,   # 0-8
        'BACK': 9,    # 9-17
        'LEFT': 18,   # 18-26
        'RIGHT': 27,  # 27-35
        'UP': 36,     # 36-44
        'DOWN': 45    # 45-53
    }
    
    # --- Cycle notation to permutation utility ---
    @staticmethod
    def cycles_to_permutation(cycles, size=54):
        perm = list(range(size))
        for cycle in cycles:
            n = len(cycle)
            for i in range(n):
                perm[cycle[i]] = cycle[(i-1)%n]
        return perm

    # --- Move cycles for each face (clockwise) ---
    MOVE_CYCLES = {
        'F': [  # right
            (0, 2, 8, 6), (1, 5, 7, 3),
            (42, 27, 47, 26), (43, 30, 46, 23), (44, 33, 45, 20)
        ],
        'B': [
            (9, 11, 17, 15), (10, 14, 16, 12),
            (36, 24, 53, 29), (37, 21, 52, 32), (38, 18, 51, 35)
        ],
        'L': [
            (18, 20, 26, 24), (19, 23, 25, 21),
            (36, 0, 45, 9), (39, 3, 48, 12), (42, 6, 51, 15)
        ],
        'R': [
            (27, 29, 35, 33), (28, 32, 34, 30),
            (2, 38, 11, 47), (5, 41, 14, 50), (8, 44, 17, 53)
        ],
        'U': [
            (36, 38, 44, 42), (37, 41, 43, 39),
            (15, 29, 2, 20), (16, 28, 1, 19), (17, 27, 0, 18)
        ],
        'D': [
            (45, 47, 53, 51), (46, 50, 52, 48),
            (6, 33, 11, 24), (7, 34, 10, 25), (8, 35, 9, 26)
        ]
    }

    # --- Permutations generated from cycles ---
    
    @property
    def PERMUTATIONS(self):
        return {move: self.cycles_to_permutation(cycles) for move, cycles in self.MOVE_CYCLES.items()}
    
    def __init__(self):
        """Initialize a solved cube"""
        self.reset()
    
    def reset(self):
        """Reset cube to solved state"""
        # Initialize with solved colors
        self.state = np.zeros(54, dtype=int)
        
        # Set face colors
        self.state[0:9] = Color.RED.value      # FRONT - Red
        self.state[9:18] = Color.ORANGE.value  # BACK - Orange
        self.state[18:27] = Color.BLUE.value   # LEFT - Blue
        self.state[27:36] = Color.GREEN.value  # RIGHT - Green
        self.state[36:45] = Color.YELLOW.value # UP - Yellow
        self.state[45:54] = Color.WHITE.value  # DOWN - White
        
        # Action history for undo functionality
        self.action_history = []
        self.state_history = []
    
    def scramble(self, num_moves: int = 20):
        """Scramble the cube with random moves"""
        moves = ['F', 'F\'', 'B', 'B\'', 'L', 'L\'', 'R', 'R\'', 'U', 'U\'', 'D', 'D\'']
        
        for _ in range(num_moves):
            move = random.choice(moves)
            self.apply_move(move)
        
        # Clear history after scramble
        self.action_history = []
        self.state_history = []
    
    def _get_inverse_permutation(self, perm: List[int]) -> List[int]:
        """Calculate the inverse of a permutation"""
        inverse = [0] * len(perm)
        for i, p in enumerate(perm):
            inverse[p] = i
        return inverse
    
    def apply_move(self, move: str) -> bool:
        """
        Apply a move to the cube.
        
        Args:
            move: Move notation (e.g., 'F', 'F\'', 'U', etc.)
            
        Returns:
            True if move was applied successfully, False otherwise
        """
        # Save current state for undo
        self.state_history.append(self.state.copy())
        
        # Parse move and apply permutation
        if move == 'F':
            self._apply_permutation('F')
        elif move == 'F\'':
            # Apply inverse permutation
            inverse_perm = self._get_inverse_permutation(self.PERMUTATIONS['F'])
            self._apply_specific_permutation(inverse_perm)
        elif move == 'B':
            self._apply_permutation('B')
        elif move == 'B\'':
            inverse_perm = self._get_inverse_permutation(self.PERMUTATIONS['B'])
            self._apply_specific_permutation(inverse_perm)
        elif move == 'L':
            self._apply_permutation('L')
        elif move == 'L\'':
            inverse_perm = self._get_inverse_permutation(self.PERMUTATIONS['L'])
            self._apply_specific_permutation(inverse_perm)
        elif move == 'R':
            self._apply_permutation('R')
        elif move == 'R\'':
            inverse_perm = self._get_inverse_permutation(self.PERMUTATIONS['R'])
            self._apply_specific_permutation(inverse_perm)
        elif move == 'U':
            self._apply_permutation('U')
        elif move == 'U\'':
            inverse_perm = self._get_inverse_permutation(self.PERMUTATIONS['U'])
            self._apply_specific_permutation(inverse_perm)
        elif move == 'D':
            self._apply_permutation('D')
        elif move == 'D\'':
            inverse_perm = self._get_inverse_permutation(self.PERMUTATIONS['D'])
            self._apply_specific_permutation(inverse_perm)
        else:
            return False
        
        # Record action
        self.action_history.append(move)
        return True
    
    def _apply_permutation(self, move: str):
        """Apply a single permutation to the cube state"""
        if move not in self.PERMUTATIONS:
            return
        
        perm = self.PERMUTATIONS[move]
        self._apply_specific_permutation(perm)
    
    def _apply_specific_permutation(self, perm: List[int]):
        """Apply a specific permutation to the cube state"""
        new_state = np.zeros_like(self.state)
        
        # Apply permutation: new_state[i] = old_state[perm[i]]
        for i in range(54):
            new_state[i] = self.state[perm[i]]
        
        self.state = new_state
    
    def get_face(self, face_name: str) -> np.ndarray:
        """Get a specific face as a 3x3 array"""
        if face_name not in self.FACE_POSITIONS:
            raise ValueError(f"Unknown face: {face_name}")
        
        start_pos = self.FACE_POSITIONS[face_name]
        face_data = self.state[start_pos:start_pos+9]
        return face_data.reshape(3, 3)
    
    def get_all_faces(self) -> Dict[str, np.ndarray]:
        """Get all faces as 3x3 arrays"""
        faces = {}
        for face_name in self.FACE_POSITIONS:
            faces[face_name] = self.get_face(face_name)
        return faces
    
    def is_solved(self) -> bool:
        """Check if the cube is solved"""
        for face_name in self.FACE_POSITIONS:
            face = self.get_face(face_name)
            center_color = face[1, 1]  # Center piece
            if not np.all(face == center_color):
                return False
        return True
    
    def get_solved_faces_count(self) -> int:
        """Count the number of solved faces"""
        solved_count = 0
        for face_name in self.FACE_POSITIONS:
            face = self.get_face(face_name)
            center_color = face[1, 1]
            if np.all(face == center_color):
                solved_count += 1
        return solved_count
    
    def get_state(self) -> np.ndarray:
        """Get current cube state as 54-element array"""
        return self.state.copy()
    
    def get_state_as_faces(self) -> np.ndarray:
        """Get current cube state as 6x9 array for compatibility"""
        faces = np.zeros((6, 9), dtype=int)
        face_order = ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN']
        
        for i, face_name in enumerate(face_order):
            start_pos = self.FACE_POSITIONS[face_name]
            faces[i] = self.state[start_pos:start_pos+9]
        
        return faces
    
    def set_state(self, state: np.ndarray):
        """Set cube state from 54-element array"""
        if len(state) == 54:
            self.state = state.copy()
        elif state.shape == (6, 9):
            # Convert from 6x9 format for compatibility
            face_order = ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN']
            for i, face_name in enumerate(face_order):
                start_pos = self.FACE_POSITIONS[face_name]
                self.state[start_pos:start_pos+9] = state[i]
        else:
            raise ValueError(f"Invalid state shape: {state.shape}")
    
    def undo_last_move(self) -> bool:
        """Undo the last move"""
        if not self.state_history:
            return False
        
        self.state = self.state_history.pop()
        if self.action_history:
            self.action_history.pop()
        return True
    
    def get_move_count(self) -> int:
        """Get the number of moves made since reset"""
        return len(self.action_history)
    
    def get_action_history(self) -> List[str]:
        """Get the history of moves"""
        return self.action_history.copy()
    
    def __repr__(self) -> str:
        solved_faces = self.get_solved_faces_count()
        return f"CubeSimulator(solved_faces={solved_faces}/6, moves={self.get_move_count()})" 