"""
Cube simulator for the CubeBench environment.
Implements a 3x3x3 Rubik's cube with state management and move validation.
Uses a 54-element state array with explicit permutation mappings.

Expanded state representation:

                    36 37 38
                    39 40 41
                    42 43 44
         18 19 20   0  1  2   27 28 29 
         21 22 23   3  4  5   30 31 32
         24 25 26   6  7  8   33 34 35
                   45 46 47
                   48 49 50
                   51 52 53
                    9 10 11
                   12 13 14
                   15 16 17                
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import random


State = np.ndarray

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
    FACE_TO_STATE = {"FRONT": 0, "BACK": 1, "LEFT": 2, "RIGHT": 3, "UP": 4, "DOWN": 5}
    STATE_TO_FACE = {0: "FRONT", 1: "BACK", 2: "LEFT", 3: "RIGHT", 4: "UP", 5: "DOWN"}
    
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
    @property
    def MOVE_CYCLES(cls):
        return {
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
        self.state: State = np.zeros(54, dtype=int)
        
        # Set face colors
        self.state[0:9] = 0
        self.state[9:18] = 1
        self.state[18:27] = 2
        self.state[27:36] = 3
        self.state[36:45] = 4
        self.state[45:54] = 5
        
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
    
    def is_solved(self) -> bool:
        """Check if the cube is solved"""
        return np.all(self.state == np.array([0] * 9 + [1] * 9 + [2] * 9 + [3] * 9 + [4] * 9 + [5] * 9)).item()
    
    def get_state(self) -> State:
        """Get current cube state as 54-element array"""
        return self.state.copy()
    
    def set_state(self, state: State):
        """Set cube state from 54-element array"""
        if len(state) == 54:
            self.state = state.copy()
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
        return f"CubeSimulator(moves={self.get_move_count()})" 