import numpy as np
from environment.utils.cube_simulator import CubeSimulator


def test_cube_scramble():
    cube = CubeSimulator()
    cube.scramble(num_moves=10)
    # 打乱后不应是还原状态
    assert not cube.is_solved() or cube.get_solved_faces_count() < 6

def test_cube_apply_move_and_undo():
    cube = CubeSimulator()
    state_before = cube.get_state()
    cube.apply_move('F')
    assert not np.array_equal(cube.get_state(), state_before)
    cube.undo_last_move()
    assert np.array_equal(cube.get_state(), state_before)

def test_cube_action_history():
    cube = CubeSimulator()
    cube.apply_move('F')
    cube.apply_move('U')
    history = cube.get_action_history()
    assert history == ['F', 'U']
    cube.undo_last_move()
    assert cube.get_action_history() == ['F'] 

def test_cube_action_consistency():
    cube = CubeSimulator()
    cube.apply_move('F')
    cube.apply_move('U')
    cube.apply_move('U\'')
    cube.apply_move('F\'')
    cube.apply_move('R')
    cube.apply_move('R\'')
    cube.apply_move('F')
    cube.apply_move('F\'')
    cube.apply_move('U')
    cube.apply_move('U\'')
    cube.apply_move('R')
    cube.apply_move('R\'')
    assert cube.is_solved()
