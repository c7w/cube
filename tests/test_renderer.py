import numpy as np
from environment.renderer import CubeRenderer, RenderMode, render_cube_image, render_cube_symbolic

def test_symbolic_render():
    # 构造一个还原魔方
    cube_state = np.zeros((6, 9), dtype=int)
    for i in range(6):
        cube_state[i, :] = i
    renderer = CubeRenderer(RenderMode.SYMBOLIC)
    symbolic = renderer.render(cube_state)
    assert 'symbolic' in symbolic
    for face in ['FRONT', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN']:
        assert face in symbolic['symbolic']
        assert symbolic['symbolic'][face]['solved']
    assert symbolic['symbolic']['overall']['is_solved']

def test_image_render():
    cube_state = np.zeros((6, 9), dtype=int)
    for i in range(6):
        cube_state[i, :] = i
    img = render_cube_image(cube_state, image_size=100)
    assert img.shape == (100, 100, 3)
    assert img.dtype == np.uint8

def test_render_to_string():
    cube_state = np.zeros((6, 9), dtype=int)
    for i in range(6):
        cube_state[i, :] = i
    renderer = CubeRenderer(RenderMode.SYMBOLIC)
    s = renderer.render_to_string(cube_state)
    assert isinstance(s, str)
    assert 'Cube State:' in s 