import pytest
from environment.action_space import ActionSpace, ActionType, CubeAction, ViewAction, SpecialAction

def test_action_space_basic():
    action_space = ActionSpace(include_view_actions=True, include_special_actions=True)
    assert len(action_space) == len(CubeAction) + len(ViewAction) + len(SpecialAction)
    # 测试动作索引和反查
    for idx in range(len(action_space)):
        action_type, action_name = action_space.get_action(idx)
        idx2 = action_space.get_action_idx(action_type, action_name)
        assert idx == idx2
    # 测试 cube/view/special 动作获取
    cube_actions = action_space.get_cube_actions()
    view_actions = action_space.get_view_actions()
    special_actions = action_space.get_special_actions()
    assert set(cube_actions) == set([a.value for a in CubeAction])
    assert set(view_actions) == set([a.value for a in ViewAction])
    assert set(special_actions) == set([a.value for a in SpecialAction])
    # 测试 gym space
    gym_space = action_space.to_gym_space()
    assert gym_space.n == len(action_space) 