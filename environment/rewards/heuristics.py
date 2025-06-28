# https://solvethecube.com/
from typing import List, Tuple, Dict

# 定义一个面的数据类型别名
Face = Tuple[int, ...]
# 定义相邻面列表的数据类型别名
NearFaces = List[Face]
# 定义最终映射表的数据类型别名
FaceMap = Dict[Face, NearFaces]

def _rotate_90_degrees(key_face: Face, near_faces: NearFaces) -> Tuple[Face, NearFaces]:
    """
    根据基准面和其相邻面，计算顺时针旋转90度后的新状态。

    这个函数是所有逻辑的核心。

    Args:
        key_face: 旋转前的基准面。
        near_faces: 旋转前基准面的相邻面 [left, up, right, down]。

    Returns:
        一个元组，包含旋转后的新基准面和新的相邻面列表。
    """

    def _rotate_face_once(face: Face) -> Face:
        """
        对单个面（一个9元素的元组）进行一次顺时针90度旋转。
        规律:
        [0, 1, 2,      [6, 3, 0,
         3, 4, 5,  ->   7, 4, 1,
         6, 7, 8]       8, 5, 2]
        """
        return (face[6], face[3], face[0], face[7], face[4], face[1], face[8], face[5], face[2])

    # 1. 旋转基准面本身
    new_key_face = _rotate_face_once(key_face)

    # 2. 相邻面进行轮转: [下, 左, 上, 右]，并且每个面自身也旋转
    old_left, old_up, old_right, old_down = near_faces
    
    new_near_faces = [
        _rotate_face_once(old_down),   # 新的左面是旋转后的旧的下面
        _rotate_face_once(old_left),   # 新的上面是旋转后的旧的左面
        _rotate_face_once(old_up),     # 新的右面是旋转后的旧的上面
        _rotate_face_once(old_right),  # 新的下面是旋转后的旧的右面
    ]

    return new_key_face, new_near_faces

def _generate_full_face_map() -> FaceMap:
    """
    从6个基础锚点状态生成完整的24个旋转状态映射表。
    """
    # 步骤 1: 定义6个面的“锚点”状态。
    # key是面的表示，value是它相邻的[左, 上, 右, 下]四个面。
    # 注意：为了可以作为字典的键，我们使用元组(tuple)而不是列表(list)。
    anchor_states: Dict[str, Tuple[Face, NearFaces]] = {
        "front": (
            (0, 1, 2, 3, 4, 5, 6, 7, 8), 
            [(18, 19, 20, 21, 22, 23, 24, 25, 26), (36, 37, 38, 39, 40, 41, 42, 43, 44), (27, 28, 29, 30, 31, 32, 33, 34, 35), (45, 46, 47, 48, 49, 50, 51, 52, 53)]
        ),
        "back": (
            (17, 16, 15, 14, 13, 12, 11, 10, 9), 
            [(27, 28, 29, 30, 31, 32, 33, 34, 35), (44, 43, 42, 41, 40, 39, 38, 37, 36), (18, 19, 20, 21, 22, 23, 24, 25, 26), (53, 52, 51, 50, 49, 48, 47, 46, 45)]
        ),
        "left": (
            (18, 19, 20, 21, 22, 23, 24, 25, 26), 
            [(17, 16, 15, 14, 13, 12, 11, 10, 9), (38, 41, 44, 37, 40, 43, 36, 39, 42), (0, 1, 2, 3, 4, 5, 6, 7, 8), (51, 48, 45, 52, 49, 46, 53, 50, 47)]
        ),
        "right": (
            (27, 28, 29, 30, 31, 32, 33, 34, 35), 
            [(0, 1, 2, 3, 4, 5, 6, 7, 8), (42, 39, 36, 43, 40, 37, 44, 41, 38), (17, 16, 15, 14, 13, 12, 11, 10, 9), (47, 50, 53, 46, 49, 52, 45, 48, 51)]
        ),
        "up": (
            (36, 37, 38, 39, 40, 41, 42, 43, 44), 
            [(24, 21, 18, 25, 22, 19, 26, 23, 20), (9, 10, 11, 12, 13, 14, 15, 16, 17), (29, 32, 35, 28, 31, 34, 27, 30, 33), (0, 1, 2, 3, 4, 5, 6, 7, 8)]
        ),
        "down": (
            (45, 46, 47, 48, 49, 50, 51, 52, 53), 
            [(20, 23, 26, 19, 22, 25, 18, 21, 24), (0, 1, 2, 3, 4, 5, 6, 7, 8), (33, 30, 27, 34, 31, 28, 35, 32, 29), (9, 10, 11, 12, 13, 14, 15, 16, 17)]
        ),
    }

    # 步骤 2: 循环生成所有状态
    full_map: FaceMap = {}
    for face_name, (base_key, base_near_faces) in anchor_states.items():
        # print(f"--- Generating states for {face_name.upper()} face ---")  # 注释掉调试打印
        current_key = base_key
        # 将内部的list也转为tuple
        current_near_faces = [tuple(f) for f in base_near_faces]

        # 每个锚点生成4个旋转状态 (0, 90, 180, 270度)
        for i in range(4):
            # print(f"  Rotation {i*90} degrees...")  # 注释掉调试打印
            full_map[current_key] = current_near_faces
            # 计算下一个旋转状态
            current_key, current_near_faces = _rotate_90_degrees(current_key, current_near_faces)
    
    return full_map

FULL_NEIGHBOR_MAP = _generate_full_face_map()
# For debug
# for key, value in FULL_NEIGHBOR_MAP.items():
#     print(f">> {key}")
#     print(f"  Left: {value[0]}")
#     print(f"  Up: {value[1]}")
#     print(f"  Right: {value[2]}")
#     print(f"  Down: {value[3]}")


def _get_face_colors(state: str) -> dict:
    """
    Get the colors of the faces of the cube.
    """
    return {
        "FRONT": state[4],
        "BACK": state[13],
        "LEFT": state[22],
        "RIGHT": state[31],
        "UP": state[40],
        "DOWN": state[49]
    }

def _get_standard_face() -> Dict[str, Face]:
    """
    Get the standard face of the cube.
    """
    return {
        "FRONT": (0,1,2,3,4,5,6,7,8),
        "BACK": (17,16,15,14,13,12,11,10,9),
        "LEFT": (18,19,20,21,22,23,24,25,26),
        "RIGHT": (27,28,29,30,31,32,33,34,35),
        "UP": (36,37,38,39,40,41,42,43,44),
        "DOWN": (45,46,47,48,49,50,51,52,53)
    }

def _get_against_face(face_name: str) -> str:
    """
    Get the against face of the cube.
    """
    return {
        "FRONT": "BACK",
        "BACK": "FRONT",
        "LEFT": "RIGHT",
        "RIGHT": "LEFT",
        "UP": "DOWN",
        "DOWN": "UP"
    }[face_name]

def step_1_the_cross(state: str) -> dict:
    """
    Step 1: The Cross
    (1) Current face: face[1,3,5,7] == face[4]
    (2) Neighbor face (the following conditions must be all met)
        Left: face[5] == face[4]
        Up: face[7] == face[4]
        Right: face[3] == face[4]
        Down: face[1] == face[4]
    """
    ret = {"FRONT": False, "BACK": False, "LEFT": False, "RIGHT": False, "UP": False, "DOWN": False}
    for face_name, face_value in _get_standard_face().items():
        current_face_condition = all([state[face_value[i]] == state[face_value[4]] for i in [1,3,5,7]])
        neighbor_faces = FULL_NEIGHBOR_MAP[face_value]  # left, up, right, down
        current_face_condition &= state[neighbor_faces[0][5]] == state[neighbor_faces[0][4]]
        current_face_condition &= state[neighbor_faces[1][7]] == state[neighbor_faces[1][4]]
        current_face_condition &= state[neighbor_faces[2][3]] == state[neighbor_faces[2][4]]
        current_face_condition &= state[neighbor_faces[3][1]] == state[neighbor_faces[3][4]]
        ret[face_name] = current_face_condition
    return ret

def step_2_the_corners(state: str) -> dict:
    """
    Step 2: The Corners
    (1) Current face: face[0,1,2,3,4,5,6,7,8] == face[4]
    (2) Neighbor face (the following conditions must be all met)
        Left: face[2,5,8] == face[4]
        Up: face[6,7,8] == face[4] 
        Right: face[0,3,6] == face[4]
        Down: face[0,1,2] == face[4]
    """
    ret = {"FRONT": False, "BACK": False, "LEFT": False, "RIGHT": False, "UP": False, "DOWN": False}
    for face_name, face_value in _get_standard_face().items():
        current_face_condition = all([state[face_value[i]] == state[face_value[4]] for i in [0,1,2,3,4,5,6,7,8]])
        neighbor_faces = FULL_NEIGHBOR_MAP[face_value]  # left, up, right, down
        current_face_condition &= all([state[neighbor_faces[0][i]] == state[neighbor_faces[0][4]] for i in [2,5,8]])
        current_face_condition &= all([state[neighbor_faces[1][i]] == state[neighbor_faces[1][4]] for i in [6,7,8]])
        current_face_condition &= all([state[neighbor_faces[2][i]] == state[neighbor_faces[2][4]] for i in [0,3,6]])
        current_face_condition &= all([state[neighbor_faces[3][i]] == state[neighbor_faces[3][4]] for i in [0,1,2]])
        ret[face_name] = current_face_condition
    return ret

def step_3_the_second_layer(state: str) -> dict:
    """
    Step 3: The Second Layer
    (1) Current face: face[1,2,3,4,5,6,7,8] == face[4]
    (2) Neighbor face (the following conditions must be all met)
        Left: face[1,2,5,7,8] == face[4]
        Up: face[3,5,6,7,8] == face[4] 
        Right: face[0,1,3,6,7] == face[4]
        Down: face[0,1,2,3,5] == face[4]
    """
    ret = {"FRONT": False, "BACK": False, "LEFT": False, "RIGHT": False, "UP": False, "DOWN": False}
    for face_name, face_value in _get_standard_face().items():
        current_face_condition = all([state[face_value[i]] == state[face_value[4]] for i in [1,2,3,4,5,6,7,8]])
        neighbor_faces = FULL_NEIGHBOR_MAP[face_value]  # left, up, right, down
        current_face_condition &= all([state[neighbor_faces[0][i]] == state[neighbor_faces[0][4]] for i in [1,2,5,7,8]])
        current_face_condition &= all([state[neighbor_faces[1][i]] == state[neighbor_faces[1][4]] for i in [3,5,6,7,8]])
        current_face_condition &= all([state[neighbor_faces[2][i]] == state[neighbor_faces[2][4]] for i in [0,1,3,6,7]])
        current_face_condition &= all([state[neighbor_faces[3][i]] == state[neighbor_faces[3][4]] for i in [0,1,2,3,5]])
        ret[face_name] = current_face_condition
    return ret

def step_4_the_last_layer_cross(state: str) -> dict:
    """
    Step 4: The Last Layer Cross
    (1) Satisfies step_3_the_second_layer
    (2) The against face must form a cross (face[1,3,5,7] == face[4])
    """
    ret = step_3_the_second_layer(state)
    for face_name, _ in _get_standard_face().items():
        against_face = _get_against_face(face_name)
        against_face_value = _get_standard_face()[against_face]
        ret[face_name] &= all([state[against_face_value[i]] == state[against_face_value[4]] for i in [1,3,5,7]])
    return ret

def step_5_the_last_layer_edges(state: str) -> dict:
    """
    Step 5: The Last Layer Corners
    (1) Current face: face[1,2,3,4,5,6,7,8] == face[4]
    (2) Neighbor face (the following conditions must be all met)
        Left: face[1,2,3,5,7,8] == face[4]
        Up: face[1,3,5,6,7,8] == face[4] 
        Right: face[0,1,3,5,6,7] == face[4]
        Down: face[0,1,2,3,5,7] == face[4]
    (3) The against face must form a cross (face[1,3,5,7] == face[4])
    """
    ret = {"FRONT": False, "BACK": False, "LEFT": False, "RIGHT": False, "UP": False, "DOWN": False}
    for face_name, face_value in _get_standard_face().items():
        current_face_condition = all([state[face_value[i]] == state[face_value[4]] for i in [1,2,3,4,5,6,7,8]])
        neighbor_faces = FULL_NEIGHBOR_MAP[face_value]  # left, up, right, down
        current_face_condition &= all([state[neighbor_faces[0][i]] == state[neighbor_faces[0][4]] for i in [1,2,3,5,7,8]])
        current_face_condition &= all([state[neighbor_faces[1][i]] == state[neighbor_faces[1][4]] for i in [1,3,5,6,7,8]])
        current_face_condition &= all([state[neighbor_faces[2][i]] == state[neighbor_faces[2][4]] for i in [0,1,3,5,6,7]])
        current_face_condition &= all([state[neighbor_faces[3][i]] == state[neighbor_faces[3][4]] for i in [0,1,2,3,5,7]])
        ret[face_name] = current_face_condition
    for face_name, _ in _get_standard_face().items():
        against_face = _get_against_face(face_name)
        against_face_value = _get_standard_face()[against_face]
        ret[face_name] &= all([state[against_face_value[i]] == state[against_face_value[4]] for i in [1,3,5,7]])
    return ret

def step_6_the_last_layer_corners(state: str) -> dict:
    """
    Step 6: The Last Layer Corners
    (1) Satisfies step_5_the_last_layer_edges
    (2) For each neight face (say, left face), the following conditions must be all met
        left_face[0] in [left_face[4], up_face[4], back_face[4]
        left_face[6] in [left_face[4], down_face[4], back_face[4]
    (say, up face)
        up_face[0] in [up_face[4], left_face[4], back_face[4]
        up_face[2] in [up_face[4], right_face[4], back_face[4]
    (say, right face)
        right_face[2] in [right_face[4], up_face[4], back_face[4]
        right_face[8] in [right_face[4], down_face[4], back_face[4]
    (say, down face)
        down_face[6] in [down_face[4], left_face[4], back_face[4]
        down_face[8] in [down_face[4], right_face[4], back_face[4]
    """
    ret = step_5_the_last_layer_edges(state)
    for face_name, face_value in _get_standard_face().items():
        neighbor_faces = FULL_NEIGHBOR_MAP[face_value]  # left, up, right, down
        neighbor_left_face = neighbor_faces[0]
        neighbor_up_face = neighbor_faces[1]
        neighbor_right_face = neighbor_faces[2]
        neighbor_down_face = neighbor_faces[3]
        neighbor_back_face = _get_standard_face()[_get_against_face(face_name)]
        ret[face_name] &= state[neighbor_left_face[0]] in [state[neighbor_left_face[4]], state[neighbor_up_face[4]], state[neighbor_back_face[4]]]
        ret[face_name] &= state[neighbor_left_face[6]] in [state[neighbor_left_face[4]], state[neighbor_down_face[4]], state[neighbor_back_face[4]]]
        ret[face_name] &= state[neighbor_up_face[0]] in [state[neighbor_up_face[4]], state[neighbor_left_face[4]], state[neighbor_back_face[4]]]
        ret[face_name] &= state[neighbor_up_face[2]] in [state[neighbor_up_face[4]], state[neighbor_right_face[4]], state[neighbor_back_face[4]]]
        ret[face_name] &= state[neighbor_right_face[2]] in [state[neighbor_right_face[4]], state[neighbor_up_face[4]], state[neighbor_back_face[4]]]
        ret[face_name] &= state[neighbor_right_face[8]] in [state[neighbor_right_face[4]], state[neighbor_down_face[4]], state[neighbor_back_face[4]]]
        ret[face_name] &= state[neighbor_down_face[6]] in [state[neighbor_down_face[4]], state[neighbor_left_face[4]], state[neighbor_back_face[4]]]
        ret[face_name] &= state[neighbor_down_face[8]] in [state[neighbor_down_face[4]], state[neighbor_right_face[4]], state[neighbor_back_face[4]]]
    return ret

def solved_faces(state: str) -> dict:
    """
    Get the number of solved faces of the cube.
    """
    ret = {"FRONT": False, "BACK": False, "LEFT": False, "RIGHT": False, "UP": False, "DOWN": False}
    for face_name, face_indices in _get_standard_face().items():
        # 使用每个面自己的索引范围，而不是固定的 [0,1,2,3,4,5,6,7,8]
        ret[face_name] = all([state[face_indices[i]] == state[face_indices[4]] for i in [0,1,2,3,4,5,6,7,8]])
    return ret
