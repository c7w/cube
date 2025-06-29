"""
Face rendering utilities for CubeBench environments.
"""

import numpy as np
from typing import List, Tuple, Dict

# 定义一个面的数据类型别名
Face = Tuple[int, ...]
# 定义相邻面列表的数据类型别名
NearFaces = List[Face]
# 定义最终映射表的数据类型别名
FaceMap = Dict[Face, NearFaces]


def rotate_face_once(face: Face) -> Face:
    """
    Rotate a face 90 degrees clockwise.
    """
    return (face[6], face[3], face[0], face[7], face[4], face[1], face[8], face[5], face[2])


def rotate_90_degrees(key_face: Face, near_faces: NearFaces) -> Tuple[Face, NearFaces]:
    """
    根据基准面和其相邻面，计算顺时针旋转90度后的新状态。

    这个函数是所有逻辑的核心。

    Args:
        key_face: 旋转前的基准面。
        near_faces: 旋转前基准面的相邻面 [left, up, right, down]。

    Returns:
        一个元组，包含旋转后的新基准面和新的相邻面列表。
    """
    # 1. 旋转基准面本身
    new_key_face = rotate_face_once(key_face)

    # 2. 相邻面进行轮转: [下, 左, 上, 右]，并且每个面自身也旋转
    old_left, old_up, old_right, old_down = near_faces
    
    new_near_faces = [
        rotate_face_once(old_down),   # 新的左面是旋转后的旧的下面
        rotate_face_once(old_left),   # 新的上面是旋转后的旧的左面
        rotate_face_once(old_up),     # 新的右面是旋转后的旧的上面
        rotate_face_once(old_right),  # 新的下面是旋转后的旧的右面
    ]

    return new_key_face, new_near_faces

def generate_full_face_map() -> FaceMap:
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
            current_key, current_near_faces = rotate_90_degrees(current_key, current_near_faces)
    
    return full_map

FULL_NEIGHBOR_MAP = generate_full_face_map()


def get_face_colors(state: str) -> dict:
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

def get_standard_faces() -> Dict[str, Face]:
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

def get_against_face(face_name: str) -> str:
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