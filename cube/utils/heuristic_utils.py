import numpy as np
from cube.utils.face_utils import get_standard_faces, get_against_face, FULL_NEIGHBOR_MAP

def step_1_the_cross(state: np.ndarray) -> int:
    """Checks for a completed cross on any face."""
    score = 0
    for face_name, face_value in get_standard_faces().items():
        # Check if the 4 edge pieces on the face match the center color
        if all(state[face_value[i]] == state[face_value[4]] for i in [1, 3, 5, 7]):
            # Check if the adjacent edge pieces match their respective face centers
            neighbor_faces = FULL_NEIGHBOR_MAP[face_value]  # left, up, right, down
            if (state[neighbor_faces[0][5]] == state[neighbor_faces[0][4]] and
                state[neighbor_faces[1][7]] == state[neighbor_faces[1][4]] and
                state[neighbor_faces[2][3]] == state[neighbor_faces[2][4]] and
                state[neighbor_faces[3][1]] == state[neighbor_faces[3][4]]):
                score += 1
    return score

def step_2_the_corners(state: np.ndarray) -> int:
    """Checks for a completed first layer (cross and corners) on any face."""
    score = 0
    for face_name, face_value in get_standard_faces().items():
        # Check if the entire face is solved
        if all(state[i] == state[face_value[4]] for i in face_value):
            # Check if the first layer of neighboring faces is correct
            neighbor_faces = FULL_NEIGHBOR_MAP[face_value]
            if (all(state[neighbor_faces[0][i]] == state[neighbor_faces[0][4]] for i in [2, 5, 8]) and
                all(state[neighbor_faces[1][i]] == state[neighbor_faces[1][4]] for i in [6, 7, 8]) and
                all(state[neighbor_faces[2][i]] == state[neighbor_faces[2][4]] for i in [0, 3, 6]) and
                all(state[neighbor_faces[3][i]] == state[neighbor_faces[3][4]] for i in [0, 1, 2])):
                score += 1
    return score

def step_3_the_second_layer(state: np.ndarray) -> int:
    """Checks if the first two layers are solved for any base face."""
    score = 0
    # A fully solved face indicates step 2 is done for that face.
    step2_done_faces = {face_name for face_name, is_done in solved_faces(state).items() if is_done}

    for face_name in step2_done_faces:
        neighbor_faces = FULL_NEIGHBOR_MAP[get_standard_faces()[face_name]]
        # Check middle layer edges
        if (state[neighbor_faces[0][1]] == state[neighbor_faces[0][4]] and
            state[neighbor_faces[0][7]] == state[neighbor_faces[0][4]] and
            state[neighbor_faces[1][3]] == state[neighbor_faces[1][4]] and
            state[neighbor_faces[1][5]] == state[neighbor_faces[1][4]] and
            state[neighbor_faces[2][1]] == state[neighbor_faces[2][4]] and
            state[neighbor_faces[2][7]] == state[neighbor_faces[2][4]] and
            state[neighbor_faces[3][3]] == state[neighbor_faces[3][4]] and
            state[neighbor_faces[3][5]] == state[neighbor_faces[3][4]]):
            score +=1
    return score

def step_4_the_last_layer_cross(state: np.ndarray) -> int:
    """Checks if the top face has a cross, given F2L is solved."""
    score = 0
    if step_3_the_second_layer(state) > 0: # Only check if F2L is done for at least one side
        for face_name, face_value in get_standard_faces().items():
            against_face_name = get_against_face(face_name)
            against_face_value = get_standard_faces()[against_face_name]
            # Check for cross on the opposite face
            if all(state[against_face_value[i]] == state[against_face_value[4]] for i in [1, 3, 5, 7]):
                score += 1
    return score

def step_5_the_last_layer_edges(state: np.ndarray) -> int:
    """Checks if the last layer edges are aligned."""
    score = 0
    if step_4_the_last_layer_cross(state) > 0:
        for face_name, face_value in get_standard_faces().items():
            against_face_name = get_against_face(face_name)
            neighbor_faces = FULL_NEIGHBOR_MAP[get_standard_faces()[against_face_name]]
            # Check that top-facing edges of neighbor faces match neighbor center color
            if (state[neighbor_faces[0][1]] == state[neighbor_faces[0][4]] and
                state[neighbor_faces[1][1]] == state[neighbor_faces[1][4]] and
                state[neighbor_faces[2][1]] == state[neighbor_faces[2][4]] and
                state[neighbor_faces[3][1]] == state[neighbor_faces[3][4]]):
                score += 1
    return score


def step_6_the_last_layer_corners_placed(state: np.ndarray) -> int:
    """Checks if the last layer corners are in the correct position (not necessarily oriented)."""
    score = 0
    if step_5_the_last_layer_edges(state) > 0:
        # This is a complex check, simplified here: we check if the cube is solved except for corner twists.
        # A simpler proxy is to check if all but the last layer are solved.
        num_correct = np.sum(state[:45] == np.floor(np.arange(45) / 9))
        if num_correct >= 36: # Allow for some error, but check if first 2 layers are mostly done
             score += 1 # Simplified score
    return score


def step_7_the_last_layer_corners_oriented(state: np.ndarray) -> int:
    """Checks if the last layer corners are correctly oriented (i.e., cube is solved)."""
    if np.all(state == np.floor(np.arange(54) / 9)):
        return 1
    return 0


# Helper from original file
def solved_faces(state: np.ndarray) -> dict:
    """
    Get the number of solved faces of the cube.
    """
    ret = {}
    for face_name, face_indices in get_standard_faces().items():
        ret[face_name] = all(state[i] == state[face_indices[4]] for i in face_indices)
    return ret 