"""
View management utilities for 3D cube rendering.
"""

import numpy as np
from typing import List, Tuple, Dict
from enum import Enum


class ViewType(Enum):
    """Types of view perspectives"""
    VERTEX = "vertex"  # 8 vertices × 3 rotations = 24 views
    FACE = "face"      # 6 faces × 4 rotations = 24 views


# Cube vertices (8 corners)
CUBE_VERTICES = [
    (-1, -1, -1),  # 0: bottom-left-back
    (1, -1, -1),   # 1: bottom-right-back
    (1, 1, -1),    # 2: top-right-back
    (-1, 1, -1),   # 3: top-left-back
    (-1, -1, 1),   # 4: bottom-left-front
    (1, -1, 1),    # 5: bottom-right-front
    (1, 1, 1),     # 6: top-right-front
    (-1, 1, 1)     # 7: top-left-front
]

# Face centers and their normals
FACE_CENTERS_AND_NORMALS = [
    ((0, 0, 1), (0, 0, 1)),    # 0: FRONT
    ((0, 0, -1), (0, 0, -1)),  # 1: BACK
    ((-1, 0, 0), (-1, 0, 0)),  # 2: LEFT
    ((1, 0, 0), (1, 0, 0)),    # 3: RIGHT
    ((0, 1, 0), (0, 1, 0)),    # 4: UP
    ((0, -1, 0), (0, -1, 0))   # 5: DOWN
]


def get_vertex_views() -> List[Dict]:
    """
    Generate all 24 vertex views (8 vertices × 3 rotations each).
    
    Returns:
        List of view dictionaries with position, rotation info
    """
    views = []
    
    for vertex_id, vertex_pos in enumerate(CUBE_VERTICES):
        # For each vertex, generate 3 rotations along the 3 axes connected to it
        # The 3 axes for a vertex are the 3 edges connected to that vertex
        
        # Get the 3 edge directions from this vertex
        edge_directions = _get_vertex_edge_directions(vertex_id)
        
        for rotation_id, axis in enumerate(edge_directions):
            view_id = vertex_id * 3 + rotation_id
            
            views.append({
                'view_id': view_id,
                'view_type': ViewType.VERTEX,
                'vertex_id': vertex_id,
                'vertex_pos': vertex_pos,
                'rotation_id': rotation_id,
                'rotation_axis': axis,
                'rotation_angle': 120.0,  # 120 degrees as specified
                'camera_pos': np.array(vertex_pos) * 5,  # Move camera away from cube
                'look_at': (0, 0, 0),  # Look at cube center
                'up_vector': _get_up_vector(axis)
            })
    
    return views


def get_face_views() -> List[Dict]:
    """
    Generate all 24 face views (6 faces × 4 rotations each).
    
    Returns:
        List of view dictionaries with position, rotation info
    """
    views = []
    
    for face_id, (face_center, face_normal) in enumerate(FACE_CENTERS_AND_NORMALS):
        # For each face, generate 4 rotations (0°, 90°, 180°, 270°)
        for rotation_id in range(4):
            view_id = face_id * 4 + rotation_id
            rotation_angle = rotation_id * 90.0
            
            views.append({
                'view_id': view_id,
                'view_type': ViewType.FACE,
                'face_id': face_id,
                'face_center': face_center,
                'face_normal': face_normal,
                'rotation_id': rotation_id,
                'rotation_angle': rotation_angle,
                'camera_pos': np.array(face_normal) * 5,  # Move camera away from cube
                'look_at': (0, 0, 0),  # Look at cube center
                'up_vector': _get_face_up_vector(face_normal, rotation_angle)
            })
    
    return views


def get_view_neighbors(view_type: ViewType) -> Dict[int, List[int]]:
    """
    Get neighbor relationships for view transitions.
    
    Args:
        view_type: Type of view (VERTEX or FACE)
        
    Returns:
        Dictionary mapping view_id to list of 3 (vertex) or 4 (face) neighbor view_ids
    """
    if view_type == ViewType.VERTEX:
        return _get_vertex_neighbors()
    elif view_type == ViewType.FACE:
        return _get_face_neighbors()
    else:
        raise ValueError(f"Unknown view type: {view_type}")


def _get_vertex_edge_directions(vertex_id: int) -> List[Tuple[float, float, float]]:
    """
    Get the 3 edge directions connected to a vertex.
    
    Args:
        vertex_id: Vertex index (0-7)
        
    Returns:
        List of 3 normalized direction vectors
    """
    vertex_pos = np.array(CUBE_VERTICES[vertex_id])
    
    # The 3 edges from a vertex go to the 3 adjacent vertices
    # For a cube, each vertex connects to exactly 3 other vertices
    adjacent_vertices = []
    
    for other_id, other_pos in enumerate(CUBE_VERTICES):
        if other_id == vertex_id:
            continue
        
        # Check if vertices are adjacent (Manhattan distance = 2 in cube coordinates)
        diff = np.array(other_pos) - vertex_pos
        if np.sum(np.abs(diff)) == 2:  # Adjacent vertices differ by 2 in exactly one coordinate
            adjacent_vertices.append(other_pos)
    
    # Convert to edge directions (normalized)
    edge_directions = []
    for adj_pos in adjacent_vertices:
        direction = np.array(adj_pos) - vertex_pos
        direction = direction / np.linalg.norm(direction)
        edge_directions.append(tuple(direction))
    
    return edge_directions


def _get_up_vector(axis: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Get an appropriate up vector for a given rotation axis.
    
    Args:
        axis: Rotation axis vector
        
    Returns:
        Up vector perpendicular to the axis
    """
    axis_vec = np.array(axis)
    
    # Choose a vector that's not parallel to the axis
    if abs(axis_vec[2]) < 0.9:
        up_candidate = np.array([0, 0, 1])
    else:
        up_candidate = np.array([0, 1, 0])
    
    # Make it perpendicular to the axis
    up_vec = up_candidate - np.dot(up_candidate, axis_vec) * axis_vec
    up_vec = up_vec / np.linalg.norm(up_vec)
    
    return tuple(up_vec)


def _get_face_up_vector(face_normal: Tuple[float, float, float], rotation_angle: float) -> Tuple[float, float, float]:
    """
    Get up vector for face view with rotation.
    
    Args:
        face_normal: Face normal vector
        rotation_angle: Rotation angle in degrees
        
    Returns:
        Rotated up vector
    """
    normal_vec = np.array(face_normal)
    
    # Choose initial up vector
    if abs(normal_vec[1]) < 0.9:
        initial_up = np.array([0, 1, 0])
    else:
        initial_up = np.array([1, 0, 0])
    
    # Make it perpendicular to normal
    up_vec = initial_up - np.dot(initial_up, normal_vec) * normal_vec
    up_vec = up_vec / np.linalg.norm(up_vec)
    
    # Rotate around the normal by the specified angle
    angle_rad = np.radians(rotation_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Rodrigues' rotation formula
    rotated_up = (up_vec * cos_a + 
                  np.cross(normal_vec, up_vec) * sin_a + 
                  normal_vec * np.dot(normal_vec, up_vec) * (1 - cos_a))
    
    return tuple(rotated_up)


def _get_vertex_neighbors() -> Dict[int, List[int]]:
    """
    Get neighbor relationships for vertex views.
    Each vertex view can transition to 3 neighbors.
    
    Returns:
        Dictionary mapping view_id to list of 3 neighbor view_ids
    """
    # Placeholder - you will fill this based on the cube topology
    neighbors = {}
    
    for view_id in range(24):  # 8 vertices × 3 rotations
        # For now, create empty neighbor lists
        # You will fill these based on the actual topology
        neighbors[view_id] = [
            (view_id + 1) % 24,  # Placeholder
            (view_id + 2) % 24,  # Placeholder  
            (view_id + 3) % 24   # Placeholder
        ]
    
    return neighbors


def _get_face_neighbors() -> Dict[int, List[int]]:
    """
    Get neighbor relationships for face views.
    Each face view can transition to 4 neighbors.
    
    Returns:
        Dictionary mapping view_id to list of 4 neighbor view_ids
    """
    # Placeholder - you will fill this based on the cube topology
    neighbors = {}
    
    for view_id in range(24):  # 6 faces × 4 rotations
        # For now, create empty neighbor lists
        # You will fill these based on the actual topology
        neighbors[view_id] = [
            (view_id + 1) % 24,  # Placeholder
            (view_id + 2) % 24,  # Placeholder
            (view_id + 3) % 24,  # Placeholder
            (view_id + 4) % 24   # Placeholder
        ]
    
    return neighbors 