"""
Feasibility checking utilities for trajectories against obstacles.

This module provides functions to check if trajectory segments collide with
obstacles in the environment.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

from .collision_checker import (
    GripperGeometry, 
    SUCTION_GEOMETRY, 
    PARALLEL_JAW_GEOMETRY,
    aabb_overlap
)


@dataclass
class Cuboid:
    """Represents an obstacle as a cuboid in 3D space."""
    center: np.ndarray  # 3D center position [x, y, z]
    dimensions: np.ndarray  # [width, length, height] - full dimensions
    orientation: Optional[np.ndarray] = None  # 3x3 rotation matrix (optional, defaults to axis-aligned)
    
    def __post_init__(self):
        """Initialize orientation if not provided."""
        if self.orientation is None:
            self.orientation = np.eye(3)
        else:
            self.orientation = np.array(self.orientation)
        self.center = np.array(self.center)
        self.dimensions = np.array(self.dimensions)
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box (AABB) of the cuboid.
        
        Returns:
            (min_corner, max_corner) - two 3D points defining the AABB
        """
        # Get vertices of the oriented cuboid
        half_dims = self.dimensions / 2.0
        local_vertices = np.array([
            [-half_dims[0], -half_dims[1], -half_dims[2]],
            [half_dims[0], -half_dims[1], -half_dims[2]],
            [half_dims[0], half_dims[1], -half_dims[2]],
            [-half_dims[0], half_dims[1], -half_dims[2]],
            [-half_dims[0], -half_dims[1], half_dims[2]],
            [half_dims[0], -half_dims[1], half_dims[2]],
            [half_dims[0], half_dims[1], half_dims[2]],
            [-half_dims[0], half_dims[1], half_dims[2]],
        ])
        
        # Transform to world frame
        world_vertices = self.center + (self.orientation @ local_vertices.T).T
        
        # Get AABB
        min_corner = np.min(world_vertices, axis=0)
        max_corner = np.max(world_vertices, axis=0)
        
        return min_corner, max_corner


def normalize_rotation_matrix(rot_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize a rotation matrix to ensure it's a proper rotation matrix.
    
    Uses QR decomposition to orthogonalize the matrix and ensures determinant is positive.
    
    Args:
        rot_matrix: 3x3 rotation matrix (may be invalid)
    
    Returns:
        Valid 3x3 rotation matrix
    """
    rot_matrix = np.array(rot_matrix)
    
    # Use QR decomposition to orthogonalize
    Q, _ = np.linalg.qr(rot_matrix)
    
    # Ensure right-handed coordinate system (determinant = 1)
    if np.linalg.det(Q) < 0:
        Q[:, 2] = -Q[:, 2]  # Flip one column to make determinant positive
    
    return Q


def check_gripper_obstacle_collision(
    gripper_geometry: GripperGeometry,
    gripper_position: np.ndarray,
    gripper_orientation: np.ndarray,
    obstacle: Cuboid,
    margin: float = 0.1,
) -> bool:
    """
    Check if a gripper at a given pose collides with an obstacle.
    
    Args:
        gripper_geometry: Geometry of the gripper
        gripper_position: Gripper position (grasp position)
        gripper_orientation: 3x3 rotation matrix
        obstacle: Obstacle cuboid
        margin: Safety margin (cm)
    
    Returns:
        True if collision detected
    """
    # Get gripper bounding box
    gripper_min, gripper_max = gripper_geometry.get_bounding_box(
        gripper_position, gripper_orientation
    )
    
    # Add margin to gripper
    gripper_min = gripper_min - margin
    gripper_max = gripper_max + margin
    
    # Get obstacle bounding box
    obstacle_min, obstacle_max = obstacle.get_bounding_box()
    
    # Check overlap
    return aabb_overlap(gripper_min, gripper_max, obstacle_min, obstacle_max)


def check_segment_collision(
    gripper_geometry: GripperGeometry,
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    obstacles: List[Cuboid],
    num_samples: int = 10,
    margin: float = 0.1,
) -> bool:
    """
    Check if a trajectory segment (between two poses) collides with any obstacle.
    
    Args:
        gripper_geometry: Geometry of the gripper
        start_pose: 4x4 transformation matrix for start pose
        end_pose: 4x4 transformation matrix for end pose
        obstacles: List of obstacle cuboids
        num_samples: Number of intermediate poses to check along the segment
        margin: Safety margin (cm)
    
    Returns:
        True if collision detected anywhere along the segment
    """
    # If no obstacles, segment is always feasible
    if not obstacles:
        return False
    
    start_pos = start_pose[:3, 3]
    # Normalize rotation matrices before converting to scipy Rotation
    try:
        start_rot_matrix = normalize_rotation_matrix(start_pose[:3, :3])
        start_rot = R.from_matrix(start_rot_matrix)
    except Exception:
        # If normalization fails, use identity rotation as fallback
        start_rot = R.identity()
    
    end_pos = end_pose[:3, 3]
    try:
        end_rot_matrix = normalize_rotation_matrix(end_pose[:3, :3])
        end_rot = R.from_matrix(end_rot_matrix)
    except Exception:
        # If normalization fails, use identity rotation as fallback
        end_rot = R.identity()
    
    # Sample poses along the trajectory segment
    for i in range(num_samples + 1):
        alpha = i / num_samples
        
        # Interpolate position linearly
        position = (1 - alpha) * start_pos + alpha * end_pos
        
        # Interpolate rotation using SLERP (spherical linear interpolation)
        orientation_rot = start_rot * (start_rot.inv() * end_rot) ** alpha
        orientation = orientation_rot.as_matrix()
        
        # Check collision at this pose
        for obstacle in obstacles:
            if check_gripper_obstacle_collision(
                gripper_geometry, position, orientation, obstacle, margin
            ):
                return True
    
    return False


def check_trajectory_feasibility(
    trajectory_data: Dict[str, Any],
    obstacles: List[Cuboid],
    gripper_type: str,
    num_samples_per_segment: int = 10,
    margin: float = 0.1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Check feasibility of a single trajectory against obstacles.
    
    Args:
        trajectory_data: Dictionary containing trajectory and grasp information
        obstacles: List of obstacle cuboids
        gripper_type: 'suction' or 'parallel_jaw'
        num_samples_per_segment: Number of samples to check per trajectory segment
        margin: Safety margin for collision detection (cm)
    
    Returns:
        Tuple of (feasibility_matrix, metadata):
        - feasibility_matrix: 1D boolean array, True for feasible segments
        - metadata: Dictionary with grasp information and gripper type
    """
    # Select trajectory and geometry based on gripper type
    if gripper_type == 'suction':
        trajectory = trajectory_data['suction_trajectory']
        gripper_geometry = SUCTION_GEOMETRY
        grasp = trajectory_data['suction_grasp']
    elif gripper_type == 'parallel_jaw':
        trajectory = trajectory_data['parallel_jaw_trajectory']
        gripper_geometry = PARALLEL_JAW_GEOMETRY
        grasp = trajectory_data['parallel_jaw_grasp']
    else:
        raise ValueError(f"Unknown gripper type: {gripper_type}")
    
    # Convert trajectory to numpy arrays
    trajectory_matrices = [np.array(T) for T in trajectory]
    
    # Check each segment
    num_segments = len(trajectory_matrices) - 1
    feasibility = np.ones(num_segments, dtype=bool)
    
    for i in range(num_segments):
        start_pose = trajectory_matrices[i]
        end_pose = trajectory_matrices[i + 1]
        
        # Check if this segment collides with any obstacle
        if check_segment_collision(
            gripper_geometry,
            start_pose,
            end_pose,
            obstacles,
            num_samples_per_segment,
            margin,
        ):
            feasibility[i] = False
    
    # Create metadata
    metadata = {
        'gripper_type': gripper_type,
        'grasp': grasp,
        'num_waypoints': len(trajectory_matrices),
        'num_segments': num_segments,
    }
    
    return feasibility, metadata
