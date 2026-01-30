"""
Trajectory generator for end effector poses from object trajectory and grasp pairs.

This module generates end effector trajectories by transforming object poses
using feasible grasp pairs.
"""

import json
import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.spatial.transform import Rotation as R
from .grasp_generator import GraspPose
from .collision_checker import find_feasible_grasp_pairs


def load_trajectory_from_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load trajectory data from JSON file.
    
    Args:
        filepath: Path to JSON file containing trajectory data
        
    Returns:
        List of trajectory waypoints, each with 'position' and 'orientation' or 'orientation_euler'
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data.get('trajectory', [])


def pose_to_matrix(position: np.ndarray, orientation: np.ndarray = None, orientation_euler: np.ndarray = None) -> np.ndarray:
    """
    Convert position and orientation to 4x4 homogeneous transformation matrix.
    
    Args:
        position: 3D position [x, y, z]
        orientation: Quaternion [x, y, z, w] or rotation matrix (3x3)
        orientation_euler: Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, 3] = np.array(position)
    
    if orientation_euler is not None:
        # Convert Euler angles to rotation matrix
        r = R.from_euler('xyz', orientation_euler, degrees=False)
        T[:3, :3] = r.as_matrix()
    elif orientation is not None:
        orientation = np.array(orientation)
        if orientation.shape == (4,):
            # Quaternion [x, y, z, w]
            r = R.from_quat(orientation)
            T[:3, :3] = r.as_matrix()
        elif orientation.shape == (3, 3):
            # Rotation matrix
            T[:3, :3] = orientation
        else:
            raise ValueError(f"Invalid orientation shape: {orientation.shape}")
    else:
        # Default: identity rotation
        T[:3, :3] = np.eye(3)
    
    return T


def generate_end_effector_trajectory(
    object_trajectory: List[Dict[str, Any]],
    grasp: GraspPose,
) -> List[np.ndarray]:
    """
    Generate end effector trajectory from object trajectory and a grasp pose.
    
    The object trajectory contains poses in world frame.
    The grasp pose is in object frame (plate frame).
    The end effector pose in world frame is: T_world_end_eff = T_world_obj * T_obj_end_eff
    
    Args:
        object_trajectory: List of object poses (dicts with 'position' and 'orientation'/'orientation_euler')
        grasp: Grasp pose in object frame
        
    Returns:
        List of 4x4 transformation matrices representing end effector poses in world frame
    """
    # Get grasp transform in object frame
    T_grasp = grasp.to_matrix()
    
    end_effector_trajectory = []
    
    for waypoint in object_trajectory:
        # Get object pose in world frame
        position = waypoint.get('position', [0.0, 0.0, 0.0])
        orientation = waypoint.get('orientation', None)
        orientation_euler = waypoint.get('orientation_euler', None)
        
        T_object = pose_to_matrix(position, orientation, orientation_euler)
        
        # Transform to end effector frame: T_world_end_eff = T_world_obj * T_obj_end_eff
        T_end_effector = T_object @ T_grasp
        
        end_effector_trajectory.append(T_end_effector)
    
    return end_effector_trajectory


def generate_trajectories_for_all_pairs(
    trajectory_filepath: str,
    feasible_pairs: List[Tuple[GraspPose, GraspPose]],
) -> List[Dict[str, Any]]:
    """
    Generate end effector trajectories for all feasible grasp pairs.
    
    Args:
        trajectory_filepath: Path to JSON file containing object trajectory
        feasible_pairs: List of (suction_grasp, parallel_jaw_grasp) tuples
        
    Returns:
        List of dictionaries, each containing:
        - 'suction_grasp': suction grasp pose
        - 'parallel_jaw_grasp': parallel jaw grasp pose
        - 'suction_trajectory': list of 4x4 matrices for suction gripper end effector
        - 'parallel_jaw_trajectory': list of 4x4 matrices for parallel jaw gripper end effector
        - 'object_trajectory': list of 4x4 matrices for object poses (same for all pairs)
    """
    # Load object trajectory
    object_trajectory = load_trajectory_from_json(trajectory_filepath)
    
    if not object_trajectory:
        raise ValueError(f"No trajectory data found in {trajectory_filepath}")
    
    print(f"Loaded {len(object_trajectory)} waypoints from trajectory file")
    print(f"Generating trajectories for {len(feasible_pairs)} feasible grasp pairs...")
    
    # Convert object trajectory to transformation matrices
    object_trajectory_matrices = []
    for waypoint in object_trajectory:
        position = waypoint.get('position', [0.0, 0.0, 0.0])
        orientation = waypoint.get('orientation', None)
        orientation_euler = waypoint.get('orientation_euler', None)
        T_object = pose_to_matrix(position, orientation, orientation_euler)
        object_trajectory_matrices.append(T_object)
    
    all_trajectories = []
    
    for i, (suction_grasp, parallel_jaw_grasp) in enumerate(feasible_pairs):
        # Generate trajectory for suction gripper
        suction_trajectory = generate_end_effector_trajectory(object_trajectory, suction_grasp)
        
        # Generate trajectory for parallel jaw gripper
        parallel_jaw_trajectory = generate_end_effector_trajectory(object_trajectory, parallel_jaw_grasp)
        
        all_trajectories.append({
            'suction_grasp': suction_grasp,
            'parallel_jaw_grasp': parallel_jaw_grasp,
            'suction_trajectory': suction_trajectory,
            'parallel_jaw_trajectory': parallel_jaw_trajectory,
            'object_trajectory': object_trajectory_matrices,  # Same for all pairs
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Generated trajectories for {i + 1}/{len(feasible_pairs)} pairs...")
    
    print(f"Generated {len(all_trajectories)} trajectory pairs")
    
    return all_trajectories


def main():
    """Example usage of the trajectory generator."""
    import os
    import sys
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from bimanual.grasp_generator import PlateGraspGenerator
    from bimanual.suction_grasp_generator import SuctionGraspGenerator
    from bimanual.collision_checker import find_feasible_grasp_pairs
    
    # Configuration
    trajectory_filepath = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'dummy_trajectory.json'
    )
    plate_width = 10.0
    plate_length = 20.0
    plate_thickness = 2.0
    
    print("Trajectory Generator")
    print("=" * 60)
    
    # Generate or load grasps
    print("\nGenerating grasps...")
    parallel_jaw_generator = PlateGraspGenerator(
        plate_width=plate_width,
        plate_length=plate_length,
        plate_thickness=plate_thickness,
        max_gripper_opening=5.0,
        grasp_offset=1.0,
    )
    parallel_jaw_grasps = parallel_jaw_generator.generate_all_grasps(
        num_grasps_per_edge=5,
        include_corners=True,
    )
    
    suction_generator = SuctionGraspGenerator(
        plate_width=plate_width,
        plate_length=plate_length,
        plate_thickness=plate_thickness,
        suction_cup_radius=1.0,
        min_distance_from_edge=1.5,
    )
    suction_grasps = suction_generator.generate_all_grasps(
        num_grasps_x=5,
        num_grasps_y=5,
        include_top=True,
        include_bottom=True,
    )
    
    # Find feasible pairs
    print("\nFinding feasible grasp pairs...")
    feasible_pairs = find_feasible_grasp_pairs(
        suction_grasps,
        parallel_jaw_grasps,
        plate_width=plate_width,
        plate_length=plate_length,
        plate_thickness=plate_thickness,
        margin=0.5
    )
    
    # Generate trajectories
    print("\nGenerating end effector trajectories...")
    all_trajectories = generate_trajectories_for_all_pairs(
        trajectory_filepath,
        feasible_pairs,
    )
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total trajectory pairs: {len(all_trajectories)}")
    print(f"  Waypoints per trajectory: {len(all_trajectories[0]['suction_trajectory']) if all_trajectories else 0}")
    print(f"{'='*60}\n")
    
    return all_trajectories


if __name__ == "__main__":
    main()
