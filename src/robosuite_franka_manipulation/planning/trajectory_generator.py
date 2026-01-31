"""
Trajectory generator for end effector poses based on grasp pairs.

Generates simple pick-and-lift trajectories for feasible grasps.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from robosuite_franka_manipulation.schema import GraspSchema
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation import constants


def generate_end_effector_trajectory(
    object_trajectory: List[np.ndarray],
    grasp: GraspSchema,
) -> List[np.ndarray]:
    """
    Generate end effector trajectory from object trajectory and a grasp pose.
    
    Args:
        object_trajectory: List of object poses (4x4 matrices) in world frame.
        grasp: Grasp pose (GraspSchema/StampedPose) in WORLD frame.
        
    Returns:
        List of 4x4 transformation matrices representing end effector poses in world frame.
    """
    # Assumption: object_trajectory[0] corresponds to the object pose at the time 'grasp' was generated.
    # So T_world_obj_0 = object_trajectory[0]
    # T_world_grasp = grasp.to_matrix()
    # T_obj_grasp = inv(T_world_obj_0) @ T_world_grasp
    
    # Then for any T_world_obj_i in trajectory:
    # T_world_eef_i = T_world_obj_i @ T_obj_grasp
    
    T_world_obj_start = object_trajectory[0]
    T_world_grasp = grasp.to_matrix()
    
    T_obj_grasp = np.linalg.inv(T_world_obj_start) @ T_world_grasp
    
    end_effector_trajectory = []
    
    for T_object in object_trajectory:
        # Transform to end effector frame: T_world_end_eff = T_world_obj * T_obj_grasp
        T_end_effector = T_object @ T_obj_grasp
        end_effector_trajectory.append(T_end_effector)
    
    return end_effector_trajectory

def generate_lift_trajectory(
    env: DualArmEnvironment,
    feasible_pairs: List[Tuple[GraspSchema, GraspSchema]],
    object_name: str,
    lift_height: float = constants.DEFAULT_LIFT_HEIGHT,
    steps: int = constants.DEFAULT_LIFT_STEPS
) -> List[Dict[str, Any]]:
    """
    Generate lift trajectories for all feasible grasp pairs.
    
    Args:
        env: DualArmEnvironment
        feasible_pairs: List of (grasp0, grasp1) tuples
        object_name: Name of object
        lift_height: Height to lift
        steps: Number of waypoints
        
    Returns:
        List of dictionaries with trajectories.
    """
    # Get object current pose
    T_obj_start_mat = env.get_object_pose(object_name)
    if T_obj_start_mat is None:
        raise ValueError(f"Could not get pose for {object_name}")
    
    # Generate object trajectory (linear lift)
    object_trajectory_matrices = []
    start_pos = T_obj_start_mat[:3, 3]
    start_rot = T_obj_start_mat[:3, :3]
    
    for i in range(steps + 1):
        alpha = i / steps
        pos = start_pos + np.array([0, 0, lift_height * alpha])
        T_step = np.eye(4)
        T_step[:3, :3] = start_rot
        T_step[:3, 3] = pos
        object_trajectory_matrices.append(T_step)
        
    all_trajectories = []
    
    for i, (grasp0, grasp1) in enumerate(feasible_pairs):
        # Generate trajectory for robot 0
        traj0 = generate_end_effector_trajectory(object_trajectory_matrices, grasp0)
        
        # Generate trajectory for robot 1
        traj1 = generate_end_effector_trajectory(object_trajectory_matrices, grasp1)
        
        all_trajectories.append({
            'grasp_robot0': grasp0,
            'grasp_robot1': grasp1,
            'trajectory_robot0': traj0,
            'trajectory_robot1': traj1,
            'object_trajectory': object_trajectory_matrices,
        })
        
    return all_trajectories