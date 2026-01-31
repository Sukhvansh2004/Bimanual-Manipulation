"""
Trajectory generator for end effector poses based on grasp pairs.

Generates simple pick-and-lift trajectories for feasible grasps.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose
from robosuite_franka_manipulation.transforms import TransformManager

logger = logging.getLogger(__name__)

def generate_end_effector_trajectory(
    object_trajectory: List[StampedPose],
    grasp: StampedPose,
    tf_manager: Optional[TransformManager] = None
) -> List[StampedPose]:
    """
    Generate end effector trajectory from object trajectory and a grasp pose.
    
    Args:
        object_trajectory: List of object poses (StampedPose).
        grasp: Grasp pose (GraspSchema/StampedPose).
        tf_manager: Optional TransformManager to handle frame conversions.
        
    Returns:
        List of StampedPose objects representing end effector poses in world frame.
    """
    if not object_trajectory:
        return []

    # 1. Ensure Grasp is in World Frame
    if grasp.frame_id != constants.WORLD_FRAME:
        if tf_manager:
            grasp = tf_manager.transform_pose(grasp, constants.WORLD_FRAME)
        else:
            logger.warning(f"Grasp frame is '{grasp.frame_id}', expected 'world'. No tf_manager provided.")
            # Proceeding, assuming caller knows what they are doing or matrix math handles it implicitly if consistent
            # But the logic below assumes world frame logic for T_obj_grasp derivation if T_world_obj_start is world.

    # 2. Ensure Object Trajectory is in World Frame
    if object_trajectory[0].frame_id != constants.WORLD_FRAME:
        if tf_manager:
            object_trajectory = [tf_manager.transform_pose(p, constants.WORLD_FRAME) for p in object_trajectory]
        else:
            logger.warning(f"Object trajectory frame is '{object_trajectory[0].frame_id}', expected 'world'.")

    # Assumption: object_trajectory[0] corresponds to the object pose at the time 'grasp' was generated.
    T_world_obj_start = object_trajectory[0].to_matrix()
    T_world_grasp = grasp.to_matrix()
    
    T_obj_grasp = np.linalg.inv(T_world_obj_start) @ T_world_grasp
    
    end_effector_trajectory = []
    
    for obj_pose in object_trajectory:
        T_object = obj_pose.to_matrix()
        # Transform to end effector frame: T_world_end_eff = T_world_obj * T_obj_grasp
        T_end_effector = T_object @ T_obj_grasp
        end_effector_trajectory.append(StampedPose.from_matrix(T_end_effector, frame_id=constants.WORLD_FRAME))
    
    return end_effector_trajectory

def generate_lift_trajectory(
    env: DualArmEnvironment,
    feasible_pairs: List[Tuple[StampedPose, StampedPose]],
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
    object_trajectory = []
    start_pos = T_obj_start_mat[:3, 3]
    start_rot = T_obj_start_mat[:3, :3]
    
    for i in range(steps + 1):
        alpha = i / steps
        pos = start_pos + np.array([0, 0, lift_height * alpha])
        T_step = np.eye(4)
        T_step[:3, :3] = start_rot
        T_step[:3, 3] = pos
        object_trajectory.append(StampedPose.from_matrix(T_step, frame_id=constants.WORLD_FRAME))
        
    all_trajectories = []
    
    for i, (grasp0, grasp1) in enumerate(feasible_pairs):
        # Generate trajectory for robot 0
        traj0 = generate_end_effector_trajectory(object_trajectory, grasp0)
        
        # Generate trajectory for robot 1
        traj1 = generate_end_effector_trajectory(object_trajectory, grasp1)
        
        all_trajectories.append({
            'grasp_robot0': grasp0,
            'grasp_robot1': grasp1,
            'trajectory_robot0': traj0,
            'trajectory_robot1': traj1,
            'object_trajectory': object_trajectory,
        })
        
    return all_trajectories