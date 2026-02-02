"""
Trajectory verification utilities.
"""

import numpy as np
import logging
import os
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
import mujoco
from contextlib import nullcontext

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.planning.utils.ik_utils import enumerate_ik_solutions_for_arm
from robosuite_franka_manipulation.planning.utils.collision_utils import in_collision
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
    set_qpos,
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_native_mj_object
)
from robosuite.utils import transform_utils as T
from robosuite_franka_manipulation import constants

logger = logging.getLogger(__name__)

from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose
from robosuite_franka_manipulation.transforms import TransformManager


@dataclass
class WaypointFeasibilityResult:
    """Result of feasibility check for a single trajectory waypoint."""
    waypoint_index: int
    eef_pose: StampedPose
    joint_config: Optional[np.ndarray]  # None if IK failed
    is_feasible: bool  # True if IK succeeded and no collision
    ik_success: bool  # True if IK found a solution
    collision_free: bool  # True if no collision detected

def check_trajectory_feasibility(
    env: DualArmEnvironment,
    robot_id: int,
    eef_trajectory: List[StampedPose],
    object_trajectory: Optional[List[StampedPose]] = None,
    object_name: Optional[str] = None,
    check_collisions: bool = True,
    env_lock = None,
    tf_manager: Optional[TransformManager] = None,
    ignore_names: Optional[List[str]] = None
) -> Tuple[List[WaypointFeasibilityResult], bool]:
    """
    Check trajectory feasibility and return detailed per-waypoint results.
    
    Unlike check_trajectory_feasibility which returns None on first failure,
    this function checks ALL waypoints and returns detailed results for each.
    
    Args:
        env: Environment instance
        robot_id: Robot ID (0 or 1)
        eef_trajectory: List of StampedPose objects representing EEF poses.
        object_trajectory: Optional list of StampedPose objects for object poses.
        object_name: Name of the object (required if object_trajectory is provided).
        check_collisions: Whether to check collisions.
        env_lock: Optional threading.RLock
        tf_manager: Optional TransformManager to handle frame conversions.
        ignore_names: List of geom names to ignore in collision checking.
        
    Returns:
        Tuple of (list of WaypointFeasibilityResult, overall_feasible bool)
    """
    if tf_manager is None:
        tf_manager = TransformManager(env, env_lock=env_lock)

    if object_trajectory and len(object_trajectory) != len(eef_trajectory):
        logger.error("Object trajectory length mismatch")
        return [], False

    # Ensure frames are World
    if eef_trajectory and eef_trajectory[0].frame_id != constants.WORLD_FRAME:
        eef_trajectory = [tf_manager.transform_pose(p, constants.WORLD_FRAME) for p in eef_trajectory]

    if object_trajectory and object_trajectory[0].frame_id != constants.WORLD_FRAME:
        object_trajectory = [tf_manager.transform_pose(p, constants.WORLD_FRAME) for p in object_trajectory]

    ctx = env_lock if env_lock is not None else nullcontext()

    results: List[WaypointFeasibilityResult] = []
    all_feasible = True

    with ctx:
        model = env.sim.model
        data = env.sim.data
        
        jids = get_arm_joint_ids(model, robot_id)
        qpos_idx = get_joint_qpos_indices(model, jids)
        
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)
        
        # Object joint index if needed
        obj_joint_addr = None
        if object_name:
            try:
                obj_joint_id = model.joint(f"{object_name}_joint0").id
                obj_qpos_addr = model.jnt_qposadr[obj_joint_id]
                obj_joint_addr = obj_qpos_addr
            except Exception:
                pass

        try:
            last_config = None # Keep track of the last valid configuration for warm start
            
            for i, pose in enumerate(eef_trajectory):
                # 1. Update Object Pose if provided
                if object_trajectory is not None and obj_joint_addr is not None:
                    obj_pose = object_trajectory[i]
                    pos = obj_pose.position
                    quat = T.convert_quat(obj_pose.quaternion, to="wxyz")
                    data.qpos[obj_joint_addr:obj_joint_addr+3] = pos
                    data.qpos[obj_joint_addr+3:obj_joint_addr+7] = quat
                
                # 2. Solve IK
                # The input pose is the end-effector (flange) pose. 
                # We must tell IK to solve for the flange site, not the default TCP site.
                site_name = f"gripper{robot_id}_right_ft_frame"

                solutions = enumerate_ik_solutions_for_arm(
                    env=env,
                    robot_id=robot_id,
                    target_pose=pose,
                    target_frame=constants.WORLD_FRAME,
                    arm="right",
                    site_name=site_name,
                    q_warm_start=last_config,
                    num_seed_configs=constants.TRAJ_IK_NUM_SEED_CONFIGS,
                    max_solutions=constants.TRAJ_IK_MAX_SOLUTIONS,
                    env_lock=None,  # Already locked
                    tf_manager=tf_manager
                )
                
                if not solutions:
                    # IK failed
                    results.append(WaypointFeasibilityResult(
                        waypoint_index=i,
                        eef_pose=pose,
                        joint_config=None,
                        is_feasible=False,
                        ik_success=False,
                        collision_free=True  # Unknown
                    ))
                    all_feasible = False
                    # We don't update last_config here if IK failed
                    continue
                
                # Iterate over solutions to find a collision-free one
                found_valid_solution = False
                first_result = None # Store the first result in case none are collision free

                for q_sol in solutions:
                    # 3. Check Collision
                    collision_free = True
                    if check_collisions:
                        set_qpos(data, qpos_idx, q_sol)
                        mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
                        collision_free = not in_collision(env, env_lock=None, ignore_names=ignore_names)
                    
                    is_feasible = collision_free
                    
                    result = WaypointFeasibilityResult(
                        waypoint_index=i,
                        eef_pose=pose,
                        joint_config=q_sol,
                        is_feasible=is_feasible,
                        ik_success=True,
                        collision_free=collision_free
                    )

                    if first_result is None:
                        first_result = result

                    if is_feasible:
                        results.append(result)
                        found_valid_solution = True
                        last_config = q_sol # Update warm start
                        break
                
                if not found_valid_solution:
                    all_feasible = False
                    if first_result:
                        results.append(first_result)
                        last_config = first_result.joint_config

                
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
    
    return results, all_feasible


def save_trajectory(
    output_dir: str, 
    filename: str,
    eef_trajectory: Optional[List[StampedPose]] = None,
    object_trajectory: Optional[List[StampedPose]] = None,
    waypoint_results: Optional[List[WaypointFeasibilityResult]] = None,
) -> None:
    """Save trajectory to npz file with optional feasibility results.
    
    Args:
        output_dir: Output directory
        filename: Output filename
        eef_trajectory: List of end-effector StampedPose objects (optional if waypoint_results provided)
        object_trajectory: List of object StampedPose objects (optional)
        waypoint_results: List of WaypointFeasibilityResult with joint configs and feasibility (optional)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    
    data_dict = {}
    
    # Save EEF trajectory poses
    if eef_trajectory:
        data_dict['eef_positions'] = np.array([p.position for p in eef_trajectory])
        data_dict['eef_quaternions'] = np.array([p.quaternion for p in eef_trajectory])
        data_dict['eef_frame_ids'] = np.array([p.frame_id for p in eef_trajectory])
    
    # Save object trajectory poses
    if object_trajectory:
        data_dict['object_positions'] = np.array([p.position for p in object_trajectory])
        data_dict['object_quaternions'] = np.array([p.quaternion for p in object_trajectory])
        data_dict['object_frame_ids'] = np.array([p.frame_id for p in object_trajectory])
    
    # Save waypoint results with feasibility info
    if waypoint_results:
        # Feasibility flags (1 = feasible, 0 = infeasible)
        data_dict['is_feasible'] = np.array([int(r.is_feasible) for r in waypoint_results], dtype=np.int32)
        data_dict['ik_success'] = np.array([int(r.ik_success) for r in waypoint_results], dtype=np.int32)
        data_dict['collision_free'] = np.array([int(r.collision_free) for r in waypoint_results], dtype=np.int32)
        
        # Joint configurations (NaN for failed IK)
        joint_configs = []
        for r in waypoint_results:
            if r.joint_config is not None:
                joint_configs.append(r.joint_config)
            else:
                # Use NaN array for failed IK
                joint_configs.append(np.full(constants.DEFAULT_ROBOT_ACTION_DIM, np.nan))
        data_dict['joint_configs'] = np.array(joint_configs)
        
        # EEF poses from results (in case eef_trajectory not separately provided)
        if not eef_trajectory:
            data_dict['eef_positions'] = np.array([r.eef_pose.position for r in waypoint_results])
            data_dict['eef_quaternions'] = np.array([r.eef_pose.quaternion for r in waypoint_results])
            data_dict['eef_frame_ids'] = np.array([r.eef_pose.frame_id for r in waypoint_results])
    
    # Save as npz
    np.savez(filepath, **data_dict)
    logger.info(f"Trajectory saved to {filepath}")
