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
            for i, pose in enumerate(eef_trajectory):
                # 1. Update Object Pose if provided
                if object_trajectory is not None and obj_joint_addr is not None:
                    obj_pose = object_trajectory[i]
                    pos = obj_pose.position
                    quat = T.convert_quat(obj_pose.quaternion, to="wxyz")
                    data.qpos[obj_joint_addr:obj_joint_addr+3] = pos
                    data.qpos[obj_joint_addr+3:obj_joint_addr+7] = quat
                
                # 2. Solve IK
                solutions = enumerate_ik_solutions_for_arm(
                    env=env,
                    robot_id=robot_id,
                    target_pose=pose,
                    target_frame=constants.WORLD_FRAME,
                    arm="right",
                    num_seed_configs=16,
                    max_solutions=1,
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
                    continue
                
                q_sol = solutions[0]
                
                # 3. Check Collision
                collision_free = True
                if check_collisions:
                    set_qpos(data, qpos_idx, q_sol)
                    mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
                    collision_free = not in_collision(env, env_lock=None, ignore_names=ignore_names)
                
                is_feasible = collision_free
                if not is_feasible:
                    all_feasible = False
                
                results.append(WaypointFeasibilityResult(
                    waypoint_index=i,
                    eef_pose=pose,
                    joint_config=q_sol,
                    is_feasible=is_feasible,
                    ik_success=True,
                    collision_free=collision_free
                ))
                
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
    
    return results, all_feasible


def save_trajectory(trajectory: List[StampedPose], output_dir: str, filename: str) -> None:
    """Save trajectory to npz file.
    
    Args:
        trajectory: List of StampedPose objects
        output_dir: Output directory
        filename: Output filename
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    
    # Extract data from StampedPose objects
    positions = np.array([p.position for p in trajectory])
    quaternions = np.array([p.quaternion for p in trajectory])
    frame_ids = np.array([p.frame_id for p in trajectory])
    
    # Save as npz
    np.savez(
        filepath,
        positions=positions,
        quaternions=quaternions,
        frame_ids=frame_ids
    )
    logger.info(f"Trajectory saved to {filepath}")
