"""
Trajectory generator for end effector poses based on grasp pairs.

Generates simple pick-and-lift trajectories for feasible grasps.
"""

import numpy as np
from typing import List, Optional, Tuple
import logging
import mujoco
from contextlib import nullcontext
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose
from robosuite_franka_manipulation.transforms import TransformManager
from robosuite_franka_manipulation.utils.se3 import interpolate_pose
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite.utils import transform_utils as T
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_native_mj_object
)
from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    WaypointFeasibilityResult, 
    resolve_ignore_names, 
    process_waypoint,
    process_dual_waypoint,
)
from robosuite_franka_manipulation.planning.object_trajectory_planner import (
    plan_object_trajectory,
)

logger = logging.getLogger(__name__)

def generate_object_trajectory(
    start_pose: StampedPose, 
    target_pose: StampedPose, 
    tf_manager: TransformManager,
    num_steps: int = constants.OBJECT_TRAJ_NUM_STEPS,
    env: Optional[DualArmEnvironment] = None,
    object_name: Optional[str] = None,
    env_lock=None,
    pos_bounds: Optional[np.ndarray] = None,
) -> List[StampedPose]:
    """Generate a collision-free trajectory for the object from start to target pose.
    
    When *env* and *object_name* are provided, an RRT-Connect planner is used
    to find a collision-free path (ignoring contacts with robot arms/grippers).
    If the planner fails, or if *env* is not provided, falls back to simple
    SE(3) interpolation.
    
    Args:
        start_pose: Start pose as StampedPose
        target_pose: Target pose as StampedPose
        tf_manager: TransformManager for frame conversion
        num_steps: Number of interpolation steps
        env: (Optional) DualArmEnvironment for collision-aware planning
        object_name: (Optional) Name of the object in MuJoCo (required when env is given)
        env_lock: (Optional) Threading lock for environment access
        pos_bounds: (Optional) Workspace bounds as (2, 3) numpy array
            ``[[lo_x, lo_y, lo_z], [hi_x, hi_y, hi_z]]``.
            Overrides auto-computed bounds when provided.
        
    Returns:
        List of StampedPose objects representing the trajectory
    """
    # Normalize frames to world
    if start_pose.frame_id != constants.WORLD_FRAME:
        start_pose = tf_manager.transform_pose(start_pose, constants.WORLD_FRAME)
    if target_pose.frame_id != constants.WORLD_FRAME:
        target_pose = tf_manager.transform_pose(target_pose, constants.WORLD_FRAME)
            
    # If still mismatch (should not happen if tf_manager works), warn
    if start_pose.frame_id != target_pose.frame_id:
        logger.warning(f"Frame mismatch in generate_object_trajectory: {start_pose.frame_id} vs {target_pose.frame_id}")

    # ---- Collision-aware planning (when env is available) ----
    if env is not None and object_name is not None:
        logger.info("Using RRT-Connect planner for collision-aware object trajectory.")
        trajectory, success = plan_object_trajectory(
            env=env,
            object_name=object_name,
            start_pose=start_pose,
            target_pose=target_pose,
            tf_manager=tf_manager,
            num_steps=num_steps,
            env_lock=env_lock,
            pos_bounds=pos_bounds,
        )
        if success:
            return trajectory
        else:
            logger.warning("RRT planner failed — returned trajectory is a direct interpolation fallback.")
            return trajectory

    # ---- Fallback: simple SE(3) interpolation ----
    logger.info("No environment provided — using direct SE(3) interpolation for object trajectory.")
    frame_id = start_pose.frame_id
    
    trajectory_points = []
    for i in range(num_steps + 1):
        t = i / float(num_steps)
        pos, quat = interpolate_pose(start_pose.position, start_pose.quaternion, target_pose.position, target_pose.quaternion, t)
        trajectory_points.append(StampedPose(position=pos, quaternion=quat, frame_id=frame_id))

    return trajectory_points


def generate_end_effector_trajectory(
    object_trajectory: List[StampedPose],
    grasp: StampedPose,
    tf_manager: TransformManager
) -> List[StampedPose]:
    """
    Generate end effector trajectory from object trajectory and a grasp pose.
    
    The grasp pose is assumed to already be at the robot's grip_site frame
    (i.e., the grasp service has already applied any necessary offsets).
    
    Args:
        object_trajectory: List of object poses (StampedPose).
        grasp: Grasp pose (GraspSchema/StampedPose) - already at grip_site frame.
        tf_manager: TransformManager to handle frame conversions.
        
    Returns:
        List of StampedPose objects representing end effector poses in world frame.
    """
    if not object_trajectory:
        return []

    # 1. Ensure Grasp is in World Frame
    if grasp.frame_id != constants.WORLD_FRAME:
        grasp = tf_manager.transform_pose(grasp, constants.WORLD_FRAME)

    # 2. Ensure Object Trajectory is in World Frame
    if object_trajectory[0].frame_id != constants.WORLD_FRAME:
        object_trajectory = [tf_manager.transform_pose(p, constants.WORLD_FRAME) for p in object_trajectory]

    # Grasp is already at grip_site frame (no offset needed)
    T_world_grasp = grasp.to_matrix()

    # Assumption: object_trajectory[0] corresponds to the object pose at the time 'grasp' was generated.
    T_world_obj_start = object_trajectory[0].to_matrix()
    
    # Compute grasp pose relative to object
    T_obj_grasp = np.linalg.inv(T_world_obj_start) @ T_world_grasp
    
    end_effector_trajectory = []
    
    for obj_pose in object_trajectory:
        T_object = obj_pose.to_matrix()
        # Transform to end effector frame: T_world_end_eff = T_world_obj * T_obj_grasp
        T_end_effector = T_object @ T_obj_grasp
        end_effector_trajectory.append(StampedPose.from_matrix(T_end_effector, frame_id=constants.WORLD_FRAME))
    
    return end_effector_trajectory


def check_trajectory_feasibility(
    env: DualArmEnvironment,
    robot_id: int,
    eef_trajectory: List[StampedPose],
    tf_manager: TransformManager,
    object_trajectory: Optional[List[StampedPose]] = None,
    object_name: Optional[str] = None,
    check_collisions: bool = True,
    env_lock = None,
    ignore_names: Optional[List[str]] = None,
    arm: str = "right"
) -> Tuple[List[WaypointFeasibilityResult], bool]:
    """
    Check trajectory feasibility and return detailed per-waypoint results.
    
    Args:
        env: Environment instance
        robot_id: Robot ID (0 or 1)
        eef_trajectory: List of StampedPose objects representing EEF poses.
        tf_manager: TransformManager to handle frame conversions.
        object_trajectory: Optional list of StampedPose objects for object poses.
        object_name: Name of the object (required if object_trajectory is provided).
        check_collisions: Whether to check collisions.
        env_lock: Optional threading.RLock
        ignore_names: List of geom names to ignore in collision checking.
        arm: Arm side ("right" or "left").
        
    Returns:
        Tuple of (list of WaypointFeasibilityResult, overall_feasible bool)
    """
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
            
            # Resolve ignore names and pairs
            # Object-gripper contacts are selectively ignored; object-arm contacts are still detected
            current_ignore_names, current_ignore_pairs = resolve_ignore_names(object_name, ignore_names, robot_id=robot_id)
            logger.debug(f"Collision checking: ignore_names={current_ignore_names}, ignore_pairs={current_ignore_pairs}")

            for i, pose in enumerate(eef_trajectory):
                # 1. Update Object Pose if provided
                if object_trajectory is not None and obj_joint_addr is not None:
                    obj_pose = object_trajectory[i]
                    pos = obj_pose.position
                    quat = T.convert_quat(obj_pose.quaternion, to="wxyz")
                    data.qpos[obj_joint_addr:obj_joint_addr+3] = pos
                    data.qpos[obj_joint_addr+3:obj_joint_addr+7] = quat
                
                # 2. Check Waypoint
                result, new_config = process_waypoint(
                    env=env,
                    robot_id=robot_id,
                    waypoint_index=i,
                    pose=pose,
                    qpos_idx=qpos_idx,
                    check_collisions=check_collisions,
                    ignore_names=current_ignore_names,
                    last_config=last_config,
                    tf_manager=tf_manager,
                    arm=arm,
                    ignore_pairs=current_ignore_pairs,
                )
                
                results.append(result)
                
                if not result.is_feasible:
                    all_feasible = False
                
                last_config = new_config

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
    
    return results, all_feasible


def check_dual_trajectory_feasibility(
    env: DualArmEnvironment,
    eef_trajectory0: List[StampedPose],
    eef_trajectory1: List[StampedPose],
    tf_manager: TransformManager,
    object_trajectory: Optional[List[StampedPose]] = None,
    object_name: Optional[str] = None,
    check_collisions: bool = True,
    env_lock = None,
    ignore_names: Optional[List[str]] = None,
    arm: str = "right"
) -> Tuple[List[WaypointFeasibilityResult], List[WaypointFeasibilityResult], bool]:
    """
    Check simultaneous feasibility of two robot trajectories.
    
    Returns:
        Tuple of (results0, results1, overall_feasible bool)
    """
    if len(eef_trajectory0) != len(eef_trajectory1):
        logger.error("EEF trajectories length mismatch")
        return [], [], False

    if object_trajectory and len(object_trajectory) != len(eef_trajectory0):
        logger.error("Object trajectory length mismatch")
        return [], [], False

    ctx = env_lock if env_lock is not None else nullcontext()

    results0: List[WaypointFeasibilityResult] = []
    results1: List[WaypointFeasibilityResult] = []
    all_feasible = True

    logger.debug(f"Checking dual-arm trajectory feasibility for {len(eef_trajectory0)} waypoints...")

    with ctx:
        model = env.sim.model
        data = env.sim.data
        
        jids0 = get_arm_joint_ids(model, 0)
        jids1 = get_arm_joint_ids(model, 1)
        qpos_idx0 = get_joint_qpos_indices(model, jids0)
        qpos_idx1 = get_joint_qpos_indices(model, jids1)
        
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
            last_config0 = None 
            last_config1 = None
            
            # Resolve ignore names and pairs for dual-arm
            # Both robots hold the object at different times, so create pairs for both grippers
            current_ignore_names, current_ignore_pairs = resolve_ignore_names(object_name, ignore_names)
            # Add object-gripper pairs for both robots
            if object_name:
                current_ignore_pairs.extend([
                    (object_name, "gripper0"),
                    (object_name, "gripper1"),
                ])

            for i in range(len(eef_trajectory0)):
                # 1. Update Object Pose if provided
                if object_trajectory is not None and obj_joint_addr is not None:
                    obj_pose = object_trajectory[i]
                    pos = obj_pose.position
                    quat = T.convert_quat(obj_pose.quaternion, to="wxyz")
                    data.qpos[obj_joint_addr:obj_joint_addr+3] = pos
                    data.qpos[obj_joint_addr+3:obj_joint_addr+7] = quat
                
                # 2. Check Waypoint for both
                res0, res1, new_config0, new_config1 = process_dual_waypoint(
                    env=env,
                    waypoint_index=i,
                    pose0=eef_trajectory0[i],
                    pose1=eef_trajectory1[i],
                    qpos_idx0=qpos_idx0,
                    qpos_idx1=qpos_idx1,
                    check_collisions=check_collisions,
                    ignore_names=current_ignore_names,
                    last_config0=last_config0,
                    last_config1=last_config1,
                    tf_manager=tf_manager,
                    arm=arm,
                    ignore_pairs=current_ignore_pairs,
                )
                
                results0.append(res0)
                results1.append(res1)
                
                if not (res0.is_feasible and res1.is_feasible):
                    all_feasible = False
                    logger.debug(f"  Waypoint {i} INFEASIBLE (Robot 0: {res0.is_feasible}, Robot 1: {res1.is_feasible})")
                
                if new_config0 is not None: last_config0 = new_config0
                if new_config1 is not None: last_config1 = new_config1

            if all_feasible:
                logger.debug("Dual-arm trajectory is FULLY FEASIBLE.")
            else:
                feasible_count = sum(1 for r0, r1 in zip(results0, results1) if r0.is_feasible and r1.is_feasible)
                logger.debug(f"Dual-arm trajectory is INFEASIBLE ({feasible_count}/{len(eef_trajectory0)} waypoints feasible).")

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
    
    return results0, results1, all_feasible
