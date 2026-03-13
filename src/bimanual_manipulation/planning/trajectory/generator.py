"""
Trajectory generator for end effector poses based on grasp pairs.

Generates simple pick-and-lift trajectories for feasible grasps.

* :func:`generate_object_trajectory` — collision-aware (RRT) or direct
  SE(3) interpolation for the object.
* :func:`generate_end_effector_trajectory` — maps an object trajectory
  to EEF poses via a fixed grasp transform.
"""

import numpy as np
from typing import List, Optional
import logging

from bimanual_manipulation import constants
from bimanual_manipulation.transforms.stamped_pose import StampedPose
from bimanual_manipulation.transforms import TransformManager
from bimanual_manipulation.utils.se3 import interpolate_pose
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.planning.trajectory.object_planner import (
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

    If *env* is not provided, falls back to simple SE(3) interpolation.

    Raises:
        RuntimeError: If the RRT planner fails to find a collision-free path.

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
        logger.warning(
            f"Frame mismatch in generate_object_trajectory: "
            f"{start_pose.frame_id} vs {target_pose.frame_id}"
        )

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
        return trajectory

    # ---- Fallback: simple SE(3) interpolation ----
    logger.info("No environment provided — using direct SE(3) interpolation for object trajectory.")
    frame_id = start_pose.frame_id

    trajectory_points = []
    for i in range(num_steps + 1):
        t = i / float(num_steps)
        pos, quat = interpolate_pose(
            start_pose.position, start_pose.quaternion,
            target_pose.position, target_pose.quaternion, t,
        )
        trajectory_points.append(
            StampedPose(position=pos, quaternion=quat, frame_id=frame_id)
        )

    return trajectory_points


def generate_end_effector_trajectory(
    object_trajectory: List[StampedPose],
    grasp: StampedPose,
    tf_manager: TransformManager,
) -> List[StampedPose]:
    """Generate end effector trajectory from object trajectory and a grasp pose.

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
        object_trajectory = [
            tf_manager.transform_pose(p, constants.WORLD_FRAME) for p in object_trajectory
        ]

    T_world_grasp = grasp.to_matrix()

    # Assumption: object_trajectory[0] corresponds to the object pose at the
    # time 'grasp' was generated.
    T_world_obj_start = object_trajectory[0].to_matrix()

    # Compute grasp pose relative to object
    T_obj_grasp = np.linalg.inv(T_world_obj_start) @ T_world_grasp

    end_effector_trajectory = []

    for obj_pose in object_trajectory:
        T_object = obj_pose.to_matrix()
        # Transform to end effector frame: T_world_end_eff = T_world_obj * T_obj_grasp
        T_end_effector = T_object @ T_obj_grasp
        end_effector_trajectory.append(
            StampedPose.from_matrix(T_end_effector, frame_id=constants.WORLD_FRAME)
        )

    return end_effector_trajectory
