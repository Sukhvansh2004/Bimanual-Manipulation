"""
SE(3) utilities for the object RRT planner.

Provides SE(3) distance, steering, random sampling, and segment-level
collision checking.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.utils.se3 import (
    SE3_log,
    compute_se3_distance,
    interpolate_pose,
    pose_to_SE3,
)
from bimanual_manipulation.planning.rrt.se3.collision import (
    check_object_pose_collision,
)

logger = logging.getLogger(__name__)


def se3_pose_distance(
    pos1: np.ndarray, quat1: np.ndarray,
    pos2: np.ndarray, quat2: np.ndarray,
    rotation_weight: float = None,
) -> float:
    """Compute weighted distance between two SE(3) poses.

    Thin wrapper around :func:`utils.se3.compute_se3_distance` that accepts
    two poses instead of a pre-computed twist.

    Args:
        pos1, quat1: First pose (quaternion in [x,y,z,w]).
        pos2, quat2: Second pose.
        rotation_weight: Weight for rotation distance in radians
                         (default from constants).

    Returns:
        Scalar distance.
    """
    if rotation_weight is None:
        rotation_weight = constants.RRT_ROTATION_WEIGHT

    T1 = pose_to_SE3(pos1, quat1)
    T2 = pose_to_SE3(pos2, quat2)
    T_rel = np.linalg.inv(T1) @ T2
    twist = SE3_log(T_rel)
    return compute_se3_distance(twist, translation_weight=1.0, rotation_weight=rotation_weight)


def sample_random_pose(
    pos_bounds: np.ndarray,
    start_quat: np.ndarray,
    goal_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a random SE(3) pose within workspace bounds.

    Position is sampled uniformly in the bounding box. Orientation is
    sampled by interpolating between start/goal orientation at a random
    fraction, then perturbing with a small random rotation.

    Args:
        pos_bounds: ``(2, 3)`` array of ``[[x_min, y_min, z_min], [x_max, y_max, z_max]]``.
        start_quat: Start orientation ``[x,y,z,w]``.
        goal_quat: Goal orientation ``[x,y,z,w]``.

    Returns:
        ``(position, quaternion)`` tuple.
    """
    pos = np.random.uniform(pos_bounds[0], pos_bounds[1])

    t = np.random.uniform(0.0, 1.0)
    _, base_quat = interpolate_pose(np.zeros(3), start_quat, np.zeros(3), goal_quat, t)
    r_base = R.from_quat(base_quat)

    noise_angle = np.random.uniform(0, constants.RRT_ORIENTATION_NOISE)
    noise_axis = np.random.randn(3)
    noise_axis /= np.linalg.norm(noise_axis) + 1e-10
    r_noise = R.from_rotvec(noise_angle * noise_axis)

    r_sample = r_base * r_noise
    return pos, r_sample.as_quat()


def steer(
    from_pos: np.ndarray, from_quat: np.ndarray,
    to_pos: np.ndarray, to_quat: np.ndarray,
    step_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Steer from one pose toward another, limited by *step_size*.

    If the distance is less than *step_size*, returns the target pose.

    Args:
        from_pos, from_quat: Current pose.
        to_pos, to_quat: Target pose.
        step_size: Maximum step distance.

    Returns:
        ``(position, quaternion)`` of the steered pose.
    """
    dist = se3_pose_distance(from_pos, from_quat, to_pos, to_quat)
    if dist <= step_size:
        return to_pos.copy(), to_quat.copy()

    t = step_size / dist
    pos, quat = interpolate_pose(from_pos, from_quat, to_pos, to_quat, t)
    return pos, quat


def collision_free_segment(
    env: DualArmEnvironment,
    object_name: str,
    pos1: np.ndarray, quat1: np.ndarray,
    pos2: np.ndarray, quat2: np.ndarray,
    object_geom_ids: np.ndarray,
    robot_mask: np.ndarray,
    obj_joint_addr: int,
    resolution: float = None,
    env_lock=None,
    skip_endpoints: bool = False,
) -> bool:
    """Check if a straight-line SE(3) segment is collision free.

    Discretises the segment and checks each intermediate pose.

    Args:
        env: Environment.
        object_name: Object name.
        pos1, quat1: Start of segment.
        pos2, quat2: End of segment.
        object_geom_ids: Pre-computed geom IDs.
        robot_mask: Boolean mask ``(ngeom,)`` for robot/gripper geoms.
        obj_joint_addr: qpos address for the object's free joint.
        resolution: Check resolution (distance between checks).
        env_lock: Optional threading lock.
        skip_endpoints: If True, skip collision checks at the exact
            start and end of the segment.

    Returns:
        True if the segment is collision-free.
    """
    if resolution is None:
        resolution = constants.RRT_COLLISION_RESOLUTION

    dist = se3_pose_distance(pos1, quat1, pos2, quat2)
    n_checks = max(2, int(np.ceil(dist / resolution)))

    start_idx = 1 if skip_endpoints else 0
    end_idx = n_checks - 1 if skip_endpoints else n_checks

    for i in range(start_idx, end_idx + 1):
        t = i / float(n_checks)
        pos, quat = interpolate_pose(pos1, quat1, pos2, quat2, t)
        if check_object_pose_collision(
            env, object_name, pos, quat,
            object_geom_ids, robot_mask, obj_joint_addr,
            env_lock=env_lock,
        ):
            return False
    return True
