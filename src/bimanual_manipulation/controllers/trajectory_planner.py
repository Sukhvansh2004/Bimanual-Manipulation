"""Trajectory planning utilities for arm controllers.

Provides quintic polynomial trajectory generation and smoothing in
SE(3) space for smooth, jerk-limited end-effector motion.
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.transforms import StampedPose, TransformManager
from bimanual_manipulation.utils.se3 import (
    SE3_log,
    compute_se3_distance,
    interpolate_pose,
    pose_to_SE3,
)
from bimanual_manipulation.utils.trajectory_math import quintic_scalar

logger = logging.getLogger(__name__)


def plan_quintic_trajectory(
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    num_points: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plan a quintic polynomial trajectory in SE(3) space.

    Uses Lie algebra se(3) for proper coupled rotation-translation
    interpolation with C2 continuity (zero velocity and acceleration
    at start and end).

    Args:
        start_pos: Starting position ``[x, y, z]``.
        start_quat: Starting quaternion ``[x, y, z, w]``.
        target_pos: Target position ``[x, y, z]``.
        target_quat: Target quaternion ``[x, y, z, w]``.
        num_points: Number of waypoints (default from constants).

    Returns:
        ``(positions, quaternions)`` arrays of shape ``(N, 3)`` and ``(N, 4)``.
    """
    if num_points is None:
        num_points = constants.TRAJECTORY_NUM_POINTS

    T_start = pose_to_SE3(start_pos, start_quat)
    T_target = pose_to_SE3(target_pos, target_quat)

    T_start_inv = np.eye(4)
    T_start_inv[:3, :3] = T_start[:3, :3].T
    T_start_inv[:3, 3] = -T_start[:3, :3].T @ T_start[:3, 3]

    T_rel = T_start_inv @ T_target
    twist = SE3_log(T_rel)

    total_se3_distance = compute_se3_distance(
        twist,
        translation_weight=1.0,
        rotation_weight=constants.TRAJECTORY_ORIENTATION_SCALING,
    )

    if total_se3_distance < constants.TRAJECTORY_MIN_DISTANCE:
        return np.array([target_pos]), np.array([target_quat])

    s_trajectory = quintic_scalar(num_points)

    positions = []
    quaternions = []
    for s in s_trajectory:
        pos, quat = interpolate_pose(
            start_pos, start_quat, target_pos, target_quat, s,
        )
        positions.append(pos)
        quaternions.append(quat)

    return np.array(positions), np.array(quaternions)


def smoothen_trajectory(
    trajectory: List[StampedPose],
    num_steps: int,
    tf_manager: TransformManager,
    base_frame: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothen a trajectory with quintic polynomial timing.

    The resulting trajectory has ``num_steps`` waypoints, starting and
    ending at rest with zero acceleration (jerk-limited).

    Args:
        trajectory: List of waypoints to smoothen.
        num_steps: Number of output waypoints.
        tf_manager: For frame conversions.
        base_frame: Target frame for all waypoints.

    Returns:
        ``(positions, quaternions)`` arrays.
    """
    if not trajectory:
        return np.array([]), np.array([])

    base_trajectory = []
    for pose in trajectory:
        if pose.frame_id != base_frame:
            base_trajectory.append(tf_manager.transform_pose(pose, base_frame))
        else:
            base_trajectory.append(pose)

    if len(base_trajectory) == 1:
        pos = np.tile(base_trajectory[0].position, (num_steps, 1))
        quat = np.tile(base_trajectory[0].quaternion, (num_steps, 1))
        return pos, quat

    positions = np.array([p.position for p in base_trajectory])
    quats = np.array([p.quaternion for p in base_trajectory])

    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_dist = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_dist = cumulative_dist[-1]

    if total_dist < 1e-8:
        cumulative_dist = np.linspace(0, 1, len(base_trajectory))
        total_dist = 1.0

    s_profile = quintic_scalar(num_steps) * total_dist

    smooth_pos = []
    smooth_quat = []

    for s in s_profile:
        idx = np.searchsorted(cumulative_dist, s)

        if idx == 0:
            smooth_pos.append(positions[0])
            smooth_quat.append(quats[0])
        elif idx >= len(cumulative_dist):
            smooth_pos.append(positions[-1])
            smooth_quat.append(quats[-1])
        else:
            d0 = cumulative_dist[idx - 1]
            d1 = cumulative_dist[idx]
            alpha = (s - d0) / (d1 - d0) if (d1 - d0) > 1e-9 else 0.0

            p_interp, q_interp = interpolate_pose(
                positions[idx - 1], quats[idx - 1],
                positions[idx], quats[idx], alpha,
            )
            smooth_pos.append(p_interp)
            smooth_quat.append(q_interp)

    return np.array(smooth_pos), np.array(smooth_quat)
