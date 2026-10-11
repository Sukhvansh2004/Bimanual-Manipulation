"""Pure trajectory math utilities.

Shared by both controllers and planning — no dependency on either package.
Provides quintic polynomial profiling, joint-space and SE(3) trajectory
generation, and trajectory resampling/smoothing.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from bimanual_manipulation import constants
from bimanual_manipulation.transforms import StampedPose, TransformManager
from bimanual_manipulation.utils.se3 import (
    SE3_log,
    compute_se3_distance,
    interpolate_pose,
    pose_to_SE3,
)

logger = logging.getLogger(__name__)


def quintic_scalar(n: int) -> np.ndarray:
    """Quintic (jerk-limited) profile from 0 → 1 over *n* steps.

    .. math::
        s(t) = 6t^5 - 15t^4 + 10t^3

    Args:
        n: Number of steps.

    Returns:
        1-D array of shape ``(n,)`` with values in ``[0, 1]``; ``[1.0]``
        if ``n <= 1``.
    """
    if n <= 1:
        return np.array([1.0])
    t = np.linspace(0.0, 1.0, n)
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def plan_quintic_joint_trajectory(
    start: np.ndarray,
    target: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    """Plan a quintic joint-space trajectory from *start* to *target*.

    Args:
        start: Start joint configuration ``(D,)``.
        target: Target joint configuration ``(D,)``.
        num_steps: Number of trajectory waypoints.

    Returns:
        Array of shape ``(num_steps, D)``.
    """
    s = quintic_scalar(num_steps)
    return start[np.newaxis, :] + np.outer(s, target - start)


def velocity_profile_joint_trajectory(
    waypoints: np.ndarray,
    num_steps: Optional[int] = None,
) -> np.ndarray:
    """Apply quintic velocity profiling to a multi-waypoint joint trajectory.

    Re-parameterizes time along the existing path using a quintic
    (jerk-limited) profile so the robot accelerates smoothly from rest
    and decelerates to rest, while *preserving the path shape*.

    Unlike :func:`plan_quintic_joint_trajectory`, which only interpolates
    between two endpoints, this function keeps all intermediate waypoints
    and merely adjusts the timing/spacing.

    Args:
        waypoints: Joint trajectory ``(N, D)`` — the raw waypoints.
        num_steps: Desired output length.  If ``None``, keeps the same
            number of waypoints as the input.

    Returns:
        Array of shape ``(M, D)`` where ``M = num_steps or N``. If
        ``N <= 2`` the input is returned unchanged (``num_steps`` ignored);
        a zero-length path yields ``M`` copies of the first waypoint.
    """
    N = len(waypoints)
    if N <= 2:
        return waypoints

    M = num_steps if num_steps is not None else N

    # Compute cumulative arc-length along the original path
    diffs = np.diff(waypoints, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cum_lengths[-1]

    if total_length < 1e-12:
        return np.tile(waypoints[0], (M, 1))

    # Normalise to [0, 1]
    cum_norm = cum_lengths / total_length

    # Generate quintic-profiled parameter values in [0, 1]
    s = quintic_scalar(M)

    # Interpolate along the original path at the quintic-profiled times
    profiled = np.empty((M, waypoints.shape[1]))
    for k in range(M):
        # Find which segment s[k] falls in
        seg_idx = int(np.clip(
            np.searchsorted(cum_norm, s[k], side="right") - 1,
            0, N - 2,
        ))
        seg_start = cum_norm[seg_idx]
        seg_end = cum_norm[seg_idx + 1]
        t_local = (
            0.0 if seg_end - seg_start < 1e-12
            else (s[k] - seg_start) / (seg_end - seg_start)
        )
        profiled[k] = waypoints[seg_idx] + t_local * diffs[seg_idx]

    return profiled


def resample_joint_trajectory(
    waypoints: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Resample a ``(N, D)`` joint trajectory to *target_length* waypoints.

    Uses cubic interpolation via :func:`scipy.interpolate.interp1d` over
    uniform parameter values.

    Args:
        waypoints: Joint trajectory array of shape ``(N, D)``.
        target_length: Desired number of output waypoints.

    Returns:
        Resampled array of shape ``(target_length, D)``. The input is
        returned as-is if ``N == target_length``; tiled if ``N < 2``.
    """
    n, d = waypoints.shape
    if n == target_length:
        return waypoints
    if n < 2:
        return np.tile(waypoints[0], (target_length, 1))

    old_t = np.linspace(0.0, 1.0, n)
    new_t = np.linspace(0.0, 1.0, target_length)
    return interp1d(old_t, waypoints, axis=0, kind="cubic")(new_t)


# ---------------------------------------------------------------------------
# SE(3) trajectory planning
# ---------------------------------------------------------------------------


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
        num_points: Number of waypoints (default:
            ``constants.TRAJECTORY_NUM_POINTS``).

    Returns:
        ``(positions, quaternions)`` arrays of shape ``(N, 3)`` and ``(N, 4)``
        (quaternions ``[x, y, z, w]``). If the weighted se(3) distance is
        below ``constants.TRAJECTORY_MIN_DISTANCE``, ``N = 1`` and only the
        target pose is returned.
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


def smooth_trajectory(
    trajectory: List[StampedPose],
    num_steps: int,
    tf_manager: TransformManager,
    base_frame: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothen a trajectory with quintic polynomial timing.

    Waypoints are converted to ``base_frame`` and re-sampled along their
    cumulative positional arc length with a quintic profile, interpolating
    each segment with :func:`interpolate_pose`. The result starts and ends
    at rest with zero acceleration (jerk-limited).

    Args:
        trajectory: List of waypoints to smoothen.
        num_steps: Number of output waypoints.
        tf_manager: For frame conversions.
        base_frame: Target frame for all waypoints.

    Returns:
        ``(positions, quaternions)`` arrays of shape ``(num_steps, 3)`` and
        ``(num_steps, 4)`` (quaternions ``[x, y, z, w]``) in ``base_frame``.
        Two empty arrays if ``trajectory`` is empty.
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
