"""Pure trajectory math utilities.

Shared by both controllers and planning — no dependency on either package.
Provides quintic polynomial profiling, joint-space trajectory generation,
and trajectory resampling.
"""

from typing import Optional

import numpy as np
from scipy.interpolate import interp1d


def quintic_scalar(n: int) -> np.ndarray:
    """Quintic (jerk-limited) profile from 0 → 1 over *n* steps.

    .. math::
        s(t) = 6t^5 - 15t^4 + 10t^3

    Args:
        n: Number of steps.

    Returns:
        1-D array of shape ``(n,)`` with values in ``[0, 1]``.
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
        Array of shape ``(M, D)`` where ``M = num_steps or N``.
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

    Uses cubic interpolation via :func:`scipy.interpolate.interp1d`.

    Args:
        waypoints: Joint trajectory array of shape ``(N, D)``.
        target_length: Desired number of output waypoints.

    Returns:
        Resampled array of shape ``(target_length, D)``.
    """
    n, d = waypoints.shape
    if n == target_length:
        return waypoints
    if n < 2:
        return np.tile(waypoints[0], (target_length, 1))

    old_t = np.linspace(0.0, 1.0, n)
    new_t = np.linspace(0.0, 1.0, target_length)
    return interp1d(old_t, waypoints, axis=0, kind="cubic")(new_t)
