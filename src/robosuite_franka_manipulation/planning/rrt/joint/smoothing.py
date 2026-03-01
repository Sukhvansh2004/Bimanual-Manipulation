"""
Joint-space path post-processing.

Provides shortcutting, Bézier smoothing, and densification for
joint-space paths, delegating to the generic utilities in
:mod:`rrt.path_smoothing`.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional

import mujoco
import numpy as np

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.planning.utils.collision_utils import (
    _ensure_geom_caches,
    _get_ids_for_prefixes,
)
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_native_mj_object,
)
from robosuite_franka_manipulation.planning.rrt.joint.collision import (
    check_arm_config_collision,
    collision_free_joint_segment,
)
from robosuite_franka_manipulation.planning.rrt.path_smoothing import (
    shortcut_generic,
    densify_generic,
    bezier_smooth_generic,
    compute_bezier_control_points,
    eval_cubic_bezier,
)

logger = logging.getLogger(__name__)


def shortcut_joint_path(
    env: DualArmEnvironment,
    path: List[np.ndarray],
    q_other_arm: np.ndarray,
    robot_id_a: int,
    robot_id_b: int,
    env_lock=None,
    max_iterations: int = None,
) -> List[np.ndarray]:
    """Shortcut-smooth a joint-space path by randomly removing waypoints.

    Randomly picks two non-adjacent waypoints and tries a direct
    connection.  If the segment is collision-free, all intermediate
    waypoints are removed.

    Args:
        env: Simulation environment.
        path: List of joint configurations ``(7,)`` from RRT.
        q_other_arm: Fixed configuration of the other arm ``(7,)``.
        robot_id_a: Robot ID of the other arm.
        robot_id_b: Robot ID of the planned arm.
        env_lock: Optional threading lock.
        max_iterations: Number of random shortcut attempts.

    Returns:
        Smoothed path (list of joint configurations).
    """
    if max_iterations is None:
        max_iterations = constants.ARM_RRT_SMOOTHING_ITERATIONS

    if len(path) <= 2:
        return list(path)

    ctx = env_lock if env_lock is not None else nullcontext()

    with ctx:
        model = env.sim.model
        data = env.sim.data
        native_model = get_native_mj_object(model)
        native_data = get_native_mj_object(data)

        jids_a = get_arm_joint_ids(native_model, robot_id_a)
        jids_b = get_arm_joint_ids(native_model, robot_id_b)
        qpos_idx_a = get_joint_qpos_indices(native_model, jids_a)
        qpos_idx_b = get_joint_qpos_indices(native_model, jids_b)

        _ensure_geom_caches(model)
        arm_b_geom_ids = np.array(
            _get_ids_for_prefixes(
                [f"robot{robot_id_b}_", f"gripper{robot_id_b}_"]
            ),
            dtype=int,
        )

        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            def _seg_free(q1, q2):
                return collision_free_joint_segment(
                    env, q1, q2,
                    q_other_arm, qpos_idx_a, qpos_idx_b,
                    native_model, native_data, arm_b_geom_ids,
                    skip_endpoints=True,
                )

            return shortcut_generic(
                path,
                segment_free_fn=_seg_free,
                max_iterations=max_iterations,
            )
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(native_model, native_data)


def densify_joint_path(
    path: List[np.ndarray],
    num_steps: int,
) -> np.ndarray:
    """Resample a sparse joint-space path to *num_steps* waypoints.

    Uses linear interpolation along the piecewise-linear path in joint
    space.

    Args:
        path: Sparse list of joint configurations.
        num_steps: Desired number of output waypoints.

    Returns:
        Array of shape ``(num_steps, D)`` with D = joint dimension.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if len(path) == 0:
        raise ValueError("path is empty")
    if len(path) == 1:
        return np.tile(path[0], (num_steps, 1))

    def _dist(q1, q2):
        return float(np.linalg.norm(q1 - q2))

    def _interp(q1, q2, t):
        return q1 + t * (q2 - q1)

    result = densify_generic(
        path, num_steps, distance_fn=_dist, interpolate_fn=_interp,
    )
    return np.array(result)


def bezier_smooth_joint_path(
    env: DualArmEnvironment,
    path: List[np.ndarray],
    q_other_arm: np.ndarray,
    robot_id_a: int,
    robot_id_b: int,
    env_lock=None,
    samples_per_segment: int = None,
    tension: float = None,
) -> List[np.ndarray]:
    """Smooth a joint-space path with per-segment cubic Bézier curves.

    Delegates to :func:`path_smoothing.bezier_smooth_generic` with
    joint-space-specific callbacks for collision checking, Bézier
    evaluation, and linear interpolation fallback.

    Args:
        env: Simulation environment.
        path: Shortcut-smoothed list of joint configurations ``(D,)``.
        q_other_arm: Fixed configuration of the other arm ``(D,)``.
        robot_id_a: Robot ID of the other arm.
        robot_id_b: Robot ID of the planned arm.
        env_lock: Optional threading lock.
        samples_per_segment: Number of dense samples per segment.
        tension: Bézier tension parameter (0 → linear, higher → curvier).

    Returns:
        Densely sampled smoothed path (list of joint configurations).
    """
    if samples_per_segment is None:
        samples_per_segment = constants.BEZIER_SAMPLES_PER_SEGMENT
    if tension is None:
        tension = constants.BEZIER_TENSION

    if len(path) <= 1:
        return list(path)

    ctx = env_lock if env_lock is not None else nullcontext()

    with ctx:
        model = env.sim.model
        data = env.sim.data
        native_model = get_native_mj_object(model)
        native_data = get_native_mj_object(data)

        jids_a = get_arm_joint_ids(native_model, robot_id_a)
        jids_b = get_arm_joint_ids(native_model, robot_id_b)
        qpos_idx_a = get_joint_qpos_indices(native_model, jids_a)
        qpos_idx_b = get_joint_qpos_indices(native_model, jids_b)

        _ensure_geom_caches(model)
        arm_b_geom_ids = np.array(
            _get_ids_for_prefixes(
                [f"robot{robot_id_b}_", f"gripper{robot_id_b}_"]
            ),
            dtype=int,
        )

        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            def _collision(q):
                return check_arm_config_collision(
                    env, q, qpos_idx_b,
                    q_other_arm, qpos_idx_a,
                    native_model, native_data, arm_b_geom_ids,
                )

            def _eval_segment(path_local, seg_idx, sps):
                ctrl_segs = compute_bezier_control_points(
                    path_local, tension=tension,
                )
                C0, C1, C2, C3 = ctrl_segs[seg_idx]
                return [
                    eval_cubic_bezier(C0, C1, C2, C3, k / float(sps))
                    for k in range(sps + 1)
                ]

            def _linear_fallback(start, end, num_samples):
                return [
                    start + (k / float(num_samples)) * (end - start)
                    for k in range(num_samples + 1)
                ]

            return bezier_smooth_generic(
                path,
                collision_fn=_collision,
                eval_segment_fn=_eval_segment,
                linear_fallback_fn=_linear_fallback,
                samples_per_segment=samples_per_segment,
            )

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(native_model, native_data)
