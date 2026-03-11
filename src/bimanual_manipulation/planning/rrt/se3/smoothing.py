"""
SE(3) path post-processing for RRT-planned object trajectories.

Provides SE(3)-specific implementations of the generic post-processing
pipeline defined in :mod:`rrt.path_smoothing`:

* **Shortcutting** — SE(3) collision-aware wrapper around
  :func:`path_smoothing.shortcut_generic`.
* **Bézier smoothing** — composite cubic Bézier in R³ (position) and
  SO(3) (rotation via De Casteljau SLERP cascade) per segment, with
  linear fallback on collision.
* **Densification** — arc-length resampling with SE(3) distance/interpolation.

The joint-space counterparts live in :mod:`rrt.joint.smoothing`.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import List, Optional, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.planning.rrt.se3.collision import (
    check_object_pose_collision,
)
from bimanual_manipulation.planning.rrt.se3.utilities import (
    collision_free_segment,
    se3_pose_distance,
)
from bimanual_manipulation.planning.utils.collision_utils import (
    get_geom_ids_by_prefix,
    get_robot_geom_mask,
)
from bimanual_manipulation.planning.utils.mujoco_utils import get_native_mj_object
from bimanual_manipulation.utils.se3 import interpolate_pose

from bimanual_manipulation.planning.rrt.path_smoothing import (
    shortcut_generic,
    densify_generic,
    bezier_smooth_generic,
    compute_bezier_control_points,
    eval_cubic_bezier,
)
from bimanual_manipulation.planning.rrt.visualizer import RRTVisualizer

logger = logging.getLogger(__name__)


# ============================================================================
# SE(3) shortcutting
# ============================================================================

def _shortcut_path(
    env: DualArmEnvironment,
    object_name: str,
    path: List[Tuple[np.ndarray, np.ndarray]],
    object_geom_ids: np.ndarray,
    robot_mask: np.ndarray,
    obj_joint_addr: int,
    max_iterations: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Iteratively remove redundant waypoints via random shortcutting.

    Delegates to :func:`path_smoothing.shortcut_generic` with an SE(3)
    collision callback.
    """
    def _seg_free(p1, p2):
        return collision_free_segment(
            env, object_name,
            p1[0], p1[1], p2[0], p2[1],
            object_geom_ids, robot_mask, obj_joint_addr,
            skip_endpoints=True,
        )

    return shortcut_generic(
        path,
        segment_free_fn=_seg_free,
        max_iterations=max_iterations,
    )


# ============================================================================
# Quaternion / SO(3) Bézier helpers
# ============================================================================

def _slerp_quaternion(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions [x,y,z,w]."""
    r0 = R.from_quat(q0)
    r1 = R.from_quat(q1)
    slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
    return slerp([t]).as_quat()[0]


def _decasteljau_slerp(quats: List[np.ndarray], t: float) -> np.ndarray:
    """De Casteljau algorithm on the quaternion sphere.

    Generalises the cubic Bézier to SO(3) using successive SLERP
    operations instead of linear combinations.
    """
    if len(quats) == 1:
        return quats[0]
    reduced = []
    for i in range(len(quats) - 1):
        reduced.append(_slerp_quaternion(quats[i], quats[i + 1], t))
    return _decasteljau_slerp(reduced, t)


def _bezier_smooth_one_segment(
    path: List[Tuple[np.ndarray, np.ndarray]],
    seg_idx: int,
    samples_per_segment: int,
    tension: float = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Evaluate the Bézier curve for a **single** segment of the path.

    The full path is needed so that Catmull-Rom tangents at the segment
    endpoints use the neighbouring waypoints for continuity.

    Returns ``samples_per_segment + 1`` (pos, quat) samples.
    """
    if tension is None:
        tension = constants.BEZIER_TENSION
    positions = [p for p, _ in path]
    quaternions = [q for _, q in path]
    n = len(positions)

    # --- Position control points ---
    def _tangent(idx):
        if idx == 0:
            return positions[1] - positions[0]
        elif idx == n - 1:
            return positions[-1] - positions[-2]
        else:
            return positions[idx + 1] - positions[idx - 1]

    P0 = positions[seg_idx]
    P3 = positions[seg_idx + 1]
    chord = np.linalg.norm(P3 - P0)
    t0, t1 = _tangent(seg_idx), _tangent(seg_idx + 1)
    t0n, t1n = np.linalg.norm(t0), np.linalg.norm(t1)
    P1 = P0 + (tension * (chord / t0n) * t0 if t0n > 1e-12 else 0.0)
    P2 = P3 - (tension * (chord / t1n) * t1 if t1n > 1e-12 else 0.0)

    # --- Rotation control quaternions ---
    def _log_rel(qa, qb):
        return (R.from_quat(qa).inv() * R.from_quat(qb)).as_rotvec()

    def _apply_rv(q, rv):
        return (R.from_quat(q) * R.from_rotvec(rv)).as_quat()

    def _rot_tangent(idx):
        if idx == 0:
            return _log_rel(quaternions[0], quaternions[1])
        elif idx == n - 1:
            return _log_rel(quaternions[-2], quaternions[-1])
        else:
            fwd = _log_rel(quaternions[idx], quaternions[idx + 1])
            bwd = _log_rel(quaternions[idx - 1], quaternions[idx])
            return 0.5 * (fwd + bwd)

    q0 = quaternions[seg_idx]
    q3 = quaternions[seg_idx + 1]
    angle_chord = np.linalg.norm(_log_rel(q0, q3))
    rv0, rv1 = _rot_tangent(seg_idx), _rot_tangent(seg_idx + 1)
    rv0n, rv1n = np.linalg.norm(rv0), np.linalg.norm(rv1)
    s0 = tension * (angle_chord / rv0n) if rv0n > 1e-12 else 0.0
    s1 = tension * (angle_chord / rv1n) if rv1n > 1e-12 else 0.0
    q1 = _apply_rv(q0, s0 * rv0)
    q2 = _apply_rv(q3, -s1 * rv1)
    qctrl = [q0, q1, q2, q3]

    # --- Sample ---
    pts: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(samples_per_segment + 1):
        t = k / float(samples_per_segment)
        pos = eval_cubic_bezier(P0, P1, P2, P3, t)
        quat = _decasteljau_slerp(qctrl, t)
        pts.append((pos, quat))
    return pts


def _linear_interpolate_segment(
    p_start: Tuple[np.ndarray, np.ndarray],
    p_end: Tuple[np.ndarray, np.ndarray],
    num_samples: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Linearly interpolate (position lerp + quaternion slerp) between two poses.

    Returns ``num_samples + 1`` points (inclusive of both endpoints).
    """
    pts: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(num_samples + 1):
        t = k / float(num_samples)
        pos, quat = interpolate_pose(
            p_start[0], p_start[1], p_end[0], p_end[1], t,
        )
        pts.append((pos, quat))
    return pts


# ============================================================================
# High-Level API
# ============================================================================

def smooth_path(
    env: DualArmEnvironment,
    object_name: str,
    path: List[Tuple[np.ndarray, np.ndarray]],
    env_lock=None,
    max_smoothing_iterations: int = None,
    rrt_vis: Optional[RRTVisualizer] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Smooth an RRT path using shortcutting followed by hybrid Bézier fitting.

    **Phase 1 – Shortcutting**: Randomly select two waypoints and try to
    connect them directly.  If the direct connection is collision-free,
    remove the intermediate waypoints.

    **Phase 2 – Hybrid Bézier smoothing**: For each segment, fit a cubic
    Bézier curve and collision-check it.  Segments where the Bézier is
    collision-free get the smooth curve; those that collide fall back to
    linear interpolation.

    Args:
        env: Environment.
        object_name: Object name.
        path: List of (position, quaternion) waypoints.
        env_lock: Optional threading lock.
        max_smoothing_iterations: Number of shortcut attempts (Phase 1).
        rrt_vis: Optional :class:`RRTVisualizer` for Meshcat debugging.

    Returns:
        Smoothed path.
    """
    if max_smoothing_iterations is None:
        max_smoothing_iterations = constants.RRT_SMOOTHING_ITERATIONS

    if len(path) <= 2:
        return path

    ctx = env_lock if env_lock is not None else nullcontext()

    with ctx:
        model = env.sim.model
        data = env.sim.data

        object_geom_ids = get_geom_ids_by_prefix(model, object_name)
        robot_mask = get_robot_geom_mask(model)

        try:
            obj_joint_id = model.joint(f"{object_name}_joint0").id
            obj_joint_addr = int(model.jnt_qposadr[obj_joint_id])
        except Exception:
            return path

        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            # Phase 1: shortcutting
            shortcut = _shortcut_path(
                env, object_name, path,
                object_geom_ids, robot_mask, obj_joint_addr,
                max_iterations=max_smoothing_iterations,
            )
            logger.info(
                "Phase 1 shortcutting: %d -> %d waypoints",
                len(path), len(shortcut),
            )

            if rrt_vis is not None:
                rrt_vis.draw_shortcut_path(shortcut)

            # Phase 2: per-segment hybrid Bézier smoothing
            samples_per_seg = max(
                constants.BEZIER_SAMPLES_PER_SEGMENT,
                40 // max(1, len(shortcut) - 1),
            )

            positions = [p for p, _ in shortcut]
            tension = constants.BEZIER_TENSION
            ctrl_segments = compute_bezier_control_points(positions, tension=tension)
            if rrt_vis is not None and ctrl_segments:
                rrt_vis.draw_bezier_control_polygon(ctrl_segments)

            def _se3_collision(pose_pair):
                pos, quat = pose_pair
                return check_object_pose_collision(
                    env, object_name, pos, quat,
                    object_geom_ids, robot_mask, obj_joint_addr,
                )

            def _se3_eval_segment(path_local, seg_idx, sps):
                return _bezier_smooth_one_segment(
                    path_local, seg_idx,
                    samples_per_segment=sps,
                    tension=tension,
                )

            def _se3_linear_fallback(start, end, num_samples):
                return _linear_interpolate_segment(start, end, num_samples)

            result = bezier_smooth_generic(
                shortcut,
                collision_fn=_se3_collision,
                eval_segment_fn=_se3_eval_segment,
                linear_fallback_fn=_se3_linear_fallback,
                samples_per_segment=samples_per_seg,
            )

            if rrt_vis is not None:
                rrt_vis.draw_bezier_curve(result, collision_free=True)
                rrt_vis.draw_smoothed_path(result)

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))

    return result


def densify_path(
    path: List[Tuple[np.ndarray, np.ndarray]],
    num_points: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Resample a sparse SE(3) path to *num_points* evenly-spaced waypoints.

    Uses SE(3) interpolation to produce smooth intermediate poses.
    Delegates to :func:`path_smoothing.densify_generic` with SE(3)
    distance and interpolation callbacks.

    Args:
        path: Sparse path from the planner.
        num_points: Desired number of output waypoints.

    Returns:
        Dense path with *num_points* waypoints.
    """
    if len(path) <= 1 or num_points <= 1:
        return path

    def _dist(s1, s2):
        return se3_pose_distance(s1[0], s1[1], s2[0], s2[1])

    def _interp(s1, s2, t):
        pos, quat = interpolate_pose(s1[0], s1[1], s2[0], s2[1], t)
        return (pos, quat)

    return densify_generic(
        path, num_points,
        distance_fn=_dist,
        interpolate_fn=_interp,
    )
