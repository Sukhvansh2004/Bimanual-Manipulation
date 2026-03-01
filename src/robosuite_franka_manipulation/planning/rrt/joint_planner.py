"""
Joint-space RRT-Connect planner for collision-free arm trajectory planning.

Plans a collision-free path in the 7-DOF joint space of one arm (arm B)
while the other arm (arm A) is held at a known fixed configuration.
Collision checking is performed against:

* **Inter-arm collisions** — arm B vs arm A + gripper geoms.
* **Environment obstacles** — arm B vs tables, shelves, walls, etc.
* **Object avoidance** — arm B vs a grasped object held by arm A.

The core RRT-Connect loop is delegated to
:func:`rrt_planner.rrt_connect_generic`; this module supplies
joint-space-specific callbacks (distance, steer, sample, collision)
and a post-processing pipeline:

1. **Shortcutting** — via :func:`path_smoothing.shortcut_generic`.
2. **Bézier smoothing** — per-segment cubic Bézier in R^D with
   collision checking and linear fallback.
3. **Densification** — via :func:`path_smoothing.densify_generic`.

Optional Meshcat visualization is supported via the same
:class:`RRTVisualizer` used by the SE(3) planner (tree edges and paths
are projected to 3-D end-effector positions through forward kinematics).

Usage example::

    result = rrt_connect_joint_space(
        env=env,
        robot_id_b=1,
        start_q=current_q,
        goal_q=target_q,
        q_other_arm=traj_a[-1],
        robot_id_a=0,
    )
    if result.success:
        smoothed = shortcut_joint_path(env, ..., result.path, ...)
        dense = densify_joint_path(smoothed, 200)
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mujoco
import numpy as np

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.planning.rrt.core import rrt_connect_generic
from robosuite_franka_manipulation.planning.utils.collision_utils import (
    _ensure_geom_caches,
    _get_ids_for_prefixes,
    in_collision,
)
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_joint_limits,
    get_eef_site_name,
    get_native_mj_object,
    set_qpos,
)
from robosuite_franka_manipulation.planning.rrt.path_smoothing import (
    shortcut_generic,
    densify_generic,
    bezier_smooth_generic,
    _compute_bezier_control_points,
    _eval_cubic_bezier,
)
from robosuite_franka_manipulation.planning.rrt.visualizer import RRTVisualizer

logger = logging.getLogger(__name__)


# ============================================================================
# Data containers
# ============================================================================


@dataclass
class JointRRTResult:
    """Result of joint-space RRT-Connect planning."""

    success: bool
    path: List[np.ndarray] = field(default_factory=list)
    iterations: int = 0
    tree_a_size: int = 0
    tree_b_size: int = 0


# ============================================================================
# Collision checking in joint space
# ============================================================================

def _check_arm_config_collision(
    env: DualArmEnvironment,
    q_b: np.ndarray,
    qpos_idx_b: np.ndarray,
    q_a: np.ndarray,
    qpos_idx_a: np.ndarray,
    native_model,
    native_data,
    arm_b_geom_ids: np.ndarray,
    *,
    contact_threshold: float = constants.COLLISION_CONTACT_THRESHOLD,
) -> bool:
    """Check whether arm B at *q_b* collides with anything.

    Sets both arms' configurations in MuJoCo, runs forward kinematics,
    and scans the contact buffer for contacts involving arm B.

    Contacts between arm A and the environment / object are **not**
    flagged — only contacts where at least one geom belongs to arm B.

    The caller is responsible for state backup/restore.

    Args:
        env: Simulation environment.
        q_b: Candidate joint configuration for arm B ``(7,)``.
        qpos_idx_b: qpos indices of arm B's joints.
        q_a: Fixed joint configuration of arm A ``(7,)``.
        qpos_idx_a: qpos indices of arm A's joints.
        native_model: Low-level ``mujoco.MjModel``.
        native_data: Low-level ``mujoco.MjData``.
        arm_b_geom_ids: Array of geom IDs belonging to arm B.
        contact_threshold: Contact distance threshold.

    Returns:
        ``True`` if arm B has any collision contact.
    """
    data = env.sim.data
    model = env.sim.model

    set_qpos(data, qpos_idx_a, q_a)
    set_qpos(data, qpos_idx_b, q_b)
    mujoco.mj_fwdPosition(native_model, native_data)

    ncon = int(data.ncon)
    if ncon <= 0:
        return False

    _ensure_geom_caches(model)

    g1 = np.asarray(data.contact.geom1[:ncon])
    g2 = np.asarray(data.contact.geom2[:ncon])
    dist = np.asarray(data.contact.dist[:ncon])

    # Distance filter
    mask = dist <= contact_threshold
    if not np.any(mask):
        return False

    # Only contacts involving arm B
    b_mask = np.zeros(model.ngeom, dtype=bool)
    b_mask[arm_b_geom_ids] = True
    involves_b = b_mask[g1] | b_mask[g2]
    mask &= involves_b

    # Exclude arm A <-> object contacts (arm A is grasping, those are expected)
    # — not needed because we only look at contacts involving arm B.

    return bool(np.any(mask))


def _collision_free_joint_segment(
    env: DualArmEnvironment,
    q_start: np.ndarray,
    q_end: np.ndarray,
    q_a: np.ndarray,
    qpos_idx_a: np.ndarray,
    qpos_idx_b: np.ndarray,
    native_model,
    native_data,
    arm_b_geom_ids: np.ndarray,
    resolution: float = None,
    skip_endpoints: bool = False,
) -> bool:
    """Check if a straight-line joint-space segment is collision-free.

    Linearly interpolates between *q_start* and *q_end* at *resolution*
    angular increments and checks each for collisions.

    Args:
        env: Simulation environment.
        q_start: Start joint configuration ``(7,)``.
        q_end: End joint configuration ``(7,)``.
        q_a: Fixed arm A configuration ``(7,)``.
        qpos_idx_a: qpos indices of arm A's joints.
        qpos_idx_b: qpos indices of arm B's joints.
        native_model: ``mujoco.MjModel``.
        native_data: ``mujoco.MjData``.
        arm_b_geom_ids: Geom IDs belonging to arm B.
        resolution: Angular resolution (radians) between checks.
        skip_endpoints: If ``True``, skip checks at exact start/end.

    Returns:
        ``True`` if the segment is collision-free.
    """
    if resolution is None:
        resolution = constants.ARM_RRT_COLLISION_RESOLUTION

    diff = q_end - q_start
    dist = np.linalg.norm(diff)
    n_checks = max(2, int(np.ceil(dist / resolution)))

    start_idx = 1 if skip_endpoints else 0
    end_idx = n_checks - 1 if skip_endpoints else n_checks

    for i in range(start_idx, end_idx + 1):
        t = i / float(n_checks)
        q_interp = q_start + t * diff
        if _check_arm_config_collision(
            env, q_interp, qpos_idx_b, q_a, qpos_idx_a,
            native_model, native_data, arm_b_geom_ids,
        ):
            return False
    return True


# ============================================================================
# Joint-space utilities
# ============================================================================

def _joint_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """L2 distance in joint space."""
    return float(np.linalg.norm(q1 - q2))


def _steer(
    from_q: np.ndarray,
    to_q: np.ndarray,
    step_size: float,
) -> np.ndarray:
    """Steer from *from_q* toward *to_q*, limited by *step_size*.

    Returns *to_q* directly if the distance is within *step_size*.
    """
    diff = to_q - from_q
    dist = np.linalg.norm(diff)
    if dist <= step_size:
        return to_q.copy()
    return from_q + (step_size / dist) * diff


def _sample_random_config(
    joint_lo: np.ndarray,
    joint_hi: np.ndarray,
) -> np.ndarray:
    """Uniformly sample a random joint configuration within limits."""
    return np.random.uniform(joint_lo, joint_hi)


# ============================================================================
# FK helper for visualisation
# ============================================================================


def _make_ee_pos_fn(data, qpos_idx_b, native_model, native_data, robot_id_b):
    """Return a callable ``q -> pos`` mapping joint configs to 3-D EE positions.

    Uses the end-effector *site* registered for the given robot.  The
    returned function sets ``qpos``, runs :func:`mujoco.mj_fwdPosition`,
    and reads back the site position.  **Must** be called within a sim
    state backup / restore context.

    Returns ``None`` if the site cannot be resolved.
    """
    site_name = get_eef_site_name(robot_id_b)
    try:
        # Validate that the site exists
        native_model.site(site_name)
    except (KeyError, ValueError):
        logger.warning(
            "EE site '%s' not found for robot%d — "
            "visualisation will be limited.",
            site_name, robot_id_b,
        )
        return None

    def _ee_pos(q: np.ndarray) -> np.ndarray:
        set_qpos(data, qpos_idx_b, q)
        mujoco.mj_fwdPosition(native_model, native_data)
        return np.array(native_data.site(site_name).xpos, copy=True)

    return _ee_pos


# ============================================================================
# RRT-Connect core
# ============================================================================

def rrt_connect_joint_space(
    env: DualArmEnvironment,
    robot_id_b: int,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    q_other_arm: np.ndarray,
    robot_id_a: int,
    env_lock=None,
    manipulated_object: Optional[str] = None,
    max_iterations: int = None,
    step_size: float = None,
    goal_bias: float = None,
    rrt_vis: Optional[RRTVisualizer] = None,
) -> JointRRTResult:
    """Plan a collision-free joint-space path for arm B using RRT-Connect.

    Arm A is held at *q_other_arm*.  Collision checking detects contacts
    between arm B and arm A, the environment, and (optionally) a
    grasped object.

    Args:
        env: Simulation environment.
        robot_id_b: Robot ID of the arm to plan for.
        start_q: Start joint configuration ``(7,)``.
        goal_q: Goal joint configuration ``(7,)``.
        q_other_arm: Fixed joint configuration of the other arm.
        robot_id_a: Robot ID of the other arm.
        env_lock: Optional threading lock.
        manipulated_object: Object held by arm A (contacts between arm A
            gripper and this object are expected and ignored).
        max_iterations: Maximum RRT-Connect iterations.
        step_size: Step size for tree extension (radians).
        goal_bias: Probability of sampling the goal.
        rrt_vis: Optional :class:`RRTVisualizer` for Meshcat debugging.
            Tree edges and paths are projected to 3-D end-effector
            positions via forward kinematics.

    Returns:
        :class:`JointRRTResult` with success flag and path.
    """
    if max_iterations is None:
        max_iterations = constants.ARM_RRT_MAX_ITERATIONS
    if step_size is None:
        step_size = constants.ARM_RRT_STEP_SIZE
    if goal_bias is None:
        goal_bias = constants.ARM_RRT_GOAL_BIAS

    ctx = env_lock if env_lock is not None else nullcontext()

    with ctx:
        model = env.sim.model
        data = env.sim.data
        native_model = get_native_mj_object(model)
        native_data = get_native_mj_object(data)

        # ── Index maps ────────────────────────────────────────────────
        jids_a = get_arm_joint_ids(native_model, robot_id_a)
        jids_b = get_arm_joint_ids(native_model, robot_id_b)
        qpos_idx_a = get_joint_qpos_indices(native_model, jids_a)
        qpos_idx_b = get_joint_qpos_indices(native_model, jids_b)

        # ── Joint limits for sampling ─────────────────────────────────
        jnt_lo, jnt_hi = get_joint_limits(native_model, jids_b)
        margin = constants.ARM_RRT_JOINT_MARGIN
        jnt_lo = jnt_lo + margin
        jnt_hi = jnt_hi - margin

        # ── Arm B geom IDs ────────────────────────────────────────────
        _ensure_geom_caches(model)
        arm_b_geom_ids = np.array(
            _get_ids_for_prefixes(
                [f"robot{robot_id_b}_", f"gripper{robot_id_b}_"]
            ),
            dtype=int,
        )
        if len(arm_b_geom_ids) == 0:
            logger.error("No geoms found for robot%d", robot_id_b)
            return JointRRTResult(success=False)

        # ── Backup simulation state ───────────────────────────────────
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            # ── Fast path: try direct connection ──────────────────────
            if _collision_free_joint_segment(
                env, start_q, goal_q,
                q_other_arm, qpos_idx_a, qpos_idx_b,
                native_model, native_data, arm_b_geom_ids,
                skip_endpoints=False,
            ):
                logger.info(
                    "Joint RRT: direct path is collision-free, "
                    "no RRT needed."
                )
                return JointRRTResult(
                    success=True,
                    path=[start_q.copy(), goal_q.copy()],
                    iterations=0,
                    tree_a_size=1,
                    tree_b_size=1,
                )

            logger.info(
                "Joint RRT: direct path blocked.  Running RRT-Connect "
                "(max_iter=%d, step=%.3f rad) for robot%d...",
                max_iterations, step_size, robot_id_b,
            )

            # ── Domain callbacks for generic RRT-Connect ─────────────
            def _collision(q):
                return _check_arm_config_collision(
                    env, q, qpos_idx_b,
                    q_other_arm, qpos_idx_a,
                    native_model, native_data, arm_b_geom_ids,
                )

            def _seg_free(q1, q2):
                return _collision_free_joint_segment(
                    env, q1, q2,
                    q_other_arm, qpos_idx_a, qpos_idx_b,
                    native_model, native_data, arm_b_geom_ids,
                    skip_endpoints=True,
                )

            def _sample():
                return _sample_random_config(jnt_lo, jnt_hi)

            # ── Visualisation hooks (FK to 3-D EE position) ──────────
            _on_extend = None
            _on_collision = None
            _on_iteration = None
            _ee_pos_fn = None

            if rrt_vis is not None:
                _ee_pos_fn = _make_ee_pos_fn(
                    data, qpos_idx_b, native_model, native_data,
                    robot_id_b,
                )
                if _ee_pos_fn is not None:
                    start_ee = _ee_pos_fn(start_q)
                    goal_ee = _ee_pos_fn(goal_q)
                    rrt_vis.draw_start_goal(start_ee, goal_ee)

                    def _on_extend(parent_q, new_q, is_start):
                        rrt_vis.draw_tree_edge(
                            _ee_pos_fn(parent_q),
                            _ee_pos_fn(new_q),
                            is_tree_a=is_start,
                        )

                    def _on_collision(q):
                        rrt_vis.draw_collision_sample(_ee_pos_fn(q))

                def _on_iteration(it, ta_size, tb_size):
                    rrt_vis.log_iteration(it, ta_size, tb_size)

            success, path, iters, ta, tb = rrt_connect_generic(
                start_state=start_q.copy(),
                goal_state=goal_q.copy(),
                distance_fn=_joint_distance,
                steer_fn=_steer,
                sample_fn=_sample,
                collision_fn=_collision,
                segment_free_fn=_seg_free,
                copy_fn=lambda q: q.copy(),
                step_size=step_size,
                goal_bias=goal_bias,
                max_iterations=max_iterations,
                connect_max_steps=constants.ARM_RRT_CONNECT_MAX_STEPS,
                on_extend=_on_extend,
                on_collision=_on_collision,
                on_iteration=_on_iteration,
            )

            if success:
                logger.info(
                    "Joint RRT-Connect found path in %d iterations "
                    "(tree sizes: %d, %d)", iters, ta, tb,
                )
                if rrt_vis is not None and _ee_pos_fn is not None:
                    rrt_vis.draw_path(path, pos_fn=_ee_pos_fn)
            else:
                logger.warning(
                    "Joint RRT-Connect failed after %d iterations "
                    "for robot%d.", max_iterations, robot_id_b,
                )

            return JointRRTResult(
                success=success,
                path=path,
                iterations=iters,
                tree_a_size=ta,
                tree_b_size=tb,
            )

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(native_model, native_data)


# ============================================================================
# Path post-processing
# ============================================================================

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
                return _collision_free_joint_segment(
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
                return _check_arm_config_collision(
                    env, q, qpos_idx_b,
                    q_other_arm, qpos_idx_a,
                    native_model, native_data, arm_b_geom_ids,
                )

            def _eval_segment(path_local, seg_idx, sps):
                ctrl_segs = _compute_bezier_control_points(
                    path_local, tension=tension,
                )
                C0, C1, C2, C3 = ctrl_segs[seg_idx]
                return [
                    _eval_cubic_bezier(C0, C1, C2, C3, k / float(sps))
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
