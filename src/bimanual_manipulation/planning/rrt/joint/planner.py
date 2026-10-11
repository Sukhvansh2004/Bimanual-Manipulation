"""
Joint-space RRT-Connect planner.

Plans a collision-free path in the 7-DOF joint space of one arm (arm B)
while the other arm (arm A) is held at a known fixed configuration.

The core RRT-Connect loop is delegated to
:func:`rrt.core.rrt_connect_generic`; this module supplies
joint-space-specific callbacks (distance, steer, sample, collision).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
import threading
from typing import List, Optional

import mujoco
import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.planning.rrt.core import rrt_connect_generic
from bimanual_manipulation.utils.mujoco_collision import (
    ensure_geom_caches,
    get_geom_ids_for_patterns,
)
from bimanual_manipulation.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_joint_limits,
    get_native_mj_object,
    get_object_free_joint_qpos_indices,
)
from bimanual_manipulation.planning.rrt.joint.collision import (
    check_arm_config_collision,
    collision_free_joint_segment,
)
from bimanual_manipulation.planning.rrt.joint.utilities import (
    joint_distance,
    steer,
    sample_random_config,
    make_ee_pos_fn,
    build_temporal_samples,
)
from bimanual_manipulation.planning.rrt.visualizer import RRTVisualizer

logger = logging.getLogger(__name__)


# ============================================================================
# Data containers
# ============================================================================


@dataclass
class JointRRTResult:
    """Result of joint-space RRT-Connect planning.

    Attributes:
        success: Whether a collision-free path was found.
        path: List of ``(7,)`` joint configurations from start to goal;
            empty on failure.
        iterations: Number of RRT iterations run (``0`` for a direct path).
        tree_a_size: Size of the internal tree_a at termination.
        tree_b_size: Size of the internal tree_b at termination.
    """

    success: bool
    path: List[np.ndarray] = field(default_factory=list)
    iterations: int = 0
    tree_a_size: int = 0
    tree_b_size: int = 0


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
    env_lock: Optional[threading.RLock] = None,
    max_iterations: int = constants.ARM_RRT_MAX_ITERATIONS,
    step_size: float = constants.ARM_RRT_STEP_SIZE,
    goal_bias: float = constants.ARM_RRT_GOAL_BIAS,
    rrt_vis: Optional[RRTVisualizer] = None,
    q_other_arm_trajectory: Optional[np.ndarray] = None,
    max_temporal_samples: int = constants.ARM_RRT_TEMPORAL_SAMPLES,
    manipulated_object: Optional[str] = None,
    object_qpos_trajectory: Optional[np.ndarray] = None,
) -> JointRRTResult:
    """Plan a collision-free joint-space path for arm B using RRT-Connect.

    Arm A is held at *q_other_arm*.  Collision checking detects contacts
    between arm B and arm A, the environment, and (optionally) a
    grasped object.

    When *q_other_arm_trajectory* is provided, the planner checks
    arm B's configurations against multiple sampled arm A positions
    (temporal collision checking).  This prevents arm B from colliding
    with arm A at intermediate positions along its trajectory.

    The direct start-to-goal segment (including endpoints) is tried
    first; if clear, a two-waypoint path is returned without running
    RRT. Random samples are drawn within arm B's joint limits shrunk by
    :data:`constants.ARM_RRT_JOINT_MARGIN`. The simulator ``qpos``/``qvel``
    are restored (followed by ``mj_forward``) before returning.

    Args:
        env: Simulation environment.
        robot_id_b: Robot ID of the arm to plan for.
        start_q: Start joint configuration ``(7,)``.
        goal_q: Goal joint configuration ``(7,)``.
        q_other_arm: Fixed joint configuration of the other arm.
            Used when *q_other_arm_trajectory* is not provided.
        robot_id_a: Robot ID of the other arm.
        env_lock: Optional threading lock held for the planning call.
        max_iterations: Maximum RRT-Connect iterations.
        step_size: Step size (joint-space L2, radians) for tree extension.
        goal_bias: Probability of sampling the goal.
        rrt_vis: Optional :class:`RRTVisualizer` for Meshcat debugging.
            Tree edges and paths are projected to 3-D end-effector
            positions via forward kinematics.
        q_other_arm_trajectory: Optional ``(N, 7)`` trajectory of the
            other arm.  When provided, the collision checker validates
            arm B against multiple sampled positions from this
            trajectory rather than a single static configuration.
        max_temporal_samples: Maximum number of arm A configurations
            sampled from *q_other_arm_trajectory* for temporal
            collision checking.  Defaults to
            ``constants.ARM_RRT_TEMPORAL_SAMPLES``.
        manipulated_object: Name of the object arm A is carrying; only
            used (to find its free-joint qpos indices) when
            *object_qpos_trajectory* is given.
        object_qpos_trajectory: Optional ``(N, 7)`` free-joint qpos
            trajectory of the object **arm A is carrying**, aligned
            with *q_other_arm_trajectory*.  Keeps the carried object
            an obstacle at every sampled instant.

    Returns:
        :class:`JointRRTResult` with success flag and path. ``success``
        is ``False`` (empty path) if no geoms are found for arm B.
    """
    # ── Build (arm A config, object pose) snapshots ────────────────
    #    When a trajectory is provided, sample representative configs
    #    for conservative temporal collision avoidance.
    temporal_samples = build_temporal_samples(
        q_other_arm, q_other_arm_trajectory, max_temporal_samples,
        object_qpos_trajectory,
    )
    if q_other_arm_trajectory is not None and len(q_other_arm_trajectory) > 1:
        logger.info(
            "Temporal collision checking: using %d sampled arm A "
            "configs from trajectory of length %d%s.",
            len(temporal_samples), len(q_other_arm_trajectory),
            " (with carried-object poses)"
            if object_qpos_trajectory is not None else "",
        )

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
        ensure_geom_caches(model)
        prefixes = [f"robot{robot_id_b}_", f"gripper{robot_id_b}_"]
        arm_b_geom_ids_list = get_geom_ids_for_patterns(prefixes)
        
        # qpos slot used to re-pose the object arm A is carrying.
        # The object rides with arm A, never with arm B, so its geoms
        # stay out of arm B's set — they are obstacles, not payload.
        obj_qpos_idx = (
            get_object_free_joint_qpos_indices(native_model, manipulated_object)
            if object_qpos_trajectory is not None else None
        )
        if object_qpos_trajectory is not None and obj_qpos_idx is None:
            logger.warning(
                "No free joint found for '%s' — carried-object poses "
                "cannot be applied during collision checking.",
                manipulated_object,
            )

        arm_b_geom_ids = np.array(arm_b_geom_ids_list, dtype=int)
        if len(arm_b_geom_ids) == 0:
            logger.error("No geoms found for robot%d", robot_id_b)
            return JointRRTResult(success=False)

        # ── Backup simulation state ───────────────────────────────────
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            # ── Fast path: try direct connection ──────────────────────
            direct_ok = all(
                collision_free_joint_segment(
                    env, start_q, goal_q,
                    q_a, qpos_idx_a, qpos_idx_b,
                    native_model, native_data, arm_b_geom_ids,
                    skip_endpoints=False,
                    obj_qpos_idx=obj_qpos_idx, obj_qpos=obj_q,
                )
                for q_a, obj_q in temporal_samples
            )
            if direct_ok:
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
                """Check arm B at *q* against every temporal snapshot.

                Args:
                    q: Arm B joint configuration.

                Returns:
                    ``True`` if any snapshot is in collision.
                """
                return any(
                    check_arm_config_collision(
                        env, q, qpos_idx_b,
                        q_a, qpos_idx_a,
                        native_model, native_data, arm_b_geom_ids,
                        obj_qpos_idx, obj_q,
                    )
                    for q_a, obj_q in temporal_samples
                )

            def _seg_free(q1, q2):
                """Check a segment's interior against every temporal snapshot.

                Args:
                    q1: Segment start configuration.
                    q2: Segment end configuration.

                Returns:
                    ``True`` if the segment is collision-free for all
                    snapshots.
                """
                return all(
                    collision_free_joint_segment(
                        env, q1, q2,
                        q_a, qpos_idx_a, qpos_idx_b,
                        native_model, native_data, arm_b_geom_ids,
                        skip_endpoints=True,
                        obj_qpos_idx=obj_qpos_idx, obj_qpos=obj_q,
                    )
                    for q_a, obj_q in temporal_samples
                )

            def _sample():
                """Sample a random arm B configuration within the margined limits.

                Returns:
                    Joint configuration.
                """
                return sample_random_config(jnt_lo, jnt_hi)

            # ── Visualisation hooks (FK to 3-D EE position) ──────────
            _on_extend = None
            _on_collision = None
            _on_iteration = None
            _ee_pos_fn = None

            if rrt_vis is not None:
                _ee_pos_fn = make_ee_pos_fn(
                    data, qpos_idx_b, native_model, native_data,
                    robot_id_b,
                )
                if _ee_pos_fn is not None:
                    start_ee = _ee_pos_fn(start_q)
                    goal_ee = _ee_pos_fn(goal_q)
                    rrt_vis.draw_start_goal(start_ee, goal_ee)
                    logger.info(
                        "RRT visualisation active: start_ee=%s, goal_ee=%s",
                        start_ee, goal_ee,
                    )

                    def _on_extend(parent_q, new_q, is_start):
                        """Draw a new tree edge projected to EE positions.

                        Args:
                            parent_q: Parent joint configuration.
                            new_q: New joint configuration.
                            is_start: ``True`` if the edge belongs to the
                                start tree (drawn as tree_a).
                        """
                        rrt_vis.draw_tree_edge(
                            _ee_pos_fn(parent_q),
                            _ee_pos_fn(new_q),
                            is_tree_a=is_start,
                        )

                    def _on_collision(q):
                        """Draw a rejected configuration at its EE position.

                        Args:
                            q: Rejected joint configuration.
                        """
                        rrt_vis.draw_collision_sample(_ee_pos_fn(q))
                else:
                    logger.warning(
                        "RRT visualisation requested but EE position "
                        "function unavailable (site not found for robot%d).",
                        robot_id_b,
                    )

                def _on_iteration(it, ta_size, tb_size):
                    """Forward iteration progress to the visualizer logger.

                    Args:
                        it: Iteration index.
                        ta_size: Size of tree_a.
                        tb_size: Size of tree_b.
                    """
                    rrt_vis.log_iteration(it, ta_size, tb_size)

            success, path, iters, ta, tb = rrt_connect_generic(
                start_state=start_q.copy(),
                goal_state=goal_q.copy(),
                distance_fn=joint_distance,
                steer_fn=steer,
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
                log_interval=200,
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
