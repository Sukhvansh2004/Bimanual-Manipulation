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
from typing import List, Optional

import mujoco
import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.planning.rrt.core import rrt_connect_generic
from bimanual_manipulation.planning.utils.collision_utils import (
    _ensure_geom_caches,
    _get_ids_for_prefixes,
)
from bimanual_manipulation.planning.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_joint_qpos_indices,
    get_joint_limits,
    get_native_mj_object,
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
)
from bimanual_manipulation.planning.rrt.visualizer import RRTVisualizer

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
    max_iterations: int = None,
    step_size: float = None,
    goal_bias: float = None,
    rrt_vis: Optional[RRTVisualizer] = None,
    q_other_arm_trajectory: Optional[np.ndarray] = None,
    max_temporal_samples: int = None,
) -> JointRRTResult:
    """Plan a collision-free joint-space path for arm B using RRT-Connect.

    Arm A is held at *q_other_arm*.  Collision checking detects contacts
    between arm B and arm A, the environment, and (optionally) a
    grasped object.

    When *q_other_arm_trajectory* is provided, the planner checks
    arm B's configurations against multiple sampled arm A positions
    (temporal collision checking).  This prevents arm B from colliding
    with arm A at intermediate positions along its trajectory.

    Args:
        env: Simulation environment.
        robot_id_b: Robot ID of the arm to plan for.
        start_q: Start joint configuration ``(7,)``.
        goal_q: Goal joint configuration ``(7,)``.
        q_other_arm: Fixed joint configuration of the other arm.
            Used when *q_other_arm_trajectory* is not provided.
        robot_id_a: Robot ID of the other arm.
        env_lock: Optional threading lock.
        max_iterations: Maximum RRT-Connect iterations.
        step_size: Step size for tree extension (radians).
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

    Returns:
        :class:`JointRRTResult` with success flag and path.
    """
    if max_iterations is None:
        max_iterations = constants.ARM_RRT_MAX_ITERATIONS
    if step_size is None:
        step_size = constants.ARM_RRT_STEP_SIZE
    if goal_bias is None:
        goal_bias = constants.ARM_RRT_GOAL_BIAS
    if max_temporal_samples is None:
        max_temporal_samples = constants.ARM_RRT_TEMPORAL_SAMPLES

    # ── Build list of arm A configs for collision checking ─────────
    #    When a trajectory is provided, sample representative configs
    #    for conservative temporal collision avoidance.
    if q_other_arm_trajectory is not None and len(q_other_arm_trajectory) > 1:
        n_traj = len(q_other_arm_trajectory)
        if n_traj <= max_temporal_samples:
            q_a_samples = [q_other_arm_trajectory[i] for i in range(n_traj)]
        else:
            indices = np.unique(
                np.linspace(0, n_traj - 1, max_temporal_samples, dtype=int)
            )
            q_a_samples = [q_other_arm_trajectory[i] for i in indices]
        logger.info(
            "Temporal collision checking: using %d sampled arm A "
            "configs from trajectory of length %d.",
            len(q_a_samples), n_traj,
        )
    else:
        q_a_samples = [np.asarray(q_other_arm)]

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
            direct_ok = all(
                collision_free_joint_segment(
                    env, start_q, goal_q,
                    q_a, qpos_idx_a, qpos_idx_b,
                    native_model, native_data, arm_b_geom_ids,
                    skip_endpoints=False,
                )
                for q_a in q_a_samples
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
                return any(
                    check_arm_config_collision(
                        env, q, qpos_idx_b,
                        q_a, qpos_idx_a,
                        native_model, native_data, arm_b_geom_ids,
                    )
                    for q_a in q_a_samples
                )

            def _seg_free(q1, q2):
                return all(
                    collision_free_joint_segment(
                        env, q1, q2,
                        q_a, qpos_idx_a, qpos_idx_b,
                        native_model, native_data, arm_b_geom_ids,
                        skip_endpoints=True,
                    )
                    for q_a in q_a_samples
                )

            def _sample():
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
                        rrt_vis.draw_tree_edge(
                            _ee_pos_fn(parent_q),
                            _ee_pos_fn(new_q),
                            is_tree_a=is_start,
                        )

                    def _on_collision(q):
                        rrt_vis.draw_collision_sample(_ee_pos_fn(q))
                else:
                    logger.warning(
                        "RRT visualisation requested but EE position "
                        "function unavailable (site not found for robot%d).",
                        robot_id_b,
                    )

                def _on_iteration(it, ta_size, tb_size):
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
