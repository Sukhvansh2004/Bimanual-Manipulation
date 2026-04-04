"""
SE(3) RRT-Connect planner for collision-free object path planning.

Plans a collision-free path for an object through the workspace using
RRT-Connect in SE(3), ignoring contacts with robot arms and grippers.

The core RRT-Connect loop is delegated to
:func:`rrt.core.rrt_connect_generic`; this module supplies
SE(3)-specific callbacks (distance, steer, sample, collision).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mujoco
import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.planning.rrt.core import rrt_connect_generic
from bimanual_manipulation.utils.mujoco_collision import (
    get_geom_ids_by_prefix,
    get_robot_geom_mask,
)
from bimanual_manipulation.utils.mujoco_utils import get_native_mj_object
from bimanual_manipulation.planning.rrt.se3.collision import (
    check_object_pose_collision,
)
from bimanual_manipulation.planning.rrt.se3.utilities import (
    se3_pose_distance,
    sample_random_pose,
    steer,
    collision_free_segment,
)
from bimanual_manipulation.planning.rrt.visualizer import RRTVisualizer

logger = logging.getLogger(__name__)


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class RRTNode:
    """A node in the RRT tree."""

    pos: np.ndarray
    quat: np.ndarray
    parent: int = -1


@dataclass
class RRTResult:
    """Result of the RRT-Connect planner."""

    success: bool
    path: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    iterations: int = 0
    tree_a_size: int = 0
    tree_b_size: int = 0


# ============================================================================
# RRT-Connect core
# ============================================================================

def rrt_connect_se3(
    env: DualArmEnvironment,
    object_name: str,
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    goal_pos: np.ndarray,
    goal_quat: np.ndarray,
    env_lock=None,
    max_iterations: int = None,
    step_size: float = None,
    goal_bias: float = None,
    pos_bounds: np.ndarray = None,
    rrt_vis: Optional["RRTVisualizer"] = None,
) -> RRTResult:
    """Plan a collision-free path for an object using RRT-Connect in SE(3).

    Ignores collisions with robot arms and grippers.

    Args:
        env: The simulation environment.
        object_name: Name of the object to plan for.
        start_pos: Start position ``[x, y, z]``.
        start_quat: Start quaternion ``[x, y, z, w]``.
        goal_pos: Goal position.
        goal_quat: Goal quaternion.
        env_lock: Optional threading lock.
        max_iterations: Maximum RRT iterations.
        step_size: Step size for tree extension.
        goal_bias: Probability of sampling the goal.
        pos_bounds: Workspace bounds ``(2, 3)`` array.
        rrt_vis: Optional :class:`RRTVisualizer` for Meshcat debugging.

    Returns:
        :class:`RRTResult` with success flag and path.
    """
    if max_iterations is None:
        max_iterations = constants.RRT_MAX_ITERATIONS
    if step_size is None:
        step_size = constants.RRT_STEP_SIZE
    if goal_bias is None:
        goal_bias = constants.RRT_GOAL_BIAS
    if pos_bounds is None:
        pos_bounds = np.array(constants.RRT_WORKSPACE_BOUNDS)

    rotation_weight = constants.RRT_ROTATION_WEIGHT

    ctx = env_lock if env_lock is not None else nullcontext()

    with ctx:
        model = env.sim.model
        data = env.sim.data

        object_geom_ids = get_geom_ids_by_prefix(model, object_name)
        robot_mask = get_robot_geom_mask(model)

        if len(object_geom_ids) == 0:
            logger.error("No geoms found for object '%s'", object_name)
            return RRTResult(success=False)

        try:
            obj_joint_id = model.joint(f"{object_name}_joint0").id
            obj_joint_addr = int(model.jnt_qposadr[obj_joint_id])
        except Exception:
            logger.error("Could not find joint for object '%s'", object_name)
            return RRTResult(success=False)

        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            # Fast path: try direct connection first
            if collision_free_segment(
                env, object_name,
                start_pos, start_quat, goal_pos, goal_quat,
                object_geom_ids, robot_mask, obj_joint_addr,
                skip_endpoints=True,
            ):
                logger.info("Direct path is collision-free, no RRT needed.")
                return RRTResult(
                    success=True,
                    path=[(start_pos.copy(), start_quat.copy()),
                          (goal_pos.copy(), goal_quat.copy())],
                    iterations=0, tree_a_size=1, tree_b_size=1,
                )

            logger.info(
                "Direct path has collisions. Running RRT-Connect "
                "(max_iter=%d)...", max_iterations,
            )

            # ── Build domain callbacks ────────────────────────────────
            def _distance(s1, s2):
                return se3_pose_distance(
                    s1[0], s1[1], s2[0], s2[1], rotation_weight,
                )

            def _steer_cb(from_s, to_s, step):
                return steer(from_s[0], from_s[1], to_s[0], to_s[1], step)

            def _sample():
                return sample_random_pose(
                    pos_bounds, start_quat, goal_quat,
                )

            def _collision(s):
                return check_object_pose_collision(
                    env, object_name, s[0], s[1],
                    object_geom_ids, robot_mask, obj_joint_addr,
                )

            def _seg_free(s1, s2):
                return collision_free_segment(
                    env, object_name,
                    s1[0], s1[1], s2[0], s2[1],
                    object_geom_ids, robot_mask, obj_joint_addr,
                    skip_endpoints=True,
                )

            def _copy(s):
                return (s[0].copy(), s[1].copy())

            # ── Visualisation hooks ───────────────────────────────────
            _on_extend = None
            _on_collision = None
            _on_iteration = None

            if rrt_vis is not None:
                rrt_vis.draw_start_goal(start_pos, goal_pos)
                rrt_vis.draw_bounds(pos_bounds)

                def _on_extend(parent_s, new_s, is_start):
                    rrt_vis.draw_tree_edge(
                        parent_s[0], new_s[0], is_tree_a=is_start,
                    )

                def _on_collision(s):
                    rrt_vis.draw_collision_sample(s[0])

                def _on_iteration(it, ta_size, tb_size):
                    rrt_vis.log_iteration(it, ta_size, tb_size)

            # ── Run generic RRT-Connect ───────────────────────────────
            success, path, iters, ta, tb = rrt_connect_generic(
                start_state=(start_pos.copy(), start_quat.copy()),
                goal_state=(goal_pos.copy(), goal_quat.copy()),
                distance_fn=_distance,
                steer_fn=_steer_cb,
                sample_fn=_sample,
                collision_fn=_collision,
                segment_free_fn=_seg_free,
                copy_fn=_copy,
                step_size=step_size,
                goal_bias=goal_bias,
                max_iterations=max_iterations,
                connect_max_steps=constants.RRT_CONNECT_MAX_STEPS,
                on_extend=_on_extend,
                on_collision=_on_collision,
                on_iteration=_on_iteration,
            )

            if success:
                logger.info(
                    "RRT-Connect found path in %d iterations "
                    "(tree sizes: %d, %d)", iters, ta, tb,
                )
                if rrt_vis is not None:
                    rrt_vis.draw_path(path)
            else:
                logger.warning(
                    "RRT-Connect failed after %d iterations.", max_iterations,
                )

            return RRTResult(
                success=success,
                path=path,
                iterations=iters,
                tree_a_size=ta,
                tree_b_size=tb,
            )

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
