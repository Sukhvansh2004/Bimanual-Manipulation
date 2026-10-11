"""
High-level collision-aware object trajectory planner.

Orchestrates the RRT-Connect planner (:mod:`rrt.se3.planner`) and path
post-processing (:mod:`rrt.se3.smoothing`) to produce a collision-free
object trajectory through the workspace.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.transforms.stamped_pose import StampedPose

from bimanual_manipulation.planning.rrt.se3 import rrt_connect_se3
from bimanual_manipulation.planning.rrt.se3 import smooth_path, densify_path

from bimanual_manipulation.planning.rrt.visualizer import RRTVisualizer

logger = logging.getLogger(__name__)


# ============================================================================
# High-Level API
# ============================================================================

def plan_object_trajectory(
    env: DualArmEnvironment,
    object_name: str,
    start_pose: StampedPose,
    target_pose: StampedPose,
    tf_manager=None,
    num_steps: int = constants.OBJECT_TRAJ_NUM_STEPS,
    env_lock=None,
    max_iterations: int = None,
    step_size: float = None,
    goal_bias: float = None,
    pos_bounds: np.ndarray = None,
    smoothing_iterations: int = None,
    rrt_vis: Optional[RRTVisualizer] = None,
) -> Tuple[List[StampedPose], bool]:
    """Plan a collision-free object trajectory from start to target pose.

    Uses RRT-Connect in SE(3) with MuJoCo collision checking.
    Ignores collisions with robot arms/grippers.

    If the direct path is collision-free, the two-waypoint RRT result is
    simply densified, giving an interpolated trajectory (fast path).
    Otherwise the RRT path is smoothed (:func:`smooth_path`) and densified
    (:func:`densify_path`).

    Args:
        env: The simulation environment.
        object_name: Name of the object to plan for.
        start_pose: Start pose (StampedPose); converted to world frame
            when *tf_manager* is given.
        target_pose: Target pose (StampedPose); converted to world frame
            when *tf_manager* is given.
        tf_manager: Optional TransformManager for frame conversion.
        num_steps: Number of trajectory intervals; the output has
            ``num_steps + 1`` waypoints.
        env_lock: Optional threading lock.
        max_iterations: Max RRT iterations (default from constants).
        step_size: RRT step size (default from constants).
        goal_bias: RRT goal bias (default from constants).
        pos_bounds: Workspace bounds ``(2, 3)`` array. When ``None``,
            computed as the start/goal bounding box padded by
            :data:`constants.RRT_BOUNDS_MARGIN`, with the lower z clamped
            to at least :data:`constants.RRT_MIN_Z`.
        smoothing_iterations: Path smoothing iterations (default from constants).
        rrt_vis: Optional :class:`RRTVisualizer` for Meshcat debugging.

    Returns:
        Tuple of (trajectory as list of ``num_steps + 1`` StampedPose in
        the start pose's frame, planning_success bool). The success flag
        is always ``True`` because failure raises.

    Raises:
        RuntimeError: If RRT planning fails to find a collision-free path.
    """
    logger.debug(
        "Planning trajectory for object '%s' from %s to %s in %d steps.",
        object_name, start_pose, target_pose, num_steps,
    )

    # Normalise to world frame
    if tf_manager is not None:
        if start_pose.frame_id != constants.WORLD_FRAME:
            start_pose = tf_manager.transform_pose(start_pose, constants.WORLD_FRAME)
        if target_pose.frame_id != constants.WORLD_FRAME:
            target_pose = tf_manager.transform_pose(target_pose, constants.WORLD_FRAME)

    frame_id = start_pose.frame_id

    start_pos = np.asarray(start_pose.position, dtype=np.float64)
    start_quat = np.asarray(start_pose.quaternion, dtype=np.float64)
    goal_pos = np.asarray(target_pose.position, dtype=np.float64)
    goal_quat = np.asarray(target_pose.quaternion, dtype=np.float64)

    # Auto-compute position bounds from start/goal with margin
    if pos_bounds is None:
        all_pos = np.stack([start_pos, goal_pos])
        margin = np.array(constants.RRT_BOUNDS_MARGIN)
        auto_lo = all_pos.min(axis=0) - margin
        auto_hi = all_pos.max(axis=0) + margin
        auto_lo[2] = max(auto_lo[2], constants.RRT_MIN_Z)
        pos_bounds = np.array([auto_lo, auto_hi])

    logger.info("Planning object trajectory: %s -> %s", start_pos, goal_pos)
    logger.info("  Workspace bounds: %s to %s", pos_bounds[0], pos_bounds[1])

    # Run RRT-Connect
    result = rrt_connect_se3(
        env=env,
        object_name=object_name,
        start_pos=start_pos,
        start_quat=start_quat,
        goal_pos=goal_pos,
        goal_quat=goal_quat,
        env_lock=env_lock,
        max_iterations=max_iterations,
        step_size=step_size,
        goal_bias=goal_bias,
        pos_bounds=pos_bounds,
        rrt_vis=rrt_vis,
    )

    if result.success:
        logger.info(
            "RRT planning succeeded (%d iterations, raw path length: %d waypoints)",
            result.iterations, len(result.path),
        )

        smoothed = smooth_path(
            env, object_name, result.path,
            env_lock=env_lock,
            max_smoothing_iterations=smoothing_iterations,
            rrt_vis=rrt_vis,
        )

        dense = densify_path(smoothed, num_steps + 1)

        trajectory = [
            StampedPose(position=pos, quaternion=quat, frame_id=frame_id)
            for pos, quat in dense
        ]
        logger.debug(
            "Final trajectory has %d waypoints after smoothing/densification.",
            len(trajectory),
        )
        return trajectory, True
    else:
        raise RuntimeError(
            "RRT planning failed — no collision-free object path found "
            f"(tried {max_iterations} iterations)."
        )
