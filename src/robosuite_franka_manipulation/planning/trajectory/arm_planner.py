"""
High-level collision-aware arm trajectory planner.

Orchestrates the joint-space RRT-Connect planner
(:func:`rrt.joint_planner.rrt_connect_joint_space`), two-phase
smoothing (shortcutting + per-segment Bézier), and densification to
produce a collision-free arm trajectory.

Falls back to linear interpolation if RRT fails.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.planning.rrt.joint import (
    rrt_connect_joint_space,
    shortcut_joint_path,
    bezier_smooth_joint_path,
    densify_joint_path,
)
from robosuite_franka_manipulation.planning.rrt.visualizer import RRTVisualizer

logger = logging.getLogger(__name__)


# ============================================================================
# High-Level API
# ============================================================================

def plan_arm_trajectory(
    env: DualArmEnvironment,
    robot_id_b: int,
    start_q: np.ndarray,
    goal_q: np.ndarray,
    q_other_arm: np.ndarray,
    robot_id_a: int,
    num_steps: int,
    env_lock=None,
    max_iterations: int = None,
    step_size: float = None,
    goal_bias: float = None,
    smoothing_iterations: int = None,
    rrt_vis: Optional[RRTVisualizer] = None,
) -> Tuple[np.ndarray, bool]:
    """Plan a collision-free joint trajectory for arm B — full pipeline.

    Runs joint-space RRT-Connect, followed by a two-phase smoothing
    pipeline (shortcutting + per-segment Bézier), and finally
    densification.  Falls back to linear interpolation if RRT fails.

    Args:
        env: Simulation environment.
        robot_id_b: Robot ID of the arm to plan for.
        start_q: Start joint configuration ``(7,)``.
        goal_q: Goal joint configuration ``(7,)``.
        q_other_arm: Fixed configuration of the other arm.
        robot_id_a: Robot ID of the other arm.
        num_steps: Number of output trajectory waypoints.
        env_lock: Optional threading lock.
        manipulated_object: Object held by arm A (for contact filtering).
        max_iterations: Max RRT iterations.
        step_size: RRT step size (radians).
        goal_bias: RRT goal bias.
        smoothing_iterations: Shortcut smoothing attempts.
        rrt_vis: Optional :class:`RRTVisualizer` for Meshcat debugging.

    Returns:
        Tuple of ``(trajectory, planning_success)`` where trajectory is
        ``(num_steps, 7)`` and planning_success is ``True`` if RRT found
        a collision-free path.
    """
    start_q = np.asarray(start_q, dtype=np.float64)
    goal_q = np.asarray(goal_q, dtype=np.float64)
    q_other_arm = np.asarray(q_other_arm, dtype=np.float64)

    result = rrt_connect_joint_space(
        env=env,
        robot_id_b=robot_id_b,
        start_q=start_q,
        goal_q=goal_q,
        q_other_arm=q_other_arm,
        robot_id_a=robot_id_a,
        env_lock=env_lock,
        max_iterations=max_iterations,
        step_size=step_size,
        goal_bias=goal_bias,
        rrt_vis=rrt_vis,
    )

    if result.success:
        logger.info(
            "Joint RRT succeeded (%d iters, %d raw waypoints).  "
            "Phase 1: shortcutting...",
            result.iterations, len(result.path),
        )
        # Phase 1: shortcut smoothing
        shortcut = shortcut_joint_path(
            env=env,
            path=result.path,
            q_other_arm=q_other_arm,
            robot_id_a=robot_id_a,
            robot_id_b=robot_id_b,
            env_lock=env_lock,
            max_iterations=smoothing_iterations,
        )
        logger.info(
            "Phase 1 shortcutting: %d -> %d waypoints.  "
            "Phase 2: Bézier smoothing...",
            len(result.path), len(shortcut),
        )
        # Phase 2: per-segment Bézier smoothing
        bezier = bezier_smooth_joint_path(
            env=env,
            path=shortcut,
            q_other_arm=q_other_arm,
            robot_id_a=robot_id_a,
            robot_id_b=robot_id_b,
            env_lock=env_lock,
        )
        # Final densification to target number of steps
        traj = densify_joint_path(bezier, num_steps)
        return traj, True

    logger.warning(
        "Joint RRT failed — falling back to linear interpolation."
    )
    # Fallback: linear interpolation (no collision guarantee)
    t = np.linspace(0.0, 1.0, num_steps).reshape(-1, 1)
    traj = start_q + t * (goal_q - start_q)
    return traj, False
