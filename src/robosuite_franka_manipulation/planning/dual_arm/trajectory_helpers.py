"""Trajectory planning and conversion helpers for the dual-arm planner.

These are stateless functions extracted from the planner class to keep
the orchestration logic focused on execution.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from robosuite_franka_manipulation.controllers.arm import RobosuiteArmController
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose
from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    plan_quintic_joint_trajectory,
)

logger = logging.getLogger(__name__)


def plan_pose_to_joint_trajectory(
    arm: RobosuiteArmController,
    target_pose: StampedPose,
    num_steps: int,
    q_warm_start: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Solve IK for *target_pose* and return a quintic reference trajectory.

    Uses the arm's IK solver for the goal configuration.  Returns both
    the trajectory and the solved target joints (needed by the RRT-based
    planner as the goal configuration).

    Returns:
        ``(trajectory, target_joints)`` — either may be ``None`` on
        IK failure.
    """
    if target_pose.frame_id != arm.base_frame:
        target_pose = arm.tf_manager.transform_pose(target_pose, arm.base_frame)

    target_joints = arm._solve_ik_for_pose(
        np.asarray(target_pose.position),
        np.asarray(target_pose.quaternion),
        q_warm_start=q_warm_start,
    )
    if target_joints is None:
        logger.warning(
            "IK failed for robot%d_%s target pose",
            arm.robot_id, arm.arm_id,
        )
        return None, None

    traj = plan_quintic_joint_trajectory(
        arm.get_joint_positions(), target_joints, num_steps,
    )
    return traj, target_joints


def normalise_trajectory(
    traj: Union[List[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Ensure trajectory is a 2-D numpy array."""
    if isinstance(traj, list):
        traj = np.array(
            [np.asarray(w, dtype=np.float64) for w in traj],
        )
    else:
        traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    return traj


def eef_to_joint_trajectory(
    arm: RobosuiteArmController,
    eef_traj: List[StampedPose],
    q_warm_start: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Convert an EEF trajectory to joint space by sequential IK."""
    joint_configs = []
    q_ws = (
        q_warm_start if q_warm_start is not None
        else arm.get_joint_positions()
    )

    for i, pose in enumerate(eef_traj):
        if pose.frame_id != arm.base_frame:
            pose = arm.tf_manager.transform_pose(
                pose, arm.base_frame,
            )

        q = arm._solve_ik_for_pose(
            np.asarray(pose.position),
            np.asarray(pose.quaternion),
            q_warm_start=q_ws,
        )
        if q is None:
            logger.warning(
                "IK failed at EEF waypoint %d/%d", i, len(eef_traj),
            )
            joint_configs.append(q_ws.copy())
        else:
            joint_configs.append(q)
            q_ws = q
    return np.array(joint_configs)
