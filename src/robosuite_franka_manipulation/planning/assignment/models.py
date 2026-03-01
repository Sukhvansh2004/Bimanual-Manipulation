"""Trajectory data models for the assignment planner.

Replaces raw ``Dict[str, Any]`` trajectory containers with typed
dataclasses that provide attribute access, validation, and convenience
properties.

Classes
-------
ObjectTrajectory
    Object SE(3) trajectory through the workspace.
RobotTrajectory
    Per-grasp robot trajectory with feasibility annotations.
TrajectoryBundle
    Complete set of trajectories needed for assignment planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ObjectTrajectory:
    """Object SE(3) trajectory through the workspace.

    Attributes:
        positions: ``(N, 3)`` array of 3-D positions.
        quaternions: ``(N, 4)`` array of quaternions ``[x, y, z, w]``.
        frame_ids: ``(N,)`` string array of coordinate frame IDs.
    """

    positions: np.ndarray
    quaternions: np.ndarray
    frame_ids: np.ndarray

    @property
    def n_waypoints(self) -> int:
        """Number of waypoints in the trajectory."""
        return len(self.positions)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions)
        self.quaternions = np.asarray(self.quaternions)
        self.frame_ids = np.asarray(self.frame_ids)


@dataclass
class RobotTrajectory:
    """Per-grasp robot trajectory with feasibility annotations.

    Each waypoint stores end-effector pose, object pose,
    IK feasibility flags, and the resulting joint configuration.

    Attributes:
        eef_positions: ``(N, 3)`` end-effector positions.
        eef_quaternions: ``(N, 4)`` end-effector quaternions.
        eef_frame_ids: ``(N,)`` EEF frame IDs.
        object_positions: ``(N, 3)`` object positions at each waypoint.
        object_quaternions: ``(N, 4)`` object quaternions.
        object_frame_ids: ``(N,)`` object frame IDs.
        is_feasible: ``(N,)`` boolean — overall feasibility per waypoint.
        ik_success: ``(N,)`` boolean — IK solve succeeded.
        collision_free: ``(N,)`` boolean — configuration is collision-free.
        joint_configs: ``(N, DOF)`` joint configurations.
    """

    eef_positions: np.ndarray
    eef_quaternions: np.ndarray
    eef_frame_ids: np.ndarray
    object_positions: np.ndarray
    object_quaternions: np.ndarray
    object_frame_ids: np.ndarray
    is_feasible: np.ndarray
    ik_success: np.ndarray
    collision_free: np.ndarray
    joint_configs: np.ndarray

    @property
    def n_waypoints(self) -> int:
        """Number of waypoints in the trajectory."""
        return len(self.is_feasible)

    @property
    def feasibility_ratio(self) -> float:
        """Fraction of waypoints that are feasible."""
        if self.n_waypoints == 0:
            return 0.0
        return float(np.mean(self.is_feasible.astype(float)))

    def __post_init__(self) -> None:
        self.eef_positions = np.asarray(self.eef_positions)
        self.eef_quaternions = np.asarray(self.eef_quaternions)
        self.eef_frame_ids = np.asarray(self.eef_frame_ids)
        self.object_positions = np.asarray(self.object_positions)
        self.object_quaternions = np.asarray(self.object_quaternions)
        self.object_frame_ids = np.asarray(self.object_frame_ids)
        self.is_feasible = np.asarray(self.is_feasible)
        self.ik_success = np.asarray(self.ik_success)
        self.collision_free = np.asarray(self.collision_free)
        self.joint_configs = np.asarray(self.joint_configs)


@dataclass
class TrajectoryBundle:
    """Complete set of trajectories needed for assignment planning.

    Bundles the object trajectory with per-robot grasp trajectories
    into a single typed container.

    Attributes:
        object_trajectory: Object SE(3) trajectory.
        robot0_trajectories: List of :class:`RobotTrajectory` for robot 0.
        robot1_trajectories: List of :class:`RobotTrajectory` for robot 1.
    """

    object_trajectory: ObjectTrajectory
    robot0_trajectories: List[RobotTrajectory]
    robot1_trajectories: List[RobotTrajectory]

    @property
    def n_segments(self) -> int:
        """Number of trajectory segments (waypoints - 1)."""
        return self.object_trajectory.n_waypoints - 1

    def get_robot_trajectories(self, robot_id: int) -> List[RobotTrajectory]:
        """Return the trajectory list for the given robot.

        Args:
            robot_id: 0 or 1.

        Raises:
            ValueError: If *robot_id* is not 0 or 1.
        """
        if robot_id == 0:
            return self.robot0_trajectories
        if robot_id == 1:
            return self.robot1_trajectories
        raise ValueError(f"Invalid robot_id: {robot_id} (must be 0 or 1)")

    def get_trajectory(self, robot_id: int, traj_idx: int) -> RobotTrajectory:
        """Return a specific trajectory by robot ID and index.

        Args:
            robot_id: 0 or 1.
            traj_idx: Index into the robot's trajectory list.
        """
        return self.get_robot_trajectories(robot_id)[traj_idx]
