#!/usr/bin/env python3
"""
Comprehensive tests for assignment planning models (ObjectTrajectory,
RobotTrajectory, TrajectoryBundle).
"""

import pytest
import numpy as np

from bimanual_manipulation.planning.assignment.models import (
    ObjectTrajectory,
    RobotTrajectory,
    TrajectoryBundle,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ObjectTrajectory
# ═══════════════════════════════════════════════════════════════════════════════

class TestObjectTrajectory:
    @pytest.fixture
    def sample_obj_traj(self):
        n = 10
        return ObjectTrajectory(
            positions=np.random.randn(n, 3),
            quaternions=np.tile([0, 0, 0, 1], (n, 1)).astype(float),
            frame_ids=np.array(["world"] * n),
        )

    def test_n_waypoints(self, sample_obj_traj):
        assert sample_obj_traj.n_waypoints == 10

    def test_numpy_coercion(self):
        """Lists should be coerced to numpy arrays."""
        ot = ObjectTrajectory(
            positions=[[0, 0, 0], [1, 1, 1]],
            quaternions=[[0, 0, 0, 1], [0, 0, 0, 1]],
            frame_ids=["world", "world"],
        )
        assert isinstance(ot.positions, np.ndarray)
        assert isinstance(ot.quaternions, np.ndarray)
        assert isinstance(ot.frame_ids, np.ndarray)

    def test_single_waypoint(self):
        ot = ObjectTrajectory(
            positions=np.zeros((1, 3)),
            quaternions=np.array([[0, 0, 0, 1]]),
            frame_ids=np.array(["world"]),
        )
        assert ot.n_waypoints == 1


# ═══════════════════════════════════════════════════════════════════════════════
# RobotTrajectory
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobotTrajectory:
    @pytest.fixture
    def sample_robot_traj(self):
        n = 20
        return RobotTrajectory(
            eef_positions=np.random.randn(n, 3),
            eef_quaternions=np.tile([0, 0, 0, 1], (n, 1)).astype(float),
            eef_frame_ids=np.array(["world"] * n),
            object_positions=np.random.randn(n, 3),
            object_quaternions=np.tile([0, 0, 0, 1], (n, 1)).astype(float),
            object_frame_ids=np.array(["world"] * n),
            is_feasible=np.ones(n, dtype=bool),
            ik_success=np.ones(n, dtype=bool),
            collision_free=np.ones(n, dtype=bool),
            joint_configs=np.random.randn(n, 7),
        )

    def test_n_waypoints(self, sample_robot_traj):
        assert sample_robot_traj.n_waypoints == 20

    def test_feasibility_ratio_all_feasible(self, sample_robot_traj):
        assert sample_robot_traj.feasibility_ratio == 1.0

    def test_feasibility_ratio_half(self):
        n = 10
        feas = np.array([True, False, True, False, True, False, True, False, True, False])
        rt = RobotTrajectory(
            eef_positions=np.zeros((n, 3)),
            eef_quaternions=np.tile([0, 0, 0, 1], (n, 1)),
            eef_frame_ids=np.array(["w"] * n),
            object_positions=np.zeros((n, 3)),
            object_quaternions=np.tile([0, 0, 0, 1], (n, 1)),
            object_frame_ids=np.array(["w"] * n),
            is_feasible=feas,
            ik_success=feas,
            collision_free=feas,
            joint_configs=np.zeros((n, 7)),
        )
        assert abs(rt.feasibility_ratio - 0.5) < 1e-10

    def test_feasibility_ratio_none_feasible(self):
        n = 5
        rt = RobotTrajectory(
            eef_positions=np.zeros((n, 3)),
            eef_quaternions=np.tile([0, 0, 0, 1], (n, 1)),
            eef_frame_ids=np.array(["w"] * n),
            object_positions=np.zeros((n, 3)),
            object_quaternions=np.tile([0, 0, 0, 1], (n, 1)),
            object_frame_ids=np.array(["w"] * n),
            is_feasible=np.zeros(n, dtype=bool),
            ik_success=np.zeros(n, dtype=bool),
            collision_free=np.zeros(n, dtype=bool),
            joint_configs=np.zeros((n, 7)),
        )
        assert rt.feasibility_ratio == 0.0

    def test_numpy_coercion(self):
        """List inputs should be coerced to numpy."""
        rt = RobotTrajectory(
            eef_positions=[[0, 0, 0]],
            eef_quaternions=[[0, 0, 0, 1]],
            eef_frame_ids=["w"],
            object_positions=[[0, 0, 0]],
            object_quaternions=[[0, 0, 0, 1]],
            object_frame_ids=["w"],
            is_feasible=[True],
            ik_success=[True],
            collision_free=[True],
            joint_configs=[[0] * 7],
        )
        assert isinstance(rt.joint_configs, np.ndarray)
        assert isinstance(rt.is_feasible, np.ndarray)

    def test_source_path_default_none(self, sample_robot_traj):
        assert sample_robot_traj.source_path is None

    def test_source_path_set(self):
        n = 2
        rt = RobotTrajectory(
            eef_positions=np.zeros((n, 3)),
            eef_quaternions=np.tile([0, 0, 0, 1], (n, 1)),
            eef_frame_ids=np.array(["w"] * n),
            object_positions=np.zeros((n, 3)),
            object_quaternions=np.tile([0, 0, 0, 1], (n, 1)),
            object_frame_ids=np.array(["w"] * n),
            is_feasible=np.ones(n, dtype=bool),
            ik_success=np.ones(n, dtype=bool),
            collision_free=np.ones(n, dtype=bool),
            joint_configs=np.zeros((n, 7)),
            source_path="robot0_trajectory_0.npz",
        )
        assert rt.source_path == "robot0_trajectory_0.npz"


# ═══════════════════════════════════════════════════════════════════════════════
# TrajectoryBundle
# ═══════════════════════════════════════════════════════════════════════════════

def _make_robot_traj(n: int) -> RobotTrajectory:
    return RobotTrajectory(
        eef_positions=np.random.randn(n, 3),
        eef_quaternions=np.tile([0, 0, 0, 1], (n, 1)).astype(float),
        eef_frame_ids=np.array(["world"] * n),
        object_positions=np.random.randn(n, 3),
        object_quaternions=np.tile([0, 0, 0, 1], (n, 1)).astype(float),
        object_frame_ids=np.array(["world"] * n),
        is_feasible=np.ones(n, dtype=bool),
        ik_success=np.ones(n, dtype=bool),
        collision_free=np.ones(n, dtype=bool),
        joint_configs=np.random.randn(n, 7),
    )


class TestTrajectoryBundle:
    @pytest.fixture
    def sample_bundle(self):
        n = 15
        obj = ObjectTrajectory(
            positions=np.random.randn(n, 3),
            quaternions=np.tile([0, 0, 0, 1], (n, 1)).astype(float),
            frame_ids=np.array(["world"] * n),
        )
        return TrajectoryBundle(
            object_trajectory=obj,
            robot0_trajectories=[_make_robot_traj(n) for _ in range(3)],
            robot1_trajectories=[_make_robot_traj(n) for _ in range(2)],
        )

    def test_n_segments(self, sample_bundle):
        """n_segments == n_waypoints - 1."""
        assert sample_bundle.n_segments == 14

    def test_get_robot_trajectories_0(self, sample_bundle):
        trajs = sample_bundle.get_robot_trajectories(0)
        assert len(trajs) == 3

    def test_get_robot_trajectories_1(self, sample_bundle):
        trajs = sample_bundle.get_robot_trajectories(1)
        assert len(trajs) == 2

    def test_get_robot_trajectories_invalid(self, sample_bundle):
        with pytest.raises(ValueError, match="Invalid robot_id"):
            sample_bundle.get_robot_trajectories(2)

    def test_get_trajectory(self, sample_bundle):
        t = sample_bundle.get_trajectory(0, 0)
        assert isinstance(t, RobotTrajectory)

    def test_empty_robot_trajectories(self):
        n = 5
        obj = ObjectTrajectory(
            positions=np.zeros((n, 3)),
            quaternions=np.tile([0, 0, 0, 1], (n, 1)),
            frame_ids=np.array(["w"] * n),
        )
        bundle = TrajectoryBundle(
            object_trajectory=obj,
            robot0_trajectories=[],
            robot1_trajectories=[],
        )
        assert bundle.n_segments == 4
        assert len(bundle.get_robot_trajectories(0)) == 0
