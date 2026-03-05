"""
Comprehensive test suite for the segment assignment planner.

Covers:
- AssignmentResult dataclass
- compute_segment_cost
- compute_cost_matrix  (with and without jacobian_fns)
- _compute_handover_primitive_cost  (per-robot Jacobian dispatch)
- compute_switching_cost
- _solve_optimal_assignment / compute_optimal_robot_assignment
- save / load round-trip
- load_all_trajectories
- make_ik_solver_fn / make_jacobian_fns factory closures
- Edge cases: infeasible trajectories, single-segment, no handover, forced handover
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.planning.assignment import (
    AssignmentResult,
    ObjectTrajectory,
    RobotTrajectory,
    TrajectoryBundle,
    compute_cost_matrix,
    compute_optimal_robot_assignment,
    compute_segment_cost,
    compute_switching_cost,
    load_all_trajectories,
    load_assignment_result,
    make_collision_check_fn,
    make_ik_solver_fn,
    make_jacobian_fns,
    save_assignment_result,
)
from robosuite_franka_manipulation.planning.assignment.cost import (
    _compute_handover_primitive_cost,
)
from robosuite_franka_manipulation.transforms import StampedPose


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — synthetic trajectory builders
# ═══════════════════════════════════════════════════════════════════════════

def _make_trajectory(
    n_waypoints: int = 6,
    n_joints: int = 7,
    feasible: bool = True,
    seed: int = 0,
    source_path: str | None = None,
) -> RobotTrajectory:
    """Build a minimal RobotTrajectory suitable for the assignment planner."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1, 1, (n_waypoints, n_joints))
    positions = rng.uniform(-0.5, 0.5, (n_waypoints, 3))
    quats = np.tile([0, 0, 0, 1], (n_waypoints, 1)).astype(float)
    return RobotTrajectory(
        is_feasible=np.ones(n_waypoints, dtype=bool) if feasible else np.zeros(n_waypoints, dtype=bool),
        joint_configs=q,
        eef_positions=positions,
        eef_quaternions=quats,
        eef_frame_ids=np.array(["world"] * n_waypoints),
        object_positions=positions.copy(),
        object_quaternions=quats.copy(),
        object_frame_ids=np.array(["world"] * n_waypoints),
        ik_success=np.ones(n_waypoints, dtype=bool),
        collision_free=np.ones(n_waypoints, dtype=bool),
        source_path=source_path,
    )


def _make_object_trajectory(n_waypoints: int = 6, seed: int = 99) -> ObjectTrajectory:
    """Build a minimal object trajectory."""
    rng = np.random.default_rng(seed)
    return ObjectTrajectory(
        positions=rng.uniform(-0.3, 0.3, (n_waypoints, 3)),
        quaternions=np.tile([0, 0, 0, 1], (n_waypoints, 1)).astype(float),
        frame_ids=np.array(["world"] * n_waypoints),
    )


def _make_trajectories(
    n_waypoints: int = 6,
    n_r0: int = 2,
    n_r1: int = 2,
    seed: int = 0,
) -> TrajectoryBundle:
    """Build a full TrajectoryBundle for compute_optimal_robot_assignment."""
    return TrajectoryBundle(
        object_trajectory=_make_object_trajectory(n_waypoints, seed=seed + 100),
        robot0_trajectories=[
            _make_trajectory(
                n_waypoints, seed=seed + i,
                source_path=f"robot0_trajectory_{i}.npz",
            )
            for i in range(n_r0)
        ],
        robot1_trajectories=[
            _make_trajectory(
                n_waypoints, seed=seed + 50 + i,
                source_path=f"robot1_trajectory_{i}.npz",
            )
            for i in range(n_r1)
        ],
    )


def _dummy_jacobian_fn(q: np.ndarray) -> np.ndarray:
    """Deterministic fake (6×7) Jacobian for testing."""
    rng = np.random.default_rng(int(np.abs(q).sum() * 1000) % (2**31))
    return rng.standard_normal((6, 7))


# ═══════════════════════════════════════════════════════════════════════════
# Tests — AssignmentResult
# ═══════════════════════════════════════════════════════════════════════════

class TestAssignmentResult:
    """Test the dataclass and its I/O round-trip."""

    def _sample_result(self) -> AssignmentResult:
        return AssignmentResult(
            sequence=[0, 0, 1, 1, 0],
            trajectory_paths=[
                "robot0_trajectory_0.npz",
                "robot0_trajectory_0.npz",
                "robot1_trajectory_1.npz",
                "robot1_trajectory_1.npz",
                "robot0_trajectory_0.npz",
            ],
            total_cost=42.5,
            task_costs=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            switching_costs=np.array([0.0, 0.0, 10.0, 0.0, 10.0]),
            handover_points=[2, 4],
            cost_matrix=np.random.rand(5, 2),
            trajectory_matrix=np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]),
        )

    def test_fields_accessible(self):
        r = self._sample_result()
        assert len(r.sequence) == 5
        assert r.total_cost == 42.5
        assert r.handover_points == [2, 4]

    def test_save_load_roundtrip(self, tmp_path):
        original = self._sample_result()
        fpath = str(tmp_path / "result.npz")

        save_assignment_result(original, fpath)
        assert os.path.exists(fpath)

        loaded = load_assignment_result(fpath)
        assert loaded.sequence == original.sequence
        assert loaded.trajectory_paths == original.trajectory_paths
        assert loaded.total_cost == pytest.approx(original.total_cost)
        np.testing.assert_allclose(loaded.task_costs, original.task_costs)
        np.testing.assert_allclose(loaded.switching_costs, original.switching_costs)
        assert loaded.handover_points == original.handover_points
        np.testing.assert_allclose(loaded.cost_matrix, original.cost_matrix)
        np.testing.assert_array_equal(loaded.trajectory_matrix, original.trajectory_matrix)

    def test_save_load_empty_handover_points(self, tmp_path):
        r = self._sample_result()
        r.handover_points = []
        fpath = str(tmp_path / "no_handover.npz")
        save_assignment_result(r, fpath)
        loaded = load_assignment_result(fpath)
        assert loaded.handover_points == []


# ═══════════════════════════════════════════════════════════════════════════
# Tests — compute_segment_cost
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeSegmentCost:
    """Unit tests for compute_segment_cost."""

    def test_feasible_segment_returns_finite_cost(self):
        traj = _make_trajectory(n_waypoints=5)
        cost, ok = compute_segment_cost(traj, 0, 1)
        assert ok is True
        assert np.isfinite(cost)
        assert cost >= 0

    def test_infeasible_segment_returns_inf(self):
        traj = _make_trajectory(n_waypoints=5, feasible=False)
        cost, ok = compute_segment_cost(traj, 0, 1)
        assert ok is False
        assert np.isinf(cost)

    def test_partially_infeasible_segment(self):
        traj = _make_trajectory(n_waypoints=5)
        traj.is_feasible[1] = False  # waypoint inside segment [0,2]
        cost, ok = compute_segment_cost(traj, 0, 2)
        assert ok is False

    def test_out_of_range_indices(self):
        traj = _make_trajectory(n_waypoints=5)
        cost, ok = compute_segment_cost(traj, -1, 1)
        assert ok is False
        cost2, ok2 = compute_segment_cost(traj, 0, 5)
        assert ok2 is False
        cost3, ok3 = compute_segment_cost(traj, 3, 2)
        assert ok3 is False

    def test_cost_increases_with_alpha(self):
        """Smoothness cost (alpha * ||dq||^2) should scale with alpha."""
        traj = _make_trajectory(n_waypoints=4, seed=7)
        c1, _ = compute_segment_cost(traj, 0, 1, alpha=1.0, beta=0.0)
        c2, _ = compute_segment_cost(traj, 0, 1, alpha=10.0, beta=0.0)
        assert c2 == pytest.approx(10.0 * c1, rel=1e-6)

    def test_cost_increases_with_beta(self):
        """Manipulability cost should scale with beta."""
        traj = _make_trajectory(n_waypoints=4, seed=8)
        c1, _ = compute_segment_cost(traj, 0, 1, alpha=0.0, beta=1.0)
        c2, _ = compute_segment_cost(traj, 0, 1, alpha=0.0, beta=5.0)
        assert c2 == pytest.approx(5.0 * c1, rel=1e-6)

    def test_with_jacobian_fn(self):
        """Passing a jacobian_fn should change the manipulability cost."""
        traj = _make_trajectory(n_waypoints=4, seed=9)
        c_simple, _ = compute_segment_cost(traj, 0, 1, alpha=0.0, beta=1.0)
        c_jac, _ = compute_segment_cost(
            traj, 0, 1, alpha=0.0, beta=1.0, jacobian_fn=_dummy_jacobian_fn,
        )
        # They should differ because different manipulability measures are used
        assert c_simple != pytest.approx(c_jac, rel=1e-3)

    def test_single_waypoint_segment_zero_smoothness(self):
        """A segment of length 1 should have zero smoothness cost."""
        traj = _make_trajectory(n_waypoints=3)
        # segment [0,1] has start_idx=0 end_idx=1, includes indices 0 and 1 → 2 waypoints
        # We need a segment with exactly 1 waypoint → impossible with current API
        # since start_idx < end_idx. Instead verify that identical configs → 0 smoothness.
        traj.joint_configs[0] = traj.joint_configs[1]
        cost, ok = compute_segment_cost(traj, 0, 1, alpha=1.0, beta=0.0)
        assert ok is True
        assert cost == pytest.approx(0.0, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# Tests — compute_cost_matrix
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeCostMatrix:
    """Unit tests for compute_cost_matrix."""

    def test_shape_and_info(self):
        r0 = [_make_trajectory(6, seed=0), _make_trajectory(6, seed=1)]
        r1 = [_make_trajectory(6, seed=2)]
        n_segments = 5
        cm, info = compute_cost_matrix(r0, r1, n_segments)

        assert cm.shape == (n_segments, 3)  # 2 R0 + 1 R1
        assert len(info) == 3
        assert info[0] == (0, 0)
        assert info[1] == (0, 1)
        assert info[2] == (1, 0)

    def test_all_feasible_finite(self):
        r0 = [_make_trajectory(6, seed=10)]
        r1 = [_make_trajectory(6, seed=20)]
        cm, _ = compute_cost_matrix(r0, r1, 5)
        assert np.all(np.isfinite(cm))

    def test_infeasible_trajectory_gives_inf(self):
        r0 = [_make_trajectory(6, feasible=False, seed=30)]
        r1 = [_make_trajectory(6, seed=40)]
        cm, _ = compute_cost_matrix(r0, r1, 5)
        # All R0 entries must be inf; R1 entries should be finite
        assert np.all(np.isinf(cm[:, 0]))
        assert np.all(np.isfinite(cm[:, 1]))

    def test_without_jacobian_fns(self):
        r0 = [_make_trajectory(6, seed=50)]
        r1 = [_make_trajectory(6, seed=60)]
        cm_no_jac, _ = compute_cost_matrix(r0, r1, 5, jacobian_fns=None)
        assert np.all(np.isfinite(cm_no_jac))

    def test_with_jacobian_fns_per_robot(self):
        """jacobian_fns should dispatch per-robot correctly."""
        r0 = [_make_trajectory(6, seed=50)]
        r1 = [_make_trajectory(6, seed=60)]

        call_log: Dict[int, int] = {0: 0, 1: 0}

        def _jac_factory(rid: int):
            def _fn(q):
                call_log[rid] += 1
                return _dummy_jacobian_fn(q)
            return _fn

        jac_fns = {0: _jac_factory(0), 1: _jac_factory(1)}
        cm, _ = compute_cost_matrix(r0, r1, 5, jacobian_fns=jac_fns)

        assert np.all(np.isfinite(cm))
        assert call_log[0] > 0, "Robot-0 Jacobian was never called"
        assert call_log[1] > 0, "Robot-1 Jacobian was never called"

    def test_jacobian_fns_changes_costs(self):
        """Cost values should differ when Jacobian functions are provided."""
        r0 = [_make_trajectory(6, seed=70)]
        r1 = [_make_trajectory(6, seed=80)]
        cm_no, _ = compute_cost_matrix(r0, r1, 5, jacobian_fns=None)
        jac_fns = {0: _dummy_jacobian_fn, 1: _dummy_jacobian_fn}
        cm_jac, _ = compute_cost_matrix(r0, r1, 5, jacobian_fns=jac_fns)
        assert not np.allclose(cm_no, cm_jac), "Costs should differ with Jacobian"


# ═══════════════════════════════════════════════════════════════════════════
# Tests — compute_switching_cost
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeSwitchingCost:
    """Unit tests for compute_switching_cost."""

    def _make_args(self):
        trajs = _make_trajectories(n_waypoints=6, n_r0=1, n_r1=1)
        _, traj_info = compute_cost_matrix(
            trajs.robot0_trajectories,
            trajs.robot1_trajectories,
            5,
        )
        return trajs, traj_info

    def test_same_robot_returns_zero(self):
        trajs, traj_info = self._make_args()
        cost = compute_switching_cost(
            prev_robot=0, curr_robot=0, segment_idx=1,
            trajectories=trajs, prev_state_idx=0, curr_state_idx=0,
            trajectory_info=traj_info,
        )
        assert cost == 0.0

    def test_different_robot_no_ik_returns_fallback(self):
        trajs, traj_info = self._make_args()
        cost = compute_switching_cost(
            prev_robot=0, curr_robot=1, segment_idx=1,
            trajectories=trajs, prev_state_idx=0, curr_state_idx=1,
            trajectory_info=traj_info, ik_solver_fn=None,
        )
        assert cost == constants.ASSIGNMENT_FALLBACK_SWITCH_COST

    def test_transfer_feasibility_blocks(self):
        trajs, traj_info = self._make_args()
        tf = {(0, 0, 1, 0): False}
        cost = compute_switching_cost(
            prev_robot=0, curr_robot=1, segment_idx=1,
            trajectories=trajs, prev_state_idx=0, curr_state_idx=1,
            trajectory_info=traj_info,
            transfer_feasibility=tf,
        )
        assert np.isinf(cost)

    def test_transfer_feasibility_allows(self):
        trajs, traj_info = self._make_args()
        tf = {(0, 0, 1, 0): True}
        cost = compute_switching_cost(
            prev_robot=0, curr_robot=1, segment_idx=1,
            trajectories=trajs, prev_state_idx=0, curr_state_idx=1,
            trajectory_info=traj_info,
            transfer_feasibility=tf, ik_solver_fn=None,
        )
        # Not blocked → falls through to fallback since ik_solver_fn is None
        assert cost == constants.ASSIGNMENT_FALLBACK_SWITCH_COST


# ═══════════════════════════════════════════════════════════════════════════
# Tests — _compute_handover_primitive_cost
# ═══════════════════════════════════════════════════════════════════════════

class TestHandoverPrimitiveCost:
    """Tests for the internal _compute_handover_primitive_cost function."""

    @staticmethod
    def _dummy_ik(pose: StampedPose, robot_id: int, q_warm_start=None) -> Optional[np.ndarray]:
        """Return a deterministic 7-DOF config based on the pose."""
        return np.array(pose.position.tolist() + [0.1, 0.2, 0.3, 0.4])

    def test_empty_trajectory_returns_inf(self):
        cost, ok = _compute_handover_primitive_cost(
            {}, 1.0, 1.0, 1e-6, self._dummy_ik,
        )
        assert np.isinf(cost)
        assert ok is False

    def test_feasible_handover_returns_finite(self):
        poses = [
            StampedPose(position=[0.1 * i, 0, 0], quaternion=[0, 0, 0, 1], frame_id="object")
            for i in range(3)
        ]
        h_traj = {
            "transfer_hold": poses[:2],
            "receiver_pre": poses[1:],
        }
        cost, ok = _compute_handover_primitive_cost(
            h_traj, 1.0, 1.0, 1e-6, self._dummy_ik,
        )
        assert ok is True
        assert np.isfinite(cost)
        assert cost > 0

    def test_ik_failure_returns_inf(self):
        def _failing_ik(pose, robot_id, q_warm_start=None):
            return None

        poses = [StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="w")]
        h_traj = {"transfer_hold": poses}
        cost, ok = _compute_handover_primitive_cost(
            h_traj, 1.0, 1.0, 1e-6, _failing_ik,
        )
        assert np.isinf(cost)
        assert ok is False

    def test_partial_ik_failure_returns_finite_with_penalty(self):
        """When some IK calls fail but at least one succeeds, cost should be
        finite and include a per-failure penalty."""
        from robosuite_franka_manipulation import constants as C

        call_count = [0]

        def _partial_ik(pose, robot_id, q_warm_start=None):
            """Fail on even-numbered calls, succeed on odd."""
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return None
            return np.array(pose.position.tolist() + [0.1, 0.2, 0.3, 0.4])

        poses = [
            StampedPose(position=[0.1 * i, 0, 0], quaternion=[0, 0, 0, 1], frame_id="o")
            for i in range(5)
        ]
        h_traj = {
            "transfer_hold": poses[:2],
            "receiver_approach": poses[2:],
        }
        cost, ok = _compute_handover_primitive_cost(
            h_traj, 1.0, 1.0, 1e-6, _partial_ik,
        )
        assert ok is True, "Should succeed when some IK calls succeed"
        assert np.isfinite(cost), "Cost should be finite"
        # Cost should include penalty for partial failures
        assert cost >= C.HANDOVER_IK_PARTIAL_PENALTY, (
            f"Cost {cost} should include at least one partial penalty "
            f"({C.HANDOVER_IK_PARTIAL_PENALTY})"
        )

    def test_per_robot_jacobian_dispatch(self):
        """Each arm's manipulability should use its own Jacobian function."""
        call_log: Dict[int, int] = {0: 0, 1: 0}

        def _ik(pose, robot_id, q_warm_start=None):
            return np.ones(7) * (robot_id + 1)

        def _jac_factory(rid):
            def _fn(q):
                call_log[rid] += 1
                return _dummy_jacobian_fn(q)
            return _fn

        jac_fns = {0: _jac_factory(0), 1: _jac_factory(1)}

        poses = [
            StampedPose(position=[0.1 * i, 0, 0], quaternion=[0, 0, 0, 1], frame_id="o")
            for i in range(3)
        ]
        h_traj = {
            "transfer_hold": poses,
            "receiver_approach": poses,
        }
        cost, ok = _compute_handover_primitive_cost(
            h_traj, 1.0, 1.0, 1e-6, _ik, jacobian_fns=jac_fns,
        )
        assert ok is True
        assert call_log[0] > 0, "Robot-0 Jacobian never called for transfer arm"
        assert call_log[1] > 0, "Robot-1 Jacobian never called for receiver arm"

    def test_collision_check_adds_penalty_for_colliding_handover(self):
        """Collisions add per-step penalty instead of rejecting the handover."""
        from robosuite_franka_manipulation import constants as C

        def _ik(pose, robot_id, q_warm_start=None):
            return np.ones(7) * 0.5

        def _always_colliding(q0, q1):
            return False  # collision detected

        poses = [
            StampedPose(position=[0.1 * i, 0, 0], quaternion=[0, 0, 0, 1], frame_id="o")
            for i in range(3)
        ]
        h_traj = {
            "transfer_hold": poses[:1],
            "receiver_approach": poses,
        }

        # Compute cost with collisions
        cost_col, ok_col = _compute_handover_primitive_cost(
            h_traj, 1.0, 1.0, 1e-6, _ik,
            collision_check_fn=_always_colliding,
        )
        assert ok_col is True
        assert np.isfinite(cost_col)

        # Compute cost without collisions for comparison
        cost_free, ok_free = _compute_handover_primitive_cost(
            h_traj, 1.0, 1.0, 1e-6, _ik,
            collision_check_fn=lambda q0, q1: True,
        )
        assert ok_free is True

        # Colliding cost must exceed collision-free cost by at least one
        # collision step penalty.
        assert cost_col > cost_free + C.HANDOVER_COLLISION_STEP_PENALTY, (
            f"Colliding cost ({cost_col:.2f}) should exceed free cost "
            f"({cost_free:.2f}) by at least {C.HANDOVER_COLLISION_STEP_PENALTY}"
        )

    def test_collision_check_passes_when_free(self):
        """A collision_check_fn returning True should not affect cost."""
        def _ik(pose, robot_id, q_warm_start=None):
            return np.ones(7) * 0.5

        def _always_free(q0, q1):
            return True  # no collision

        poses = [
            StampedPose(position=[0.1 * i, 0, 0], quaternion=[0, 0, 0, 1], frame_id="o")
            for i in range(3)
        ]
        h_traj = {
            "transfer_hold": poses[:1],
            "receiver_approach": poses,
        }
        cost, ok = _compute_handover_primitive_cost(
            h_traj, 1.0, 1.0, 1e-6, _ik,
            collision_check_fn=_always_free,
        )
        assert ok is True
        assert np.isfinite(cost)

    def test_robot_id_passed_correctly(self):
        """Verify transfer_robot_id and receiver_robot_id are forwarded to IK."""
        ik_calls: List[int] = []

        def _ik(pose, robot_id, q_warm_start=None):
            ik_calls.append(robot_id)
            return np.ones(7) * robot_id

        poses = [
            StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="o")
        ]
        h_traj = {
            "transfer_hold": poses,
            "receiver_pre": poses,
        }
        _compute_handover_primitive_cost(
            h_traj, 1.0, 1.0, 1e-6, _ik,
            transfer_robot_id=1,
            receiver_robot_id=0,
        )
        assert 1 in ik_calls, "transfer_robot_id=1 not passed to IK"
        assert 0 in ik_calls, "receiver_robot_id=0 not passed to IK"


# ═══════════════════════════════════════════════════════════════════════════
# Tests — compute_optimal_robot_assignment  (end-to-end unit tests)
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeOptimalRobotAssignment:
    """End-to-end tests for the public assignment API."""

    def test_basic_no_ik(self):
        """Without IK/Jacobian the planner should still produce a valid result."""
        trajs = _make_trajectories(n_waypoints=6, n_r0=2, n_r1=2)
        result = compute_optimal_robot_assignment(trajs)

        n_segments = 5
        assert len(result.sequence) == n_segments
        assert len(result.trajectory_paths) == n_segments
        assert len(result.task_costs) == n_segments
        assert len(result.switching_costs) == n_segments
        assert np.isfinite(result.total_cost)
        assert all(r in (0, 1) for r in result.sequence)

    def test_all_infeasible_raises(self):
        """If every trajectory is infeasible, a ValueError must be raised."""
        trajs = _make_trajectories(n_waypoints=4, n_r0=1, n_r1=1, seed=5)
        for t in trajs.robot0_trajectories + trajs.robot1_trajectories:
            t.is_feasible[:] = False
        with pytest.raises(ValueError, match="No feasible assignment"):
            compute_optimal_robot_assignment(trajs)

    def test_initial_robot_respected(self):
        trajs = _make_trajectories(n_waypoints=6, n_r0=1, n_r1=1)
        r0 = compute_optimal_robot_assignment(trajs, initial_robot=0)
        r1 = compute_optimal_robot_assignment(trajs, initial_robot=1)
        assert r0.sequence[0] == 0
        assert r1.sequence[0] == 1

    def test_single_robot_only_feasible(self):
        """If only one robot has feasible trajectories, the entire sequence should use it."""
        trajs = _make_trajectories(n_waypoints=6, n_r0=1, n_r1=1, seed=11)
        for t in trajs.robot1_trajectories:
            t.is_feasible[:] = False
        result = compute_optimal_robot_assignment(trajs)
        assert all(r == 0 for r in result.sequence)
        assert result.handover_points == []

    def test_handover_points_correct(self):
        """Handover points should mark every switch between robots."""
        trajs = _make_trajectories(n_waypoints=6, n_r0=1, n_r1=1)
        result = compute_optimal_robot_assignment(trajs)
        for hp in result.handover_points:
            assert result.sequence[hp] != result.sequence[hp - 1]

    def test_cost_matrix_shape(self):
        trajs = _make_trajectories(n_waypoints=6, n_r0=2, n_r1=3)
        result = compute_optimal_robot_assignment(trajs)
        assert result.cost_matrix.shape == (5, 2)
        assert result.trajectory_matrix.shape == (5, 2)

    def test_with_jacobian_fns(self):
        """Passing jacobian_fns should produce potentially different costs."""
        trajs = _make_trajectories(n_waypoints=6, n_r0=1, n_r1=1, seed=22)
        r_nojac = compute_optimal_robot_assignment(trajs, jacobian_fns=None)
        jac_fns = {0: _dummy_jacobian_fn, 1: _dummy_jacobian_fn}
        r_jac = compute_optimal_robot_assignment(trajs, jacobian_fns=jac_fns)
        # Costs should differ (different manipulability measure)
        assert r_nojac.total_cost != pytest.approx(r_jac.total_cost, rel=1e-6)

    def test_single_segment(self):
        """Edge case: only 2 waypoints → 1 segment."""
        trajs = _make_trajectories(n_waypoints=2, n_r0=1, n_r1=1, seed=33)
        result = compute_optimal_robot_assignment(trajs)
        assert len(result.sequence) == 1
        assert result.handover_points == []

    def test_transfer_feasibility_blocks_handover(self):
        """Explicit transfer_feasibility=False should prevent that pair switching."""
        trajs = _make_trajectories(n_waypoints=4, n_r0=1, n_r1=1, seed=44)
        # Block all transfers in both directions
        tf = {
            (0, 0, 1, 0): False,
            (1, 0, 0, 0): False,
        }
        result = compute_optimal_robot_assignment(trajs, transfer_feasibility=tf)
        # With blocked transfers, there should be no handovers
        assert len(result.handover_points) == 0

    def test_switching_costs_zero_no_handover(self):
        """When no handover occurs, all switching costs should be 0."""
        trajs = _make_trajectories(n_waypoints=6, n_r0=1, n_r1=1, seed=55)
        for t in trajs.robot1_trajectories:
            t.is_feasible[:] = False
        result = compute_optimal_robot_assignment(trajs)
        # Only robot0 is feasible → no handover
        assert all(sc == 0.0 for sc in result.switching_costs)

    def test_total_cost_matches_sum(self):
        """total_cost should equal J*(last segment) which includes all task + switch costs."""
        trajs = _make_trajectories(n_waypoints=6, n_r0=1, n_r1=1, seed=66)
        # Force single robot to avoid stochastic handover costs
        for t in trajs.robot1_trajectories:
            t.is_feasible[:] = False
        result = compute_optimal_robot_assignment(trajs)
        expected = float(np.sum(result.task_costs) + np.sum(result.switching_costs))
        assert result.total_cost == pytest.approx(expected, rel=1e-10)

    def test_beam_width_produces_valid_result(self):
        """With beam_width set, the solver should still return a valid assignment."""
        trajs = _make_trajectories(n_waypoints=6, n_r0=3, n_r1=3, seed=77)
        result = compute_optimal_robot_assignment(trajs, beam_width=4)
        n_segments = 5
        assert len(result.sequence) == n_segments
        assert all(r in (0, 1) for r in result.sequence)
        assert np.isfinite(result.total_cost)

    def test_beam_width_none_matches_unlimited(self):
        """beam_width=None (default) should match unlimited search."""
        trajs = _make_trajectories(n_waypoints=6, n_r0=2, n_r1=2, seed=88)
        r_unlimited = compute_optimal_robot_assignment(trajs)
        r_none = compute_optimal_robot_assignment(trajs, beam_width=None)
        assert r_unlimited.total_cost == pytest.approx(r_none.total_cost)
        assert r_unlimited.sequence == r_none.sequence

    def test_beam_width_large_matches_unlimited(self):
        """A beam_width >= n_states should give the same result as unlimited."""
        trajs = _make_trajectories(n_waypoints=6, n_r0=2, n_r1=2, seed=99)
        r_unlimited = compute_optimal_robot_assignment(trajs)
        r_large = compute_optimal_robot_assignment(trajs, beam_width=100)
        assert r_unlimited.total_cost == pytest.approx(r_large.total_cost)
        assert r_unlimited.sequence == r_large.sequence


# ═══════════════════════════════════════════════════════════════════════════
# Tests — load_all_trajectories
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadAllTrajectories:
    """Tests for loading trajectory data from disk."""

    @staticmethod
    def _save_npz(path: str, **arrays):
        np.savez(path, **arrays)

    def test_load_roundtrip(self, tmp_path):
        n_wp = 5
        rng = np.random.default_rng(0)

        # Object trajectory
        self._save_npz(
            str(tmp_path / "object_trajectory.npz"),
            object_positions=rng.random((n_wp, 3)),
            object_quaternions=np.tile([0, 0, 0, 1], (n_wp, 1)).astype(float),
            object_frame_ids=np.array(["world"] * n_wp),
        )

        # Robot trajectories
        for rid in [0, 1]:
            for ti in range(2):
                self._save_npz(
                    str(tmp_path / f"robot{rid}_trajectory_{ti}.npz"),
                    eef_positions=rng.random((n_wp, 3)),
                    eef_quaternions=np.tile([0, 0, 0, 1], (n_wp, 1)).astype(float),
                    eef_frame_ids=np.array(["world"] * n_wp),
                    object_positions=rng.random((n_wp, 3)),
                    object_quaternions=np.tile([0, 0, 0, 1], (n_wp, 1)).astype(float),
                    object_frame_ids=np.array(["world"] * n_wp),
                    is_feasible=np.ones(n_wp, dtype=bool),
                    ik_success=np.ones(n_wp, dtype=bool),
                    collision_free=np.ones(n_wp, dtype=bool),
                    joint_configs=rng.random((n_wp, 7)),
                )

        trajs = load_all_trajectories(str(tmp_path))
        assert len(trajs.robot0_trajectories) == 2
        assert len(trajs.robot1_trajectories) == 2
        assert trajs.object_trajectory.positions.shape == (n_wp, 3)
        # Verify source_path is populated from file name
        assert trajs.robot0_trajectories[0].source_path == "robot0_trajectory_0.npz"
        assert trajs.robot0_trajectories[1].source_path == "robot0_trajectory_1.npz"
        assert trajs.robot1_trajectories[0].source_path == "robot1_trajectory_0.npz"

    def test_max_robot_limits(self, tmp_path):
        """max_robot0 / max_robot1 select only the first N trajectories."""
        n_wp = 5
        rng = np.random.default_rng(42)

        self._save_npz(
            str(tmp_path / "object_trajectory.npz"),
            object_positions=rng.random((n_wp, 3)),
            object_quaternions=np.tile([0, 0, 0, 1], (n_wp, 1)).astype(float),
            object_frame_ids=np.array(["world"] * n_wp),
        )
        for rid in [0, 1]:
            for ti in range(5):
                self._save_npz(
                    str(tmp_path / f"robot{rid}_trajectory_{ti}.npz"),
                    eef_positions=rng.random((n_wp, 3)),
                    eef_quaternions=np.tile([0, 0, 0, 1], (n_wp, 1)).astype(float),
                    eef_frame_ids=np.array(["world"] * n_wp),
                    object_positions=rng.random((n_wp, 3)),
                    object_quaternions=np.tile([0, 0, 0, 1], (n_wp, 1)).astype(float),
                    object_frame_ids=np.array(["world"] * n_wp),
                    is_feasible=np.ones(n_wp, dtype=bool),
                    ik_success=np.ones(n_wp, dtype=bool),
                    collision_free=np.ones(n_wp, dtype=bool),
                    joint_configs=rng.random((n_wp, 7)),
                )

        trajs = load_all_trajectories(str(tmp_path), max_robot0=3, max_robot1=2)
        assert len(trajs.robot0_trajectories) == 3
        assert len(trajs.robot1_trajectories) == 2
        # Verify numericaly sorted: 0, 1, 2
        assert trajs.robot0_trajectories[0].source_path == "robot0_trajectory_0.npz"
        assert trajs.robot0_trajectories[2].source_path == "robot0_trajectory_2.npz"
        assert trajs.robot1_trajectories[1].source_path == "robot1_trajectory_1.npz"

    def test_loaded_trajectories_work_with_assignment(self, tmp_path):
        """End-to-end: save, load, run assignment."""
        n_wp = 4
        rng = np.random.default_rng(1)

        self._save_npz(
            str(tmp_path / "object_trajectory.npz"),
            object_positions=rng.random((n_wp, 3)),
            object_quaternions=np.tile([0, 0, 0, 1], (n_wp, 1)).astype(float),
            object_frame_ids=np.array(["world"] * n_wp),
        )
        for rid in [0, 1]:
            self._save_npz(
                str(tmp_path / f"robot{rid}_trajectory_0.npz"),
                eef_positions=rng.random((n_wp, 3)),
                eef_quaternions=np.tile([0, 0, 0, 1], (n_wp, 1)).astype(float),
                eef_frame_ids=np.array(["world"] * n_wp),
                object_positions=rng.random((n_wp, 3)),
                object_quaternions=np.tile([0, 0, 0, 1], (n_wp, 1)).astype(float),
                object_frame_ids=np.array(["world"] * n_wp),
                is_feasible=np.ones(n_wp, dtype=bool),
                ik_success=np.ones(n_wp, dtype=bool),
                collision_free=np.ones(n_wp, dtype=bool),
                joint_configs=rng.random((n_wp, 7)),
            )

        trajs = load_all_trajectories(str(tmp_path))
        result = compute_optimal_robot_assignment(trajs)
        assert len(result.sequence) == n_wp - 1
        assert np.isfinite(result.total_cost)
        # trajectory_paths should reference the actual NPZ filenames
        for p in result.trajectory_paths:
            assert p.endswith(".npz")
            assert p.startswith("robot")

    def test_missing_object_trajectory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_all_trajectories(str(tmp_path))

    def test_no_robot_trajectories_gives_empty_lists(self, tmp_path):
        rng = np.random.default_rng(2)
        self._save_npz(
            str(tmp_path / "object_trajectory.npz"),
            object_positions=rng.random((3, 3)),
            object_quaternions=np.tile([0, 0, 0, 1], (3, 1)).astype(float),
            object_frame_ids=np.array(["world"] * 3),
        )
        trajs = load_all_trajectories(str(tmp_path))
        assert len(trajs.robot0_trajectories) == 0
        assert len(trajs.robot1_trajectories) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Tests — make_ik_solver_fn / make_jacobian_fns (mocked, no real env)
# ═══════════════════════════════════════════════════════════════════════════

class TestFactoryFunctions:
    """Test the factory helpers using mocked environment."""

    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.sim = MagicMock()
        env.sim.model = MagicMock()
        env.sim.data = MagicMock()
        return env

    @pytest.fixture
    def mock_tf_manager(self):
        tf = MagicMock(spec=["transform_pose"])
        tf.transform_pose.side_effect = lambda p, frame: p
        return tf

    @patch("robosuite_franka_manipulation.planning.assignment.factories.create_solver_cache")
    @patch("robosuite_franka_manipulation.planning.assignment.factories.enumerate_ik_solutions_for_arm")
    @patch("robosuite_franka_manipulation.planning.assignment.factories.apply_ik_orientation_correction")
    def test_make_ik_solver_fn_calls_ik(
        self, mock_correction, mock_ik, mock_cache, mock_env, mock_tf_manager,
    ):
        mock_correction.side_effect = lambda p: p
        mock_ik.return_value = [np.zeros(7)]
        mock_cache.return_value = MagicMock()

        ik_fn = make_ik_solver_fn(mock_env, mock_tf_manager, env_lock=None)
        pose = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        result = ik_fn(pose, robot_id=0)

        assert result is not None
        assert result.shape == (7,)
        mock_correction.assert_called_once()
        mock_ik.assert_called_once()
        # Verify robot_id was forwarded
        _, kwargs = mock_ik.call_args
        assert kwargs["robot_id"] == 0

    @patch("robosuite_franka_manipulation.planning.assignment.factories.create_solver_cache")
    @patch("robosuite_franka_manipulation.planning.assignment.factories.enumerate_ik_solutions_for_arm")
    @patch("robosuite_franka_manipulation.planning.assignment.factories.apply_ik_orientation_correction")
    def test_make_ik_solver_fn_returns_none_on_failure(
        self, mock_correction, mock_ik, mock_cache, mock_env, mock_tf_manager,
    ):
        mock_correction.side_effect = lambda p: p
        mock_ik.return_value = []  # no solutions
        mock_cache.return_value = MagicMock()

        ik_fn = make_ik_solver_fn(mock_env, mock_tf_manager)
        pose = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        result = ik_fn(pose, robot_id=1)

        assert result is None

    @patch("robosuite_franka_manipulation.planning.assignment.factories.compute_arm_jacobian")
    def test_make_jacobian_fns_returns_per_robot(self, mock_jac, mock_env):
        mock_jac.return_value = np.eye(6, 7)

        fns = make_jacobian_fns(mock_env, env_lock=None)
        assert 0 in fns
        assert 1 in fns

        q = np.zeros(7)
        J0 = fns[0](q)
        J1 = fns[1](q)

        assert J0.shape == (6, 7)
        assert J1.shape == (6, 7)
        assert mock_jac.call_count == 2

        # Verify correct robot_id was passed in each call
        calls = mock_jac.call_args_list
        assert calls[0][1]["robot_id"] == 0
        assert calls[1][1]["robot_id"] == 1

    @patch("robosuite_franka_manipulation.planning.assignment.factories.compute_arm_jacobian")
    def test_make_jacobian_fns_with_lock(self, mock_jac, mock_env):
        mock_jac.return_value = np.eye(6, 7)
        import threading
        lock = threading.RLock()

        fns = make_jacobian_fns(mock_env, env_lock=lock)
        fns[0](np.zeros(7))

        _, kwargs = mock_jac.call_args
        assert kwargs["env_lock"] is lock


class TestMakeCollisionCheckFn:
    """Test the make_collision_check_fn factory (mocked environment)."""

    @patch("robosuite_franka_manipulation.planning.assignment.factories.nullcontext")
    def test_collision_check_fn_calls_mujoco(self, _mock_ctx):
        """Verify the factory produces a callable that accesses MuJoCo."""
        import threading

        # Mock environment
        env = MagicMock()
        model = MagicMock()
        data = MagicMock()
        env.sim.model = model
        env.sim.data = data
        data.qpos = np.zeros(20)
        data.qvel = np.zeros(20)
        lock = threading.RLock()

        # Mock the imports used by the factory
        with patch(
            "robosuite_franka_manipulation.planning.assignment.factories.get_arm_joint_ids",
            side_effect=lambda m, rid: list(range(rid * 7, rid * 7 + 7)),
        ), patch(
            "robosuite_franka_manipulation.planning.assignment.factories.get_joint_qpos_indices",
            side_effect=lambda m, jids: np.array(jids),
        ), patch(
            "robosuite_franka_manipulation.planning.assignment.factories.check_inter_arm_collision",
            return_value=False,
        ) as mock_collision, patch(
            "robosuite_franka_manipulation.planning.assignment.factories.in_collision",
            return_value=False,
        ) as mock_in_collision, patch(
            "robosuite_franka_manipulation.planning.assignment.factories.get_native_mj_object",
            side_effect=lambda x: x,
        ), patch("mujoco.mj_forward"):
            fn = make_collision_check_fn(env, env_lock=lock)
            result = fn(np.ones(7), np.ones(7) * 2)

            assert result is True  # no collision
            mock_collision.assert_called_once()
            mock_in_collision.assert_called_once()

    @patch("robosuite_franka_manipulation.planning.assignment.factories.nullcontext")
    def test_collision_detected_returns_false(self, _mock_ctx):
        """When inter-arm collision is detected, the fn should return False."""
        env = MagicMock()
        model = MagicMock()
        data = MagicMock()
        env.sim.model = model
        env.sim.data = data
        data.qpos = np.zeros(20)
        data.qvel = np.zeros(20)

        with patch(
            "robosuite_franka_manipulation.planning.assignment.factories.get_arm_joint_ids",
            side_effect=lambda m, rid: list(range(rid * 7, rid * 7 + 7)),
        ), patch(
            "robosuite_franka_manipulation.planning.assignment.factories.get_joint_qpos_indices",
            side_effect=lambda m, jids: np.array(jids),
        ), patch(
            "robosuite_franka_manipulation.planning.assignment.factories.check_inter_arm_collision",
            return_value=True,  # collision!
        ), patch(
            "robosuite_franka_manipulation.planning.assignment.factories.in_collision",
            return_value=False,
        ), patch(
            "robosuite_franka_manipulation.planning.assignment.factories.get_native_mj_object",
            side_effect=lambda x: x,
        ), patch("mujoco.mj_forward"):
            fn = make_collision_check_fn(env)
            result = fn(np.ones(7), np.ones(7) * 2)

            assert result is False  # collision detected


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Experiment config assignment field
# ═══════════════════════════════════════════════════════════════════════════

class TestExperimentConfigAssignment:
    """Verify every experiment JSON loads with a valid assignment config."""

    def test_all_experiments_have_assignment(self):
        from robosuite_franka_manipulation.experiment_config import (
            list_experiments,
            load_experiment,
        )

        exps = list_experiments()
        assert len(exps) > 0, "No experiments found"

        for exp_path in exps:
            cfg = load_experiment(exp_path)
            asgn = cfg.assignment
            assert asgn is not None, f"{exp_path}: missing assignment config"
            assert isinstance(asgn.alpha, float)
            assert isinstance(asgn.beta, float)
            assert isinstance(asgn.epsilon, float)
            assert asgn.alpha > 0
            assert asgn.beta > 0
            assert asgn.epsilon > 0

    def test_assignment_defaults_match_constants(self):
        from robosuite_franka_manipulation.experiment_config import AssignmentConfig

        default = AssignmentConfig()
        assert default.alpha == constants.ASSIGNMENT_ALPHA
        assert default.beta == constants.ASSIGNMENT_BETA
        assert default.epsilon == constants.ASSIGNMENT_EPSILON
        assert default.initial_robot is None
        assert default.transfer_feasibility is None


# ═══════════════════════════════════════════════════════════════════════════
# Tests — Planning __init__ exports
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanningExports:
    """Verify the planning package re-exports the new symbols."""

    def test_exports_available(self):
        from robosuite_franka_manipulation import planning

        for name in (
            "compute_optimal_robot_assignment",
            "AssignmentResult",
            "ObjectTrajectory",
            "RobotTrajectory",
            "TrajectoryBundle",
            "save_assignment_result",
            "load_assignment_result",
            "load_all_trajectories",
            "make_collision_check_fn",
            "make_ik_solver_fn",
            "make_jacobian_fns",
        ):
            assert hasattr(planning, name), f"Missing export: {name}"
            assert name in planning.__all__, f"'{name}' not in __all__"
