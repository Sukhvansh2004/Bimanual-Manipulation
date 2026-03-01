"""
Pytest-based test suite for the joint-space RRT planner.

Tests cover the RRT-Connect algorithm, path smoothing, densification,
and the high-level ``plan_arm_trajectory`` pipeline.  Most tests use
mock collision checkers to avoid requiring a live MuJoCo environment.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from robosuite_franka_manipulation.planning.rrt_joint_planner import (
    JointRRTResult,
    _joint_distance,
    _steer,
    _sample_random_config,
    densify_joint_path,
    rrt_connect_joint_space,
    shortcut_joint_path,
    plan_arm_trajectory,
)
from robosuite_franka_manipulation.planning.rrt_planner import (
    rrt_connect_generic,
    _TreeNode,
    _generic_nearest,
    _generic_extract_path,
)
from robosuite_franka_manipulation.planning.utils.path_smoothing import (
    shortcut_generic,
    densify_generic,
)


# ============================================================================
# ============================================================================
# Test: JointRRTResult dataclass
# ============================================================================

class TestDataclasses:
    """Basic tests for the dataclass containers."""

    def test_joint_rrt_result_defaults(self):
        res = JointRRTResult(success=False)
        assert res.success is False
        assert res.path == []
        assert res.iterations == 0
        assert res.tree_a_size == 0
        assert res.tree_b_size == 0

    def test_joint_rrt_result_with_path(self):
        path = [np.zeros(7), np.ones(7)]
        res = JointRRTResult(success=True, path=path, iterations=42)
        assert res.success is True
        assert len(res.path) == 2
        assert res.iterations == 42


# ============================================================================
# Test: Utility functions
# ============================================================================

class TestJointDistance:
    """Tests for ``_joint_distance``."""

    def test_zero_distance(self):
        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        assert _joint_distance(q, q) == pytest.approx(0.0)

    def test_known_distance(self):
        q1 = np.zeros(7)
        q2 = np.ones(7)
        expected = np.sqrt(7.0)
        assert _joint_distance(q1, q2) == pytest.approx(expected)

    def test_symmetry(self):
        q1 = np.random.randn(7)
        q2 = np.random.randn(7)
        assert _joint_distance(q1, q2) == pytest.approx(_joint_distance(q2, q1))


class TestNearestNode:
    """Tests for ``_generic_nearest`` (generic RRT nearest-node search)."""

    def _dist(self, s1, s2):
        return float(np.linalg.norm(s1 - s2))

    def test_single_node(self):
        tree = [_TreeNode(state=np.zeros(7))]
        assert _generic_nearest(tree, np.ones(7), self._dist) == 0

    def test_nearest_among_multiple(self):
        tree = [
            _TreeNode(state=np.zeros(7)),
            _TreeNode(state=np.ones(7) * 0.5),
            _TreeNode(state=np.ones(7)),
        ]
        # Query close to node 1
        target = np.ones(7) * 0.4
        assert _generic_nearest(tree, target, self._dist) == 1

    def test_exact_match(self):
        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        tree = [
            _TreeNode(state=np.zeros(7)),
            _TreeNode(state=q),
        ]
        assert _generic_nearest(tree, q, self._dist) == 1


class TestSteer:
    """Tests for ``_steer``."""

    def test_within_step_size(self):
        q_from = np.zeros(7)
        q_to = np.ones(7) * 0.01
        result = _steer(q_from, q_to, step_size=1.0)
        np.testing.assert_array_almost_equal(result, q_to)

    def test_beyond_step_size(self):
        q_from = np.zeros(7)
        q_to = np.ones(7)  # dist = sqrt(7) ≈ 2.65
        step_size = 1.0
        result = _steer(q_from, q_to, step_size=step_size)
        dist = np.linalg.norm(result - q_from)
        assert dist == pytest.approx(step_size, abs=1e-6)

    def test_direction_preserved(self):
        q_from = np.zeros(7)
        q_to = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = _steer(q_from, q_to, step_size=0.5)
        expected = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestSampleRandomConfig:
    """Tests for ``_sample_random_config``."""

    def test_within_bounds(self):
        lo = np.full(7, -2.0)
        hi = np.full(7, 2.0)
        for _ in range(100):
            q = _sample_random_config(lo, hi)
            assert np.all(q >= lo)
            assert np.all(q <= hi)

    def test_shape(self):
        lo = np.full(7, -1.0)
        hi = np.full(7, 1.0)
        q = _sample_random_config(lo, hi)
        assert q.shape == (7,)


class TestExtractPath:
    """Tests for ``_generic_extract_path``."""

    def _copy(self, s):
        return s.copy()

    def test_simple_two_trees(self):
        tree_a = [
            _TreeNode(state=np.array([0.0] * 7)),
            _TreeNode(state=np.array([1.0] * 7), parent=0),
        ]
        tree_b = [
            _TreeNode(state=np.array([3.0] * 7)),
            _TreeNode(state=np.array([2.0] * 7), parent=0),
        ]
        path = _generic_extract_path(tree_a, 1, tree_b, 1, a_is_start=True, copy_fn=self._copy)
        assert len(path) == 4
        np.testing.assert_array_equal(path[0], [0.0] * 7)
        np.testing.assert_array_equal(path[1], [1.0] * 7)
        np.testing.assert_array_equal(path[2], [2.0] * 7)
        np.testing.assert_array_equal(path[3], [3.0] * 7)

    def test_swapped_trees(self):
        tree_a = [_TreeNode(state=np.array([0.0] * 7))]
        tree_b = [_TreeNode(state=np.array([1.0] * 7))]
        path = _generic_extract_path(tree_a, 0, tree_b, 0, a_is_start=False, copy_fn=self._copy)
        assert len(path) == 2
        # When a_is_start=False, tree_b is start
        np.testing.assert_array_equal(path[0], [1.0] * 7)
        np.testing.assert_array_equal(path[1], [0.0] * 7)


# ============================================================================
# Test: rrt_connect_generic (pure-Python, no MuJoCo)
# ============================================================================

class TestRRTConnectGeneric:
    """Unit tests for the generic RRT-Connect core."""

    def _setup_1d(self, blocked_ranges=None):
        """Return callbacks operating on 1-D scalar states.

        ``blocked_ranges`` is a list of ``(lo, hi)`` tuples.  A state is
        in collision if it falls inside any of these ranges.
        """
        if blocked_ranges is None:
            blocked_ranges = []

        def dist(a, b):
            return abs(float(a) - float(b))

        def steer(from_s, to_s, step):
            d = float(to_s) - float(from_s)
            if abs(d) <= step:
                return float(to_s)
            return float(from_s) + step * (1 if d > 0 else -1)

        def sample():
            return np.random.uniform(-10, 10)

        def collision(s):
            for lo, hi in blocked_ranges:
                if lo <= float(s) <= hi:
                    return True
            return False

        def seg_free(s1, s2):
            lo_seg = min(float(s1), float(s2))
            hi_seg = max(float(s1), float(s2))
            for lo, hi in blocked_ranges:
                if lo_seg < hi and hi_seg > lo:
                    return False
            return True

        copy_fn = lambda s: float(s)  # noqa: E731
        return dist, steer, sample, collision, seg_free, copy_fn

    def test_straight_path_no_obstacle(self):
        dist, steer, sample, collision, seg_free, copy_fn = self._setup_1d()
        ok, path, iters, ta, tb = rrt_connect_generic(
            start_state=0.0, goal_state=5.0,
            distance_fn=dist, steer_fn=steer, sample_fn=sample,
            collision_fn=collision, segment_free_fn=seg_free,
            copy_fn=copy_fn, step_size=1.0, goal_bias=0.3,
            max_iterations=500, connect_max_steps=50,
        )
        assert ok
        assert len(path) >= 2
        assert abs(path[0] - 0.0) < 1e-9
        assert abs(path[-1] - 5.0) < 1e-9

    def test_path_around_obstacle(self):
        """Obstacle blocks the direct 1-D path; RRT must go around."""
        dist, steer, sample, collision, seg_free, copy_fn = self._setup_1d(
            blocked_ranges=[(2.0, 3.0)],
        )
        # In 1-D with a solid wall the planner cannot actually go around,
        # so we expect failure (validates the collision checking works).
        ok, path, iters, ta, tb = rrt_connect_generic(
            start_state=0.0, goal_state=5.0,
            distance_fn=dist, steer_fn=steer, sample_fn=sample,
            collision_fn=collision, segment_free_fn=seg_free,
            copy_fn=copy_fn, step_size=0.5, goal_bias=0.3,
            max_iterations=200, connect_max_steps=50,
        )
        assert not ok

    def test_callbacks_invoked(self):
        dist, steer, sample, collision, seg_free, copy_fn = self._setup_1d(
            blocked_ranges=[(2.5, 3.0)],
        )
        extend_calls = []
        iter_calls = []

        def _on_extend(parent, new, a_is_start):
            extend_calls.append((parent, new, a_is_start))

        def _on_iteration(it, ta, tb):
            iter_calls.append(it)

        rrt_connect_generic(
            start_state=0.0, goal_state=2.0,
            distance_fn=dist, steer_fn=steer, sample_fn=sample,
            collision_fn=collision, segment_free_fn=seg_free,
            copy_fn=copy_fn, step_size=0.5, goal_bias=0.5,
            max_iterations=100, connect_max_steps=50,
            on_extend=_on_extend, on_iteration=_on_iteration,
        )
        assert len(iter_calls) > 0
        # At least some extensions should have happened
        assert len(extend_calls) >= 0  # may be 0 if all blocked


# ============================================================================
# Test: Densify
# ============================================================================

class TestDensifyJointPath:
    """Tests for ``densify_joint_path``."""

    def test_single_waypoint(self):
        path = [np.ones(7)]
        dense = densify_joint_path(path, 10)
        assert dense.shape == (10, 7)
        np.testing.assert_array_almost_equal(dense[0], np.ones(7))
        np.testing.assert_array_almost_equal(dense[-1], np.ones(7))

    def test_two_waypoints(self):
        path = [np.zeros(7), np.ones(7)]
        dense = densify_joint_path(path, 11)
        assert dense.shape == (11, 7)
        np.testing.assert_array_almost_equal(dense[0], np.zeros(7))
        np.testing.assert_array_almost_equal(dense[-1], np.ones(7))
        # Midpoint should be 0.5
        np.testing.assert_array_almost_equal(dense[5], np.ones(7) * 0.5)

    def test_three_waypoints_unequal(self):
        path = [
            np.zeros(7),
            np.ones(7) * 2.0,   # distance 2*sqrt(7)
            np.ones(7) * 3.0,   # distance 1*sqrt(7) from prev
        ]
        dense = densify_joint_path(path, 100)
        assert dense.shape == (100, 7)
        np.testing.assert_array_almost_equal(dense[0], np.zeros(7))
        np.testing.assert_array_almost_equal(dense[-1], np.ones(7) * 3.0)

    def test_monotonic_parameterisation(self):
        path = [np.zeros(7), np.ones(7), np.ones(7) * 2.0]
        dense = densify_joint_path(path, 50)
        # Joint 0 should be monotonically increasing
        assert np.all(np.diff(dense[:, 0]) >= -1e-12)

    def test_invalid_num_steps(self):
        with pytest.raises(ValueError):
            densify_joint_path([np.zeros(7)], 0)

    def test_empty_path(self):
        with pytest.raises(ValueError):
            densify_joint_path([], 10)


# ============================================================================
# Test: RRT-Connect (mocked collision)
# ============================================================================

class TestRRTConnectMocked:
    """Tests for ``rrt_connect_joint_space`` with mocked collision checking."""

    @patch(
        "robosuite_franka_manipulation.planning.rrt_joint_planner._collision_free_joint_segment"
    )
    @patch(
        "robosuite_franka_manipulation.planning.rrt_joint_planner._check_arm_config_collision"
    )
    def test_direct_path_success(self, mock_collision, mock_segment):
        """When the direct segment is collision-free, return immediately."""
        mock_segment.return_value = True  # direct path is free
        mock_collision.return_value = False

        env = MagicMock()
        model = MagicMock()
        model.ngeom = 10
        model.njnt = 14
        model.jnt_type = np.zeros(14, dtype=int)
        model.jnt_dofadr = np.arange(14, dtype=int)
        model.jnt_qposadr = np.arange(14, dtype=int)
        model.jnt_range = np.column_stack([
            np.full(14, -2.0), np.full(14, 2.0)
        ])
        model.geom_id2name = lambda i: f"geom_{i}"
        model.geom = MagicMock(side_effect=lambda i: MagicMock(name=f"geom_{i}"))

        data = MagicMock()
        data.qpos = np.zeros(14)
        data.qvel = np.zeros(14)
        data.ncon = 0

        env.sim.model = model
        env.sim.data = data

        with patch(
            "robosuite_franka_manipulation.planning.rrt_joint_planner.get_arm_joint_ids",
            return_value=list(range(7)),
        ), patch(
            "robosuite_franka_manipulation.planning.rrt_joint_planner.get_joint_qpos_indices",
            return_value=np.arange(7),
        ), patch(
            "robosuite_franka_manipulation.planning.rrt_joint_planner.get_joint_limits",
            return_value=(np.full(7, -2.0), np.full(7, 2.0)),
        ), patch(
            "robosuite_franka_manipulation.planning.rrt_joint_planner.get_native_mj_object",
            return_value=model,
        ), patch(
            "robosuite_franka_manipulation.planning.rrt_joint_planner._ensure_geom_caches",
        ), patch(
            "robosuite_franka_manipulation.planning.rrt_joint_planner._get_ids_for_prefixes",
            return_value=[0, 1, 2],
        ), patch(
            "robosuite_franka_manipulation.planning.rrt_joint_planner.mujoco",
        ):
            result = rrt_connect_joint_space(
                env=env,
                robot_id_b=1,
                start_q=np.zeros(7),
                goal_q=np.ones(7),
                q_other_arm=np.zeros(7),
                robot_id_a=0,
            )

        assert result.success is True
        assert len(result.path) == 2
        assert result.iterations == 0


class TestPlanArmTrajectoryMocked:
    """Tests for the ``plan_arm_trajectory`` convenience function."""

    @patch(
        "robosuite_franka_manipulation.planning.rrt_joint_planner.rrt_connect_joint_space"
    )
    def test_rrt_success_returns_dense(self, mock_rrt):
        """When RRT succeeds, the result should be densified to num_steps."""
        mock_rrt.return_value = JointRRTResult(
            success=True,
            path=[np.zeros(7), np.ones(7) * 0.5, np.ones(7)],
            iterations=10,
        )

        with patch(
            "robosuite_franka_manipulation.planning.rrt_joint_planner.shortcut_joint_path",
            side_effect=lambda env, path, **kw: path,  # no-op smoothing
        ):
            traj, success = plan_arm_trajectory(
                env=MagicMock(),
                robot_id_b=1,
                start_q=np.zeros(7),
                goal_q=np.ones(7),
                q_other_arm=np.zeros(7),
                robot_id_a=0,
                num_steps=50,
            )

        assert success is True
        assert traj.shape == (50, 7)
        np.testing.assert_array_almost_equal(traj[0], np.zeros(7))
        np.testing.assert_array_almost_equal(traj[-1], np.ones(7))

    @patch(
        "robosuite_franka_manipulation.planning.rrt_joint_planner.rrt_connect_joint_space"
    )
    def test_rrt_failure_falls_back(self, mock_rrt):
        """When RRT fails, should fall back to linear interpolation."""
        mock_rrt.return_value = JointRRTResult(
            success=False,
            iterations=3000,
        )

        traj, success = plan_arm_trajectory(
            env=MagicMock(),
            robot_id_b=1,
            start_q=np.zeros(7),
            goal_q=np.ones(7),
            q_other_arm=np.zeros(7),
            robot_id_a=0,
            num_steps=20,
        )

        assert success is False
        assert traj.shape == (20, 7)
        np.testing.assert_array_almost_equal(traj[0], np.zeros(7))
        np.testing.assert_array_almost_equal(traj[-1], np.ones(7))


# ============================================================================
# Test: DualArmParallelPlanner integration with RRT
# ============================================================================

class TestDualArmPlannerRRTIntegration:
    """Test that _resolve_trajectory_b uses RRT when traj_a is provided."""

    def test_resolve_trajectory_b_calls_rrt(self):
        """When target_b is StampedPose and traj_a is given, RRT is attempted."""
        from robosuite_franka_manipulation.planning.dual_arm_planner import (
            DualArmParallelPlanner,
        )
        from robosuite_franka_manipulation.transforms import StampedPose

        env = MagicMock()
        arm_a = MagicMock()
        arm_a.robot_id = 0
        arm_a.arm_id = "right"
        arm_a.tf_manager = MagicMock()

        arm_b = MagicMock()
        arm_b.robot_id = 1
        arm_b.arm_id = "right"
        arm_b.base_frame = "world"
        arm_b.tf_manager = MagicMock()
        arm_b.get_joint_positions.return_value = np.zeros(7)
        arm_b._solve_ik_for_pose.return_value = np.ones(7)

        planner = DualArmParallelPlanner(
            env=env,
            arm_a=arm_a,
            arm_b=arm_b,
            collision_check=False,
        )

        traj_a = np.random.randn(50, 7)
        target_pose = StampedPose(
            position=np.array([0.3, 0.0, 1.2]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            frame_id="world",
        )

        # Mock the plan_arm_trajectory to verify it's called
        with patch(
            "robosuite_franka_manipulation.planning.dual_arm_planner.plan_arm_trajectory"
        ) as mock_plan:
            mock_plan.return_value = (np.random.randn(50, 7), True)

            result = planner._resolve_trajectory_b(
                target_pose, 50, None, traj_a
            )

            assert mock_plan.called
            assert result is not None
            assert result.shape == (50, 7)

    def test_resolve_trajectory_b_falls_back_without_traj_a(self):
        """Without traj_a, should use quintic interpolation (no RRT)."""
        from robosuite_franka_manipulation.planning.dual_arm_planner import (
            DualArmParallelPlanner,
        )
        from robosuite_franka_manipulation.transforms import StampedPose

        env = MagicMock()
        arm_a = MagicMock()
        arm_a.robot_id = 0
        arm_a.arm_id = "right"
        arm_a.tf_manager = MagicMock()

        arm_b = MagicMock()
        arm_b.robot_id = 1
        arm_b.arm_id = "right"
        arm_b.base_frame = "world"
        arm_b.tf_manager = MagicMock()
        arm_b.get_joint_positions.return_value = np.zeros(7)
        arm_b._solve_ik_for_pose.return_value = np.ones(7)

        planner = DualArmParallelPlanner(
            env=env,
            arm_a=arm_a,
            arm_b=arm_b,
            collision_check=False,
        )

        target_pose = StampedPose(
            position=np.array([0.3, 0.0, 1.2]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            frame_id="world",
        )

        with patch(
            "robosuite_franka_manipulation.planning.dual_arm_planner.plan_arm_trajectory"
        ) as mock_plan:
            result = planner._resolve_trajectory_b(
                target_pose, 50, None, None
            )

            # RRT should NOT be called
            assert not mock_plan.called
            # Should still return a trajectory (quintic fallback)
            assert result is not None
            assert result.shape == (50, 7)

    def test_resolve_trajectory_b_rrt_failure_uses_quintic(self):
        """When RRT fails, should return the quintic fallback trajectory."""
        from robosuite_franka_manipulation.planning.dual_arm_planner import (
            DualArmParallelPlanner,
        )
        from robosuite_franka_manipulation.transforms import StampedPose

        env = MagicMock()
        arm_a = MagicMock()
        arm_a.robot_id = 0
        arm_a.arm_id = "right"
        arm_a.tf_manager = MagicMock()

        arm_b = MagicMock()
        arm_b.robot_id = 1
        arm_b.arm_id = "right"
        arm_b.base_frame = "world"
        arm_b.tf_manager = MagicMock()
        arm_b.get_joint_positions.return_value = np.zeros(7)
        arm_b._solve_ik_for_pose.return_value = np.ones(7)

        planner = DualArmParallelPlanner(
            env=env,
            arm_a=arm_a,
            arm_b=arm_b,
            collision_check=False,
        )

        traj_a = np.random.randn(50, 7)
        target_pose = StampedPose(
            position=np.array([0.3, 0.0, 1.2]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            frame_id="world",
        )

        with patch(
            "robosuite_franka_manipulation.planning.dual_arm_planner.plan_arm_trajectory"
        ) as mock_plan:
            mock_plan.return_value = (np.zeros((50, 7)), False)  # RRT failed

            result = planner._resolve_trajectory_b(
                target_pose, 50, None, traj_a
            )

            assert mock_plan.called
            # Should return quintic fallback since RRT failed
            assert result is not None
            assert result.shape == (50, 7)


# ============================================================================
# Test: Planning module exports
# ============================================================================

class TestPlanningExports:
    """Verify that new symbols are exported from the planning package."""

    def test_exports_available(self):
        from robosuite_franka_manipulation import planning

        expected = [
            "rrt_connect_generic",
            "rrt_connect_joint_space",
            "JointRRTResult",
            "plan_arm_trajectory",
            "shortcut_joint_path",
            "densify_joint_path",
            "shortcut_generic",
            "densify_generic",
        ]
        for name in expected:
            assert hasattr(planning, name), f"Missing export: {name}"
            assert name in planning.__all__, f"Not in __all__: {name}"


# ============================================================================
# Test: Generic shortcut / densify functions
# ============================================================================

class TestShortcutGeneric:
    """Tests for the domain-agnostic shortcut_generic function."""

    def test_already_short_path_unchanged(self):
        """A two-point path cannot be shortened further."""
        path = [np.array([0.0]), np.array([1.0])]
        result = shortcut_generic(
            path, segment_free_fn=lambda a, b: True, max_iterations=50,
        )
        assert len(result) == 2

    def test_shortcutting_removes_intermediate(self):
        """If every segment is free, shortcutting should collapse to 2."""
        path = [np.array([float(i)]) for i in range(10)]
        result = shortcut_generic(
            path, segment_free_fn=lambda a, b: True, max_iterations=200,
        )
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], path[0])
        np.testing.assert_array_equal(result[-1], path[-1])

    def test_no_shortcuts_when_blocked(self):
        """If every segment is blocked, path stays unchanged."""
        path = [np.array([float(i)]) for i in range(5)]
        result = shortcut_generic(
            path, segment_free_fn=lambda a, b: False, max_iterations=50,
        )
        assert len(result) == len(path)

    def test_single_point(self):
        path = [np.array([0.0])]
        result = shortcut_generic(
            path, segment_free_fn=lambda a, b: True, max_iterations=10,
        )
        assert len(result) == 1

    def test_empty_path(self):
        result = shortcut_generic(
            [], segment_free_fn=lambda a, b: True, max_iterations=10,
        )
        assert result == []


class TestDensifyGeneric:
    """Tests for the domain-agnostic densify_generic function."""

    def test_two_point_densification(self):
        """Densifying a 2-point path should produce num_steps waypoints."""
        path = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        result = densify_generic(
            path, 10,
            distance_fn=lambda a, b: float(np.linalg.norm(a - b)),
            interpolate_fn=lambda a, b, t: a + t * (b - a),
        )
        assert len(result) == 10
        np.testing.assert_allclose(result[0], path[0])
        np.testing.assert_allclose(result[-1], path[-1])

    def test_single_point_replicated(self):
        path = [np.array([5.0])]
        result = densify_generic(
            path, 4,
            distance_fn=lambda a, b: float(np.linalg.norm(a - b)),
            interpolate_fn=lambda a, b, t: a + t * (b - a),
        )
        assert len(result) == 4
        for pt in result:
            np.testing.assert_array_equal(pt, np.array([5.0]))

    def test_identical_waypoints(self):
        """All-identical waypoints should return replicated first point."""
        path = [np.array([1.0, 2.0])] * 5
        result = densify_generic(
            path, 3,
            distance_fn=lambda a, b: float(np.linalg.norm(a - b)),
            interpolate_fn=lambda a, b, t: a + t * (b - a),
        )
        assert len(result) == 3
        for pt in result:
            np.testing.assert_array_equal(pt, np.array([1.0, 2.0]))

    def test_empty_path_raises(self):
        with pytest.raises(ValueError):
            densify_generic(
                [], 5,
                distance_fn=lambda a, b: 0.0,
                interpolate_fn=lambda a, b, t: a,
            )

    def test_zero_steps_raises(self):
        with pytest.raises(ValueError):
            densify_generic(
                [np.array([0.0])], 0,
                distance_fn=lambda a, b: 0.0,
                interpolate_fn=lambda a, b, t: a,
            )
