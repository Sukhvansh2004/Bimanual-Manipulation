#!/usr/bin/env python3
"""
Comprehensive tests for trajectory math utilities (trajectory_math.py).

Covers quintic scalar profiling, joint trajectory planning,
velocity profiling, and resampling.
"""

import pytest
import numpy as np

from bimanual_manipulation.utils.trajectory_math import (
    quintic_scalar,
    plan_quintic_joint_trajectory,
    velocity_profile_joint_trajectory,
    resample_joint_trajectory,
)


# ═══════════════════════════════════════════════════════════════════════════════
# quintic_scalar
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuinticScalar:
    """Test quintic scalar profiling function."""

    def test_single_step(self):
        """n=1 should return [1.0]."""
        s = quintic_scalar(1)
        assert len(s) == 1
        np.testing.assert_allclose(s, [1.0])

    def test_boundary_values(self):
        """Profile should start at 0 and end at 1."""
        s = quintic_scalar(100)
        assert len(s) == 100
        np.testing.assert_allclose(s[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(s[-1], 1.0, atol=1e-12)

    def test_monotonically_increasing(self):
        """Quintic profile should be monotonically non-decreasing."""
        s = quintic_scalar(200)
        diffs = np.diff(s)
        assert np.all(diffs >= -1e-15), "Quintic profile should be monotonically increasing"

    def test_midpoint_value(self):
        """At t=0.5, quintic = 6*(0.5)^5 - 15*(0.5)^4 + 10*(0.5)^3 = 0.5."""
        s = quintic_scalar(101)  # Odd to hit exact midpoint
        mid = s[50]
        np.testing.assert_allclose(mid, 0.5, atol=1e-10)

    def test_symmetry(self):
        """s(t) + s(1-t) = 1 for the quintic profile (anti-symmetric around 0.5)."""
        s = quintic_scalar(100)
        for i in range(len(s)):
            np.testing.assert_allclose(s[i] + s[-(i + 1)], 1.0, atol=1e-10)

    def test_smooth_acceleration(self):
        """First and second derivatives should be zero at endpoints (zero velocity/accel)."""
        n = 1000
        s = quintic_scalar(n)
        dt = 1.0 / (n - 1)
        # Finite difference for velocity at start and end
        vel_start = (s[1] - s[0]) / dt
        vel_end = (s[-1] - s[-2]) / dt
        # Should be close to zero (exact zero at continuous limit)
        assert vel_start < 0.01
        assert vel_end < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# plan_quintic_joint_trajectory
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlanQuinticJointTrajectory:
    """Test joint-space quintic trajectory planning."""

    def test_start_and_end_match(self):
        """Trajectory should start at start and end at target."""
        start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        target = np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.3, 0.1])
        traj = plan_quintic_joint_trajectory(start, target, 50)
        assert traj.shape == (50, 7)
        np.testing.assert_array_almost_equal(traj[0], start, decimal=12)
        np.testing.assert_array_almost_equal(traj[-1], target, decimal=12)

    def test_zero_displacement(self):
        """Same start and target should produce constant trajectory."""
        start = np.array([0.5, -0.3, 0.1, 0.0, 0.5, -0.2, 0.0])
        traj = plan_quintic_joint_trajectory(start, start, 30)
        for i in range(30):
            np.testing.assert_array_almost_equal(traj[i], start)

    def test_single_step(self):
        """Single step should jump directly to target."""
        start = np.zeros(7)
        target = np.ones(7)
        traj = plan_quintic_joint_trajectory(start, target, 1)
        assert traj.shape == (1, 7)
        np.testing.assert_array_almost_equal(traj[0], target)

    def test_different_dimensions(self):
        """Should work with arbitrary dimension."""
        start = np.zeros(3)
        target = np.array([1.0, 2.0, 3.0])
        traj = plan_quintic_joint_trajectory(start, target, 20)
        assert traj.shape == (20, 3)

    def test_monotonic_progress(self):
        """Each joint should monotonically interpolate (for single-direction move)."""
        start = np.zeros(7)
        target = np.ones(7) * 2.0
        traj = plan_quintic_joint_trajectory(start, target, 100)
        for j in range(7):
            diffs = np.diff(traj[:, j])
            assert np.all(diffs >= -1e-10), f"Joint {j} not monotonically increasing"


# ═══════════════════════════════════════════════════════════════════════════════
# velocity_profile_joint_trajectory
# ═══════════════════════════════════════════════════════════════════════════════

class TestVelocityProfileJointTrajectory:
    """Test quintic velocity profiling of multi-waypoint trajectories."""

    def test_preserves_endpoints(self):
        """Start and end configurations should be preserved."""
        waypoints = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ])
        profiled = velocity_profile_joint_trajectory(waypoints)
        np.testing.assert_array_almost_equal(profiled[0], waypoints[0], decimal=5)
        np.testing.assert_array_almost_equal(profiled[-1], waypoints[-1], decimal=5)

    def test_same_length_output(self):
        """Without num_steps, output should have same length as input."""
        waypoints = np.random.randn(50, 7)
        profiled = velocity_profile_joint_trajectory(waypoints)
        assert profiled.shape == waypoints.shape

    def test_custom_num_steps(self):
        """With num_steps, output should be resampled."""
        waypoints = np.random.randn(50, 7)
        profiled = velocity_profile_joint_trajectory(waypoints, num_steps=100)
        assert profiled.shape == (100, 7)

    def test_two_waypoints_passthrough(self):
        """Two waypoints or fewer should be returned as-is."""
        waypoints = np.array([[0.0, 0.0], [1.0, 1.0]])
        profiled = velocity_profile_joint_trajectory(waypoints)
        np.testing.assert_array_equal(profiled, waypoints)

    def test_stationary_trajectory(self):
        """All-same waypoints should return constant trajectory."""
        waypoints = np.tile([0.5, -0.3, 0.1], (20, 1))
        profiled = velocity_profile_joint_trajectory(waypoints)
        for i in range(len(profiled)):
            np.testing.assert_array_almost_equal(profiled[i], waypoints[0])


# ═══════════════════════════════════════════════════════════════════════════════
# resample_joint_trajectory
# ═══════════════════════════════════════════════════════════════════════════════

class TestResampleJointTrajectory:
    """Test cubic resampling of joint trajectories."""

    def test_identity_resample(self):
        """Resampling to same length should return identical trajectory."""
        waypoints = np.random.randn(50, 7)
        resampled = resample_joint_trajectory(waypoints, 50)
        np.testing.assert_array_almost_equal(resampled, waypoints)

    def test_upsample(self):
        """Upsampling should produce more waypoints."""
        waypoints = np.array([
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.5],
            [0.75, 0.75, 0.75],
            [1.0, 1.0, 1.0],
        ])
        resampled = resample_joint_trajectory(waypoints, 100)
        assert resampled.shape == (100, 3)
        np.testing.assert_array_almost_equal(resampled[0], waypoints[0])
        np.testing.assert_array_almost_equal(resampled[-1], waypoints[-1])

    def test_downsample(self):
        """Downsampling should produce fewer waypoints."""
        waypoints = np.linspace(0, 1, 100).reshape(-1, 1)
        resampled = resample_joint_trajectory(waypoints, 20)
        assert resampled.shape == (20, 1)
        np.testing.assert_allclose(resampled[0, 0], 0.0, atol=1e-10)
        np.testing.assert_allclose(resampled[-1, 0], 1.0, atol=1e-10)

    def test_single_waypoint(self):
        """Single waypoint should tile to target length."""
        wp = np.array([[0.5, -0.3, 0.1, 0.0, 0.5, -0.2, 0.0]])
        resampled = resample_joint_trajectory(wp, 50)
        assert resampled.shape == (50, 7)
        for i in range(50):
            np.testing.assert_array_almost_equal(resampled[i], wp[0])

    def test_preserves_linear_trajectory(self):
        """Linear trajectory should be preserved after cubic resample (cubic fits linear)."""
        t = np.linspace(0, 1, 10)
        waypoints = np.column_stack([t, 2 * t, -t])  # Linear in all dims
        resampled = resample_joint_trajectory(waypoints, 50)
        assert resampled.shape == (50, 3)
        # Midpoint should be approximately [0.5, 1.0, -0.5]
        mid = resampled[25]
        np.testing.assert_allclose(mid, [0.5, 1.0, -0.5], atol=0.05)
