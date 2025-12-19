#!/usr/bin/env python3
"""
Pytest-based test suite for trajectory planning functionality.
Tests SE(3) trajectory planning, polynomial interpolation, and motion planning.
"""

import pytest
import numpy as np

from robosuite_franka_manipulation.controllers import RobosuiteArmController


class TestTrajectoryPlanning:
    """Test trajectory planning components."""

    @pytest.fixture(scope="class")
    def controller(self):
        """Fixture to provide arm controller for trajectory planning tests."""
        # Create a minimal controller for testing
        controller = RobosuiteArmController.__new__(RobosuiteArmController)
        yield controller

    def test_trapezoidal_trajectory_planning(self, controller):
        """Test trapezoidal trajectory planning."""
        # Define start and end poses
        start_pos = np.array([0.0, 0.0, 0.0])
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        target_pos = np.array([0.1, 0.2, 0.3])
        target_quat = np.array([0.0, 0.0, 0.0, 1.0])

        # Plan trajectory
        positions, quaternions, total_time = controller.plan_trapezoidal_trajectory(
            start_pos, start_quat, target_pos, target_quat,
            max_vel=0.05, max_acc=0.02, dt=0.02
        )

        # Check trajectory structure
        assert isinstance(positions, np.ndarray)
        assert isinstance(quaternions, np.ndarray)
        assert isinstance(total_time, float)
        assert len(positions) > 0
        assert len(quaternions) > 0
        assert positions.shape[1] == 3  # 3D positions
        assert quaternions.shape[1] == 4  # 4D quaternions

        # Check that trajectory starts and ends at correct poses
        np.testing.assert_array_almost_equal(positions[0], start_pos, decimal=5)
        np.testing.assert_array_almost_equal(positions[-1], target_pos, decimal=5)
        np.testing.assert_array_almost_equal(quaternions[0], start_quat, decimal=5)
        np.testing.assert_array_almost_equal(quaternions[-1], target_quat, decimal=5)

    def test_trajectory_timing(self, controller):
        """Test that trajectory timing is correct."""
        start_pos = np.array([0.0, 0.0, 0.0])
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])
        target_pos = np.array([0.1, 0.0, 0.0])
        target_quat = np.array([0.0, 0.0, 0.0, 1.0])

        dt = 0.02
        positions, quaternions, total_time = controller.plan_trapezoidal_trajectory(
            start_pos, start_quat, target_pos, target_quat, dt=dt
        )

        # Check that total time is reasonable
        assert total_time > 0
        expected_steps = len(positions) - 1
        expected_time = expected_steps * dt
        assert abs(total_time - expected_time) < dt  # Should be close

    def test_zero_displacement_trajectory(self, controller):
        """Test planning with zero displacement."""
        pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])

        positions, quaternions, total_time = controller.plan_trapezoidal_trajectory(
            pos, quat, pos, quat
        )

        # Should return minimal trajectory
        assert len(positions) == 1
        assert len(quaternions) == 1
        assert total_time == 0.0
        np.testing.assert_array_almost_equal(positions[0], pos)
        np.testing.assert_array_almost_equal(quaternions[0], quat)

    def test_quaternion_interpolation_validity(self, controller):
        """Test that quaternion interpolation produces valid quaternions."""
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])
        target_quat = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])  # 90 degree rotation

        positions, quaternions, total_time = controller.plan_trapezoidal_trajectory(
            np.array([0.0, 0.0, 0.0]), start_quat,
            np.array([0.1, 0.0, 0.0]), target_quat
        )

        # Check that all quaternions are valid (unit length)
        for quat in quaternions:
            norm = np.linalg.norm(quat)
            assert abs(norm - 1.0) < 1e-6

    def test_trajectory_smoothness(self, controller):
        """Test that trajectories are smooth."""
        start_pos = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([0.3, 0.2, 0.1])

        positions, quaternions, total_time = controller.plan_trapezoidal_trajectory(
            start_pos, np.array([0.0, 0.0, 0.0, 1.0]),
            target_pos, np.array([0.0, 0.0, 0.0, 1.0]),
            dt=0.01
        )

        # Check position continuity
        for i in range(1, len(positions)):
            displacement = np.linalg.norm(positions[i] - positions[i-1])
            # Displacement should not be too large (smooth trajectory)
            assert displacement < 0.01  # Based on dt=0.01 and reasonable velocity