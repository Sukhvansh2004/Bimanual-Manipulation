#!/usr/bin/env python3
"""
Pytest-based test suite for trajectory planning functionality.
Tests SE(3) trajectory planning using Lie algebra for proper coupled motion.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation.controllers.trajectory_planner import plan_quintic_trajectory
from bimanual_manipulation.utils.se3 import (
    skew,
    unskew,
    se3_to_matrix,
    matrix_to_se3,
    SE3_log,
    SE3_exp,
    pose_to_SE3,
    SE3_to_pose,
    compute_se3_distance,
)


class TestSE3Utilities:
    """Test SE(3) Lie algebra utilities."""
    
    def test_skew_unskew_inverse(self):
        """Test that skew and unskew are inverses."""
        v = np.array([1.0, 2.0, 3.0])
        S = skew(v)
        v_recovered = unskew(S)
        np.testing.assert_array_almost_equal(v, v_recovered)
        
        # Check skew-symmetric property
        assert np.allclose(S, -S.T)
    
    def test_se3_matrix_conversion(self):
        """Test se(3) twist to matrix and back."""
        twist = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        mat = se3_to_matrix(twist)
        twist_recovered = matrix_to_se3(mat)
        np.testing.assert_array_almost_equal(twist, twist_recovered)
    
    def test_SE3_log_exp_inverse_identity(self):
        """Test that exp(log(I)) = I."""
        T_id = np.eye(4)
        twist = SE3_log(T_id)
        T_recovered = SE3_exp(twist)
        np.testing.assert_array_almost_equal(T_id, T_recovered)
    
    def test_SE3_log_exp_inverse_translation(self):
        """Test exp(log(T)) = T for pure translation."""
        T = np.eye(4)
        T[:3, 3] = [0.5, -0.3, 0.2]
        twist = SE3_log(T)
        T_recovered = SE3_exp(twist)
        np.testing.assert_array_almost_equal(T, T_recovered, decimal=10)
    
    def test_SE3_log_exp_inverse_rotation(self):
        """Test exp(log(T)) = T for pure rotation."""
        T = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        twist = SE3_log(T)
        T_recovered = SE3_exp(twist)
        np.testing.assert_array_almost_equal(T, T_recovered, decimal=10)
    
    def test_SE3_log_exp_inverse_combined(self):
        """Test exp(log(T)) = T for combined transform."""
        T = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
        T[:3, 3] = [0.3, 0.2, 0.1]
        twist = SE3_log(T)
        T_recovered = SE3_exp(twist)
        np.testing.assert_array_almost_equal(T, T_recovered, decimal=10)
    
    def test_SE3_log_exp_inverse_90_deg_rotation(self):
        """Test exp(log(T)) = T for 90 degree rotation."""
        T = np.eye(4)
        T[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
        twist = SE3_log(T)
        T_recovered = SE3_exp(twist)
        np.testing.assert_array_almost_equal(T, T_recovered, decimal=10)
    
    def test_pose_SE3_conversion(self):
        """Test pose to SE(3) and back."""
        pos = np.array([0.3, 0.2, 0.1])
        quat = R.from_euler('xyz', [30, 45, 60], degrees=True).as_quat()
        
        T = pose_to_SE3(pos, quat)
        pos_recovered, quat_recovered = SE3_to_pose(T)
        
        np.testing.assert_array_almost_equal(pos, pos_recovered)
        # Quaternions may differ by sign
        assert np.allclose(quat, quat_recovered) or np.allclose(quat, -quat_recovered)
    
    def test_se3_distance_pure_translation(self):
        """Test SE(3) distance for pure translation."""
        twist = np.array([0, 0, 0, 0.3, 0.4, 0])  # omega=0, v=[0.3, 0.4, 0]
        distance = compute_se3_distance(twist, translation_weight=1.0, rotation_weight=1.0)
        expected = 0.5  # sqrt(0.3^2 + 0.4^2) = 0.5
        assert abs(distance - expected) < 1e-10
    
    def test_se3_distance_pure_rotation(self):
        """Test SE(3) distance for pure rotation."""
        twist = np.array([np.pi/2, 0, 0, 0, 0, 0])  # 90 deg about x
        distance = compute_se3_distance(twist, translation_weight=1.0, rotation_weight=1.0)
        expected = np.pi/2  # rotation angle
        assert abs(distance - expected) < 1e-10


class TestTrajectoryPlanning:
    """Test trajectory planning components."""

    def test_quintic_trajectory_planning(self):
        """Test quintic polynomial trajectory planning."""
        start_pos = np.array([0.0, 0.0, 0.0])
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])
        target_pos = np.array([0.1, 0.2, 0.3])
        target_quat = np.array([0.0, 0.0, 0.0, 1.0])

        num_points = 50
        positions, quaternions = plan_quintic_trajectory(
            start_pos, start_quat, target_pos, target_quat,
            num_points=num_points
        )

        assert isinstance(positions, np.ndarray)
        assert isinstance(quaternions, np.ndarray)
        assert len(positions) == num_points
        assert len(quaternions) == num_points
        assert positions.shape[1] == 3
        assert quaternions.shape[1] == 4

        np.testing.assert_array_almost_equal(positions[0], start_pos, decimal=5)
        np.testing.assert_array_almost_equal(positions[-1], target_pos, decimal=5)
        np.testing.assert_array_almost_equal(quaternions[0], start_quat, decimal=5)
        np.testing.assert_array_almost_equal(quaternions[-1], target_quat, decimal=5)

    def test_trajectory_timing(self):
        """Test that trajectory generates correct number of points."""
        start_pos = np.array([0.0, 0.0, 0.0])
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])
        target_pos = np.array([0.1, 0.0, 0.0])
        target_quat = np.array([0.0, 0.0, 0.0, 1.0])

        num_points = 100
        positions, quaternions = plan_quintic_trajectory(
            start_pos, start_quat, target_pos, target_quat, num_points=num_points
        )

        assert len(positions) == num_points
        assert len(quaternions) == num_points

    def test_zero_displacement_trajectory(self):
        """Test planning with zero displacement."""
        pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])

        positions, quaternions = plan_quintic_trajectory(
            pos, quat, pos, quat
        )

        assert len(positions) == 1
        assert len(quaternions) == 1
        np.testing.assert_array_almost_equal(positions[0], pos)
        np.testing.assert_array_almost_equal(quaternions[0], quat)

    def test_quaternion_interpolation_validity(self):
        """Test that quaternion interpolation produces valid quaternions."""
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])
        target_quat = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])

        positions, quaternions = plan_quintic_trajectory(
            np.array([0.0, 0.0, 0.0]), start_quat,
            np.array([0.1, 0.0, 0.0]), target_quat
        )

        for quat in quaternions:
            norm = np.linalg.norm(quat)
            assert abs(norm - 1.0) < 1e-6

    def test_trajectory_smoothness(self):
        """Test that trajectories are smooth (no sudden jumps)."""
        start_pos = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([0.3, 0.2, 0.1])
        
        num_points = 100
        positions, quaternions = plan_quintic_trajectory(
            start_pos, np.array([0.0, 0.0, 0.0, 1.0]),
            target_pos, np.array([0.0, 0.0, 0.0, 1.0]),
            num_points=num_points
        )

        max_displacement = 0
        for i in range(1, len(positions)):
            displacement = np.linalg.norm(positions[i] - positions[i-1])
            max_displacement = max(max_displacement, displacement)
        
        assert max_displacement < 0.02, f"Max displacement {max_displacement} exceeds limit"


# ═══════════════════════════════════════════════════════════════════════════════
# Extended SE(3) tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSE3Extended:
    """Additional edge-case tests for SE(3) utilities."""

    def test_skew_zeros(self):
        """Skew of zero vector should be zero matrix."""
        S = skew(np.zeros(3))
        np.testing.assert_array_equal(S, np.zeros((3, 3)))

    def test_skew_shape(self):
        """Skew matrix shape should be 3x3."""
        S = skew(np.array([1, 2, 3]))
        assert S.shape == (3, 3)

    def test_se3_to_matrix_shape(self):
        """se(3) matrix should be 4x4."""
        twist = np.zeros(6)
        mat = se3_to_matrix(twist)
        assert mat.shape == (4, 4)

    def test_SE3_exp_identity(self):
        """exp(zero twist) should give identity."""
        T = SE3_exp(np.zeros(6))
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_SE3_log_exp_near_180_degrees(self):
        """Test log/exp near 180-degree rotation (singularity-prone region)."""
        T = np.eye(4)
        T[:3, :3] = R.from_euler('x', 179, degrees=True).as_matrix()
        T[:3, 3] = [0.1, 0.2, 0.3]
        twist = SE3_log(T)
        T_recovered = SE3_exp(twist)
        np.testing.assert_array_almost_equal(T, T_recovered, decimal=6)

    def test_pose_SE3_identity(self):
        """Identity pose should give identity matrix."""
        T = pose_to_SE3(np.zeros(3), np.array([0, 0, 0, 1.0]))
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_se3_distance_symmetry(self):
        """Distance should not depend on sign convention (d(t) >= 0)."""
        twist = np.array([0.1, -0.2, 0.3, -0.1, 0.4, -0.5])
        d = compute_se3_distance(twist)
        assert d >= 0

    def test_se3_distance_zero(self):
        """Zero twist should have zero distance."""
        d = compute_se3_distance(np.zeros(6))
        assert d == 0.0

    def test_se3_distance_weights(self):
        """Changing weights should scale the distance appropriately."""
        twist = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # Pure 1m x-translation
        d1 = compute_se3_distance(twist, translation_weight=1.0, rotation_weight=1.0)
        d2 = compute_se3_distance(twist, translation_weight=2.0, rotation_weight=1.0)
        assert d2 > d1


class TestTrajectoryPlanningExtended:
    """Additional trajectory planning tests."""

    def test_large_rotation(self):
        """Trajectory with large rotation (~90 deg) should work."""
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])
        target_quat = R.from_euler('z', 90, degrees=True).as_quat()
        positions, quaternions = plan_quintic_trajectory(
            np.zeros(3), start_quat,
            np.zeros(3), target_quat,
            num_points=50
        )
        assert len(positions) == 50
        # All quaternions should be unit norm
        norms = np.linalg.norm(quaternions, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_negative_z_motion(self):
        """Trajectory going downward should work."""
        positions, quaternions = plan_quintic_trajectory(
            np.array([0, 0, 1.0]), np.array([0, 0, 0, 1]),
            np.array([0, 0, 0.5]), np.array([0, 0, 0, 1]),
            num_points=30
        )
        # Z should be monotonically decreasing
        z_vals = positions[:, 2]
        diffs = np.diff(z_vals)
        assert np.all(diffs <= 1e-10), "Z should be monotonically decreasing"

    def test_combined_rotation_translation(self):
        """Test trajectory with both rotation and translation."""
        start_pos = np.array([0.0, 0.0, 0.0])
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])
        target_pos = np.array([0.5, 0.3, -0.1])
        target_quat = R.from_euler('xyz', [15, 30, 45], degrees=True).as_quat()

        positions, quaternions = plan_quintic_trajectory(
            start_pos, start_quat, target_pos, target_quat,
            num_points=80
        )

        np.testing.assert_array_almost_equal(positions[0], start_pos, decimal=5)
        np.testing.assert_array_almost_equal(positions[-1], target_pos, decimal=5)
        # Verify smooth (no jumps > reasonable threshold)
        max_jump = max(np.linalg.norm(positions[i] - positions[i - 1])
                       for i in range(1, len(positions)))
        assert max_jump < 0.05