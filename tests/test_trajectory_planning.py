#!/usr/bin/env python3
"""
Pytest-based test suite for trajectory planning functionality.
Tests SE(3) trajectory planning using Lie algebra for proper coupled motion.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation.controllers import (
    RobosuiteArmController,
    # SE(3) utilities
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


class TestSE3EdgeCases:
    """Test edge cases for SE(3) operations."""

    def test_SE3_log_180_rotation(self):
        """Test SE3_log for 180 degree rotation (singularity case)."""
        T = np.eye(4)
        T[:3, :3] = R.from_euler('z', 180, degrees=True).as_matrix()
        
        twist = SE3_log(T)
        T_recovered = SE3_exp(twist)
        
        # Check rotation matrices are equivalent
        np.testing.assert_array_almost_equal(T[:3, :3], T_recovered[:3, :3], decimal=6)

    def test_SE3_exp_small_angle(self):
        """Test SE3_exp for very small rotation angles."""
        # Very small rotation
        small_angle = 1e-8
        twist = np.array([small_angle, 0, 0, 0.1, 0.2, 0.3])
        
        T = SE3_exp(twist)
        
        # Should be close to identity rotation with translation
        assert np.allclose(T[:3, :3], np.eye(3), atol=1e-6)
        # Translation should match v (since rotation is ~0)
        np.testing.assert_array_almost_equal(T[:3, 3], [0.1, 0.2, 0.3], decimal=6)

    def test_skew_zero_vector(self):
        """Test skew of zero vector."""
        v = np.zeros(3)
        S = skew(v)
        np.testing.assert_array_equal(S, np.zeros((3, 3)))

    def test_se3_matrix_zero_twist(self):
        """Test se3_to_matrix with zero twist."""
        twist = np.zeros(6)
        mat = se3_to_matrix(twist)
        np.testing.assert_array_equal(mat, np.zeros((4, 4)))

    def test_SE3_log_identity(self):
        """Test SE3_log of identity matrix gives zero twist."""
        T = np.eye(4)
        twist = SE3_log(T)
        np.testing.assert_array_almost_equal(twist, np.zeros(6))


class TestPoseConversions:
    """Test pose conversion functions."""

    def test_pose_to_SE3_identity(self):
        """Test pose_to_SE3 with identity pose."""
        pos = np.zeros(3)
        quat = np.array([0, 0, 0, 1])  # Identity quaternion
        
        T = pose_to_SE3(pos, quat)
        
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_SE3_to_pose_identity(self):
        """Test SE3_to_pose with identity matrix."""
        T = np.eye(4)
        pos, quat = SE3_to_pose(T)
        
        np.testing.assert_array_almost_equal(pos, np.zeros(3))
        # Identity quaternion can be [0,0,0,1] or [0,0,0,-1]
        assert np.allclose(quat, [0, 0, 0, 1]) or np.allclose(quat, [0, 0, 0, -1])

    def test_pose_roundtrip_arbitrary(self):
        """Test pose -> SE3 -> pose roundtrip with arbitrary pose."""
        pos = np.array([1.5, -2.3, 0.8])
        quat = R.from_euler('xyz', [25, -40, 70], degrees=True).as_quat()
        
        T = pose_to_SE3(pos, quat)
        pos_out, quat_out = SE3_to_pose(T)
        
        np.testing.assert_array_almost_equal(pos, pos_out)
        # Quaternion sign ambiguity
        assert np.allclose(quat, quat_out) or np.allclose(quat, -quat_out)


class TestTrajectoryPlanning:
    """Test trajectory planning components."""

    @pytest.fixture(scope="class")
    def controller(self):
        """Fixture to provide arm controller for trajectory planning tests."""
        # Create a minimal controller for testing
        controller = RobosuiteArmController.__new__(RobosuiteArmController)
        yield controller

    def test_quintic_trajectory_planning(self, controller):
        """Test quintic polynomial trajectory planning."""
        # Define start and end poses
        start_pos = np.array([0.0, 0.0, 0.0])
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        target_pos = np.array([0.1, 0.2, 0.3])
        target_quat = np.array([0.0, 0.0, 0.0, 1.0])

        # Plan trajectory
        # Using specific num_points for testing
        num_points = 50
        positions, quaternions = controller.plan_quintic_trajectory(
            start_pos, start_quat, target_pos, target_quat,
            num_points=num_points
        )

        # Check trajectory structure
        assert isinstance(positions, np.ndarray)
        assert isinstance(quaternions, np.ndarray)
        assert len(positions) == num_points
        assert len(quaternions) == num_points
        assert positions.shape[1] == 3  # 3D positions
        assert quaternions.shape[1] == 4  # 4D quaternions

        # Check that trajectory starts and ends at correct poses
        np.testing.assert_array_almost_equal(positions[0], start_pos, decimal=5)
        np.testing.assert_array_almost_equal(positions[-1], target_pos, decimal=5)
        np.testing.assert_array_almost_equal(quaternions[0], start_quat, decimal=5)
        np.testing.assert_array_almost_equal(quaternions[-1], target_quat, decimal=5)

    def test_trajectory_timing(self, controller):
        """Test that trajectory generates correct number of points."""
        start_pos = np.array([0.0, 0.0, 0.0])
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])
        target_pos = np.array([0.1, 0.0, 0.0])
        target_quat = np.array([0.0, 0.0, 0.0, 1.0])

        num_points = 100
        positions, quaternions = controller.plan_quintic_trajectory(
            start_pos, start_quat, target_pos, target_quat, num_points=num_points
        )

        # Check that number of points matches request
        assert len(positions) == num_points
        assert len(quaternions) == num_points

    def test_zero_displacement_trajectory(self, controller):
        """Test planning with zero displacement."""
        pos = np.array([0.0, 0.0, 0.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])

        positions, quaternions = controller.plan_quintic_trajectory(
            pos, quat, pos, quat
        )

        # Should return minimal trajectory (just the target point)
        assert len(positions) == 1
        assert len(quaternions) == 1
        np.testing.assert_array_almost_equal(positions[0], pos)
        np.testing.assert_array_almost_equal(quaternions[0], quat)

    def test_quaternion_interpolation_validity(self, controller):
        """Test that quaternion interpolation produces valid quaternions."""
        start_quat = np.array([0.0, 0.0, 0.0, 1.0])
        target_quat = np.array([0.0, 0.0, np.sin(np.pi/4), np.cos(np.pi/4)])  # 90 degree rotation

        positions, quaternions = controller.plan_quintic_trajectory(
            np.array([0.0, 0.0, 0.0]), start_quat,
            np.array([0.1, 0.0, 0.0]), target_quat
        )

        # Check that all quaternions are valid (unit length)
        for quat in quaternions:
            norm = np.linalg.norm(quat)
            assert abs(norm - 1.0) < 1e-6

    def test_trajectory_smoothness(self, controller):
        """Test that trajectories are smooth (no sudden jumps)."""
        start_pos = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([0.3, 0.2, 0.1])
        
        num_points = 100
        positions, quaternions = controller.plan_quintic_trajectory(
            start_pos, np.array([0.0, 0.0, 0.0, 1.0]),
            target_pos, np.array([0.0, 0.0, 0.0, 1.0]),
            num_points=num_points
        )

        # Check position continuity
        max_displacement = 0
        for i in range(1, len(positions)):
            displacement = np.linalg.norm(positions[i] - positions[i-1])
            max_displacement = max(max_displacement, displacement)
        
        # Total distance is roughly 0.37m. 
        # Average step is 0.37 / 100 = 0.0037m.
        # Max step shouldn't be too much larger than average.
        # Quintic profile has peak velocity higher than average.
        # 0.02 is a safe upper bound.
        assert max_displacement < 0.02, f"Max displacement {max_displacement} exceeds limit"