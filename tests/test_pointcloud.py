#!/usr/bin/env python3
"""
Comprehensive tests for point cloud utilities (pointcloud.py).

Covers parsing, deprojection, transformation, PCA pose estimation,
downsampling, filtering, validation, and helper functions.
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation.utils.pointcloud import (
    parse_camera_intrinsics,
    ensure_depth_2d,
    downsample_point_cloud,
    estimate_pose_from_point_cloud,
    transform_point_cloud,
    compute_point_cloud_bounds,
    filter_point_cloud_by_range,
    validate_point_cloud,
)


# ═══════════════════════════════════════════════════════════════════════════════
# parse_camera_intrinsics
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseCameraIntrinsics:
    """Test camera intrinsics parsing."""

    def test_1d_array(self):
        """Standard 1D [fx, fy, cx, cy] format."""
        fx, fy, cx, cy = parse_camera_intrinsics(np.array([500.0, 500.0, 320.0, 240.0]))
        assert fx == 500.0
        assert fy == 500.0
        assert cx == 320.0
        assert cy == 240.0

    def test_2d_3x3_matrix(self):
        """Standard 3x3 intrinsic matrix."""
        K = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        fx, fy, cx, cy = parse_camera_intrinsics(K)
        assert fx == 500.0
        assert fy == 500.0
        assert cx == 320.0
        assert cy == 240.0

    def test_2d_4x4_matrix(self):
        """4x4 intrinsic matrix."""
        K = np.zeros((4, 4))
        K[0, 0] = 600.0
        K[1, 1] = 600.0
        K[0, 2] = 320.0
        K[1, 2] = 240.0
        fx, fy, cx, cy = parse_camera_intrinsics(K)
        assert fx == 600.0
        assert fy == 600.0

    def test_none_raises(self):
        """None intrinsics should raise ValueError."""
        with pytest.raises(ValueError, match="not available"):
            parse_camera_intrinsics(None)

    def test_wrong_1d_length(self):
        """Wrong length 1D array should raise ValueError."""
        with pytest.raises(ValueError, match="invalid length"):
            parse_camera_intrinsics(np.array([500.0, 500.0, 320.0]))

    def test_invalid_2d_shape(self):
        """Too-small 2D matrix should raise ValueError."""
        with pytest.raises(ValueError):
            parse_camera_intrinsics(np.zeros((1, 2)))

    def test_3d_array_raises(self):
        """3D array should raise ValueError."""
        with pytest.raises(ValueError):
            parse_camera_intrinsics(np.zeros((2, 3, 4)))

    def test_list_input_coerced(self):
        """Python list should be coerced to array."""
        fx, fy, cx, cy = parse_camera_intrinsics([500.0, 500.0, 320.0, 240.0])
        assert fx == 500.0


# ═══════════════════════════════════════════════════════════════════════════════
# ensure_depth_2d
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnsureDepth2d:
    """Test depth image normalization."""

    def test_already_2d(self):
        """2D array returned as-is."""
        d = np.ones((480, 640))
        result = ensure_depth_2d(d)
        assert result.shape == (480, 640)

    def test_squeeze_channel_dim(self):
        """(H, W, 1) should squeeze to (H, W)."""
        d = np.ones((480, 640, 1))
        result = ensure_depth_2d(d)
        assert result.shape == (480, 640)

    def test_flat_buffer_with_dims(self):
        """Flat buffer should reshape to (H, W) given height/width."""
        d = np.ones(480 * 640)
        result = ensure_depth_2d(d, height=480, width=640)
        assert result.shape == (480, 640)

    def test_flat_buffer_missing_dims_raises(self):
        """Flat buffer without height/width should raise."""
        with pytest.raises(ValueError, match="height/width"):
            ensure_depth_2d(np.ones(1000))

    def test_flat_buffer_wrong_size(self):
        """Flat buffer with mismatched dimensions should raise."""
        with pytest.raises(ValueError, match="size"):
            ensure_depth_2d(np.ones(1000), height=10, width=200)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            ensure_depth_2d(None)

    def test_3d_non_singleton_raises(self):
        """(H, W, 3) should raise (not a depth image)."""
        with pytest.raises(ValueError, match="invalid shape"):
            ensure_depth_2d(np.ones((480, 640, 3)))


# ═══════════════════════════════════════════════════════════════════════════════
# downsample_point_cloud
# ═══════════════════════════════════════════════════════════════════════════════

class TestDownsamplePointCloud:
    """Test point cloud downsampling."""

    def test_random_downsample(self):
        pcd = np.random.randn(5000, 3)
        result = downsample_point_cloud(pcd, target_points=1024, method='random')
        assert result.shape == (1024, 3)

    def test_uniform_downsample(self):
        pcd = np.random.randn(5000, 3)
        result = downsample_point_cloud(pcd, target_points=1024, method='uniform')
        assert result.shape == (1024, 3)

    def test_no_downsample_if_small(self):
        """Cloud smaller than target should be returned as-is."""
        pcd = np.random.randn(100, 3)
        result = downsample_point_cloud(pcd, target_points=2048)
        assert result.shape == (100, 3)
        np.testing.assert_array_equal(result, pcd)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            downsample_point_cloud(np.random.randn(100, 3), 50, method='voxel')

    def test_random_no_replacement(self):
        """Random downsampling should use unique indices."""
        pcd = np.arange(3000).reshape(1000, 3).astype(float)
        result = downsample_point_cloud(pcd, 500, method='random')
        # All points should be unique (since input has unique points)
        _, counts = np.unique(result, axis=0, return_counts=True)
        assert np.all(counts == 1)


# ═══════════════════════════════════════════════════════════════════════════════
# estimate_pose_from_point_cloud
# ═══════════════════════════════════════════════════════════════════════════════

class TestEstimatePoseFromPointCloud:
    """Test PCA-based pose estimation."""

    def test_centroid_estimation(self):
        """Centroid should match mean of points."""
        pcd = np.random.randn(200, 3) + np.array([1.0, 2.0, 3.0])
        centroid, _ = estimate_pose_from_point_cloud(pcd)
        np.testing.assert_allclose(centroid, np.mean(pcd, axis=0), atol=1e-10)

    def test_quaternion_unit_norm(self):
        """Returned quaternion should be unit norm."""
        pcd = np.random.randn(200, 3)
        _, quat = estimate_pose_from_point_cloud(pcd)
        np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-10)

    def test_rotation_is_valid(self):
        """Quaternion should be a valid rotation (determinant +1)."""
        pcd = np.random.randn(300, 3)
        _, quat = estimate_pose_from_point_cloud(pcd)
        rot = R.from_quat(quat).as_matrix()
        np.testing.assert_allclose(np.linalg.det(rot), 1.0, atol=1e-10)

    def test_empty_cloud(self):
        """Empty cloud should return origin and identity."""
        centroid, quat = estimate_pose_from_point_cloud(np.empty((0, 3)))
        np.testing.assert_array_equal(centroid, [0, 0, 0])
        np.testing.assert_array_equal(quat, [0, 0, 0, 1])

    def test_elongated_object(self):
        """Principal axis should align with elongation direction."""
        # Create an elongated cloud along x-axis
        pcd = np.random.randn(500, 3) * np.array([10.0, 0.1, 0.1])
        centroid, quat = estimate_pose_from_point_cloud(pcd)
        rot = R.from_quat(quat).as_matrix()
        # First principal axis (column 0) should be close to x-axis or -x-axis
        principal = rot[:, 0]
        assert abs(abs(principal[0]) - 1.0) < 0.2, \
            f"Principal axis {principal} should roughly align with x-axis"


# ═══════════════════════════════════════════════════════════════════════════════
# transform_point_cloud
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransformPointCloud:
    """Test point cloud transformation."""

    def test_identity_transform(self):
        pcd = np.random.randn(100, 3)
        T = np.eye(4)
        result = transform_point_cloud(pcd, T)
        np.testing.assert_array_almost_equal(result, pcd)

    def test_pure_translation(self):
        pcd = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        result = transform_point_cloud(pcd, T)
        expected = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotation_90_z(self):
        """90-degree rotation around z-axis."""
        pcd = np.array([[1.0, 0.0, 0.0]])
        T = np.eye(4)
        T[:3, :3] = R.from_euler('z', 90, degrees=True).as_matrix()
        result = transform_point_cloud(pcd, T)
        np.testing.assert_allclose(result, [[0.0, 1.0, 0.0]], atol=1e-10)

    def test_empty_cloud(self):
        """Empty cloud should remain empty."""
        pcd = np.empty((0, 3))
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        result = transform_point_cloud(pcd, T)
        assert result.shape == (0, 3)

    def test_preserves_distances(self):
        """Pure rotation should preserve pairwise distances."""
        pcd = np.random.randn(50, 3)
        T = np.eye(4)
        T[:3, :3] = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        result = transform_point_cloud(pcd, T)
        # Check a pair's distance is preserved
        d_orig = np.linalg.norm(pcd[0] - pcd[1])
        d_new = np.linalg.norm(result[0] - result[1])
        np.testing.assert_allclose(d_orig, d_new, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# compute_point_cloud_bounds
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputePointCloudBounds:
    """Test bounding box computation."""

    def test_known_bounds(self):
        pcd = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        min_b, max_b = compute_point_cloud_bounds(pcd)
        np.testing.assert_array_equal(min_b, [-1, -2, -3])
        np.testing.assert_array_equal(max_b, [1, 2, 3])

    def test_single_point(self):
        pcd = np.array([[5.0, 6.0, 7.0]])
        min_b, max_b = compute_point_cloud_bounds(pcd)
        np.testing.assert_array_equal(min_b, [5, 6, 7])
        np.testing.assert_array_equal(max_b, [5, 6, 7])

    def test_empty_cloud(self):
        min_b, max_b = compute_point_cloud_bounds(np.empty((0, 3)))
        np.testing.assert_array_equal(min_b, [0, 0, 0])
        np.testing.assert_array_equal(max_b, [0, 0, 0])


# ═══════════════════════════════════════════════════════════════════════════════
# filter_point_cloud_by_range
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterPointCloudByRange:
    """Test range-based filtering."""

    def test_filters_far_points(self):
        pcd = np.array([
            [0.0, 0.0, 0.5],   # dist 0.5
            [0.0, 0.0, 5.0],   # dist 5.0
            [0.0, 0.0, 15.0],  # dist 15.0
        ])
        result = filter_point_cloud_by_range(pcd, min_range=0.0, max_range=10.0)
        assert len(result) == 2

    def test_filters_near_points(self):
        pcd = np.array([
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 5.0],
        ])
        result = filter_point_cloud_by_range(pcd, min_range=0.5, max_range=10.0)
        assert len(result) == 2

    def test_empty_cloud_passthrough(self):
        result = filter_point_cloud_by_range(np.empty((0, 3)))
        assert len(result) == 0

    def test_keeps_all_within_range(self):
        pcd = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = filter_point_cloud_by_range(pcd, min_range=0.0, max_range=2.0)
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# validate_point_cloud
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidatePointCloud:
    """Test point cloud validation."""

    def test_valid(self):
        pcd = np.random.randn(100, 3)
        assert validate_point_cloud(pcd) is True

    def test_empty_invalid(self):
        assert validate_point_cloud(np.empty((0, 3))) is False

    def test_wrong_shape(self):
        assert validate_point_cloud(np.ones((10, 4))) is False

    def test_nan_invalid(self):
        pcd = np.ones((10, 3))
        pcd[5, 1] = np.nan
        assert validate_point_cloud(pcd) is False

    def test_inf_invalid(self):
        pcd = np.ones((10, 3))
        pcd[0, 0] = np.inf
        assert validate_point_cloud(pcd) is False

    def test_out_of_range(self):
        pcd = np.ones((10, 3)) * 100.0
        assert validate_point_cloud(pcd, max_range=10.0) is False

    def test_not_array(self):
        assert validate_point_cloud([[1, 2, 3]]) is False

    def test_min_points(self):
        pcd = np.ones((5, 3))
        assert validate_point_cloud(pcd, min_points=10) is False
        assert validate_point_cloud(pcd, min_points=5) is True

    def test_1d_invalid(self):
        assert validate_point_cloud(np.ones(3)) is False
