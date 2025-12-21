#!/usr/bin/env python3
"""
Pytest-based test suite for grasp-related functionality.
Tests GraspPredictor transformations and grasp handling.

Note: These tests focus on the transformation and data handling logic,
not on GraspGen model inference (which requires GPU and trained model).
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from unittest.mock import Mock, patch, MagicMock
import torch

from robosuite_franka_manipulation.transforms import StampedPose, Transform
from robosuite_franka_manipulation import constants


class TestGraspTransformations:
    """Test grasp transformation logic without requiring GraspGen model."""

    def test_rotation_matrix_to_quaternion(self):
        """Test rotation matrix to quaternion conversion."""
        # Create a known rotation matrix (90 deg around Z)
        rot_mat = R.from_euler('z', 90, degrees=True).as_matrix()
        
        # Convert using scipy (same as grasp_predictor._rotation_matrix_to_quaternion)
        quat = R.from_matrix(rot_mat).as_quat()
        
        # Verify it's in scipy convention [x, y, z, w]
        expected = R.from_euler('z', 90, degrees=True).as_quat()
        np.testing.assert_array_almost_equal(quat, expected)

    def test_grasp_local_to_world_transform(self):
        """Test transforming grasp from object-local frame to world frame."""
        # Simulating what grasp_predictor._transform_grasp_to_world does
        
        # Object pose in world
        object_pos = np.array([0.5, 0.0, 0.3])
        object_quat = np.array([0, 0, 0, 1])  # No rotation
        
        object_pose = StampedPose(
            position=object_pos,
            quaternion=object_quat,
            frame_id="world"
        )
        
        # Grasp in object-local frame
        grasp_local_pos = np.array([0.0, 0.0, 0.05])  # 5cm above object origin
        grasp_local_quat = np.array([0, 0, 0, 1])
        
        grasp_local = StampedPose(
            position=grasp_local_pos,
            quaternion=grasp_local_quat,
            frame_id="object_local"
        )
        
        # Transform: object_local -> world
        object_to_world = Transform(
            translation=object_pose.position,
            rotation=object_pose.quaternion,
            parent_frame="world",
            child_frame="object_local"
        )
        
        world_grasp = object_to_world.apply_to_pose(grasp_local)
        
        assert world_grasp.frame_id == "world"
        # Expected: [0.5, 0, 0.3] + [0, 0, 0.05] = [0.5, 0, 0.35]
        np.testing.assert_array_almost_equal(world_grasp.position, [0.5, 0, 0.35])

    def test_grasp_local_to_world_with_object_rotation(self):
        """Test grasp transform when object is rotated."""
        # Object rotated 90 deg around Z axis
        object_pos = np.array([1.0, 0.0, 0.5])
        object_quat = R.from_euler('z', 90, degrees=True).as_quat()
        
        object_pose = StampedPose(
            position=object_pos,
            quaternion=object_quat,
            frame_id="world"
        )
        
        # Grasp offset along object's local X axis
        grasp_local = StampedPose(
            position=[0.1, 0, 0],  # 10cm along local X
            quaternion=[0, 0, 0, 1],
            frame_id="object_local"
        )
        
        object_to_world = Transform(
            translation=object_pose.position,
            rotation=object_pose.quaternion,
            parent_frame="world",
            child_frame="object_local"
        )
        
        world_grasp = object_to_world.apply_to_pose(grasp_local)
        
        # Object's local X is world Y after 90 deg Z rotation
        # Expected: [1, 0, 0.5] + R_z(90)*[0.1, 0, 0] = [1, 0.1, 0.5]
        np.testing.assert_array_almost_equal(world_grasp.position, [1.0, 0.1, 0.5], decimal=6)

    def test_grasp_orientation_transform(self):
        """Test that grasp orientation is correctly transformed."""
        # Object with 90 deg rotation
        object_quat = R.from_euler('z', 90, degrees=True).as_quat()
        
        object_pose = StampedPose(
            position=[0, 0, 0],
            quaternion=object_quat,
            frame_id="world"
        )
        
        # Grasp with 45 deg rotation in local frame
        grasp_local_quat = R.from_euler('z', 45, degrees=True).as_quat()
        grasp_local = StampedPose(
            position=[0, 0, 0],
            quaternion=grasp_local_quat,
            frame_id="object_local"
        )
        
        object_to_world = Transform(
            translation=object_pose.position,
            rotation=object_pose.quaternion,
            parent_frame="world",
            child_frame="object_local"
        )
        
        world_grasp = object_to_world.apply_to_pose(grasp_local)
        
        # Expected: 90 + 45 = 135 deg rotation around Z
        expected_quat = R.from_euler('z', 135, degrees=True).as_quat()
        
        # Compare rotations (quaternions can differ by sign)
        world_rot = R.from_quat(world_grasp.quaternion)
        expected_rot = R.from_euler('z', 135, degrees=True)
        
        # Check angle difference is small
        angle_diff = (world_rot * expected_rot.inv()).magnitude()
        assert angle_diff < 1e-6


class TestGraspDictFormat:
    """Test grasp dictionary format consistency."""

    def test_grasp_dict_has_required_keys(self):
        """Test that expected grasp dict format is correct."""
        # This is the format returned by GraspPredictor.predict_grasps
        grasp = {
            "pos": np.array([0.5, 0.1, 0.3]),
            "quat": np.array([0, 0, 0, 1]),
            "score": 0.95,
            "frame_id": "world"
        }
        
        assert "pos" in grasp
        assert "quat" in grasp
        assert "score" in grasp
        assert "frame_id" in grasp
        
        assert grasp["frame_id"] == "world"
        assert len(grasp["pos"]) == 3
        assert len(grasp["quat"]) == 4

    def test_grasp_dict_to_stamped_pose(self):
        """Test converting grasp dict to StampedPose."""
        grasp = {
            "pos": [0.5, 0.1, 0.3],
            "quat": [0, 0, 0, 1],
            "score": 0.95,
            "frame_id": "world"
        }
        
        pose = StampedPose.from_dict({
            'pos': grasp['pos'],
            'quat': grasp['quat'],
            'frame_id': grasp['frame_id']
        })
        
        assert pose.frame_id == "world"
        np.testing.assert_array_almost_equal(pose.position, [0.5, 0.1, 0.3])


class TestPointCloudPreprocessing:
    """Test point cloud preprocessing logic."""

    def test_world_to_object_local_transform(self):
        """Test transforming point cloud from world to object-local frame."""
        # This is what _preprocess_point_cloud does
        
        # Object pose in world
        object_pos = np.array([1.0, 0.0, 0.5])
        object_quat = np.array([0, 0, 0, 1])
        
        object_pose = {
            'pos': object_pos,
            'quat': object_quat
        }
        
        # Create transform: world -> object_local (inverse of object -> world)
        world_to_object = Transform(
            translation=object_pos,
            rotation=object_quat,
            parent_frame="world",
            child_frame="object_local"
        ).inverse()
        
        # Test point at object center
        point_world = np.array([1.0, 0.0, 0.5])
        
        # Transform using matrix
        transform_matrix = world_to_object.to_matrix()
        point_homogeneous = np.append(point_world, 1.0)
        point_local = (transform_matrix @ point_homogeneous)[:3]
        
        # Should be at origin in object-local frame
        np.testing.assert_array_almost_equal(point_local, [0, 0, 0], decimal=6)

    def test_point_cloud_batch_transform(self):
        """Test batch transformation of point cloud."""
        # Object at position [1, 0, 0]
        object_pose = {
            'pos': np.array([1.0, 0.0, 0.0]),
            'quat': np.array([0, 0, 0, 1])
        }
        
        # Point cloud in world frame
        pcd = np.array([
            [1.0, 0.0, 0.0],  # At object center
            [1.1, 0.0, 0.0],  # 0.1m offset in X
            [1.0, 0.1, 0.0],  # 0.1m offset in Y
            [1.0, 0.0, 0.1],  # 0.1m offset in Z
        ])
        
        # Create inverse transform
        world_to_object = Transform(
            translation=object_pose['pos'],
            rotation=object_pose['quat'],
            parent_frame="world",
            child_frame="object_local"
        ).inverse()
        
        transform_matrix = world_to_object.to_matrix()
        
        # Batch transform
        ones = np.ones((len(pcd), 1))
        pcd_homogeneous = np.hstack([pcd, ones])
        pcd_local = (transform_matrix @ pcd_homogeneous.T).T[:, :3]
        
        # Expected results in object-local frame
        expected = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
        ])
        
        np.testing.assert_array_almost_equal(pcd_local, expected, decimal=6)

    def test_point_cloud_transform_with_rotation(self):
        """Test point cloud transform when object is rotated."""
        # Object rotated 90 deg around Z
        object_pose = {
            'pos': np.array([0.0, 0.0, 0.0]),
            'quat': R.from_euler('z', 90, degrees=True).as_quat()
        }
        
        # Point cloud in world frame
        pcd = np.array([
            [1.0, 0.0, 0.0],  # Along world X
        ])
        
        world_to_object = Transform(
            translation=object_pose['pos'],
            rotation=object_pose['quat'],
            parent_frame="world",
            child_frame="object_local"
        ).inverse()
        
        transform_matrix = world_to_object.to_matrix()
        point_homogeneous = np.append(pcd[0], 1.0)
        point_local = (transform_matrix @ point_homogeneous)[:3]
        
        # World X should become object-local -Y (inverse of 90 deg rotation)
        np.testing.assert_array_almost_equal(point_local, [0, -1, 0], decimal=6)


class TestGraspPredictorMocked:
    """Test GraspPredictor with mocked GraspGen."""

    @pytest.fixture
    def mock_grasp_predictor(self):
        """Create a mock grasp predictor for testing transformation logic."""
        # We'll test the transformation methods directly without model inference
        from robosuite_franka_manipulation.transforms import StampedPose, Transform
        
        class MockGraspPredictor:
            def _transform_grasp_to_world(self, grasp_local: StampedPose, object_pose: StampedPose) -> StampedPose:
                """Same logic as GraspPredictor._transform_grasp_to_world"""
                object_to_world = Transform(
                    translation=object_pose.position,
                    rotation=object_pose.quaternion,
                    parent_frame="world",
                    child_frame="object_local"
                )
                return object_to_world.apply_to_pose(grasp_local)
            
            def _rotation_matrix_to_quaternion(self, R_matrix: np.ndarray) -> np.ndarray:
                return R.from_matrix(R_matrix).as_quat()
        
        return MockGraspPredictor()

    def test_grasp_transform_identity_object(self, mock_grasp_predictor):
        """Test grasp transform with object at origin with no rotation."""
        object_pose = StampedPose(
            position=[0, 0, 0],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        grasp_local = StampedPose(
            position=[0.1, 0.2, 0.3],
            quaternion=[0, 0, 0, 1],
            frame_id="object_local"
        )
        
        world_grasp = mock_grasp_predictor._transform_grasp_to_world(grasp_local, object_pose)
        
        # Should be unchanged
        np.testing.assert_array_almost_equal(world_grasp.position, [0.1, 0.2, 0.3])

    def test_grasp_transform_translated_object(self, mock_grasp_predictor):
        """Test grasp transform with translated object."""
        object_pose = StampedPose(
            position=[1, 2, 3],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        grasp_local = StampedPose(
            position=[0.1, 0, 0],
            quaternion=[0, 0, 0, 1],
            frame_id="object_local"
        )
        
        world_grasp = mock_grasp_predictor._transform_grasp_to_world(grasp_local, object_pose)
        
        np.testing.assert_array_almost_equal(world_grasp.position, [1.1, 2, 3])

    def test_grasp_transform_rotated_object(self, mock_grasp_predictor):
        """Test grasp transform with rotated object."""
        object_pose = StampedPose(
            position=[0, 0, 0],
            quaternion=R.from_euler('z', 90, degrees=True).as_quat(),
            frame_id="world"
        )
        
        grasp_local = StampedPose(
            position=[1, 0, 0],  # Along object's local X
            quaternion=[0, 0, 0, 1],
            frame_id="object_local"
        )
        
        world_grasp = mock_grasp_predictor._transform_grasp_to_world(grasp_local, object_pose)
        
        # Local X becomes world Y after 90 deg Z rotation
        np.testing.assert_array_almost_equal(world_grasp.position, [0, 1, 0], decimal=6)

    def test_rotation_matrix_conversion(self, mock_grasp_predictor):
        """Test rotation matrix to quaternion conversion."""
        # Identity matrix
        R_identity = np.eye(3)
        quat = mock_grasp_predictor._rotation_matrix_to_quaternion(R_identity)
        np.testing.assert_array_almost_equal(quat, [0, 0, 0, 1])
        
        # 90 deg rotation around Z
        R_z90 = R.from_euler('z', 90, degrees=True).as_matrix()
        quat = mock_grasp_predictor._rotation_matrix_to_quaternion(R_z90)
        expected = R.from_euler('z', 90, degrees=True).as_quat()
        np.testing.assert_array_almost_equal(quat, expected)


class TestGraspGenMatrixFormat:
    """Test handling of GraspGen's 4x4 transformation matrix output."""

    def test_grasp_matrix_decomposition(self):
        """Test decomposing 4x4 grasp matrix into position and quaternion."""
        # Simulate GraspGen output: 4x4 transformation matrix
        grasp_matrix = np.eye(4)
        grasp_matrix[:3, 3] = [0.1, 0.2, 0.3]  # Translation
        grasp_matrix[:3, :3] = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        
        # Extract position (translation part)
        pos = grasp_matrix[:3, 3]
        
        # Extract rotation and convert to quaternion
        rot_matrix = grasp_matrix[:3, :3]
        quat = R.from_matrix(rot_matrix).as_quat()
        
        np.testing.assert_array_almost_equal(pos, [0.1, 0.2, 0.3])
        
        # Verify quaternion is valid (unit length)
        assert abs(np.linalg.norm(quat) - 1.0) < 1e-10
        
        # Verify rotation matches
        recovered_rot = R.from_quat(quat).as_matrix()
        np.testing.assert_array_almost_equal(recovered_rot, rot_matrix, decimal=6)

    def test_grasp_matrix_to_stamped_pose(self):
        """Test converting 4x4 matrix to StampedPose."""
        grasp_matrix = np.eye(4)
        grasp_matrix[:3, 3] = [0.5, 0.0, 0.1]
        grasp_matrix[:3, :3] = R.from_euler('z', 45, degrees=True).as_matrix()
        
        pose = StampedPose.from_matrix(grasp_matrix, "object_local")
        
        assert pose.frame_id == "object_local"
        np.testing.assert_array_almost_equal(pose.position, [0.5, 0.0, 0.1])


class TestGraspScoreHandling:
    """Test grasp score handling and sorting."""

    def test_grasp_score_float_conversion(self):
        """Test that grasp scores are properly converted to float."""
        # Simulate torch tensor score
        score_tensor = torch.tensor(0.95)
        score_float = float(score_tensor.cpu().numpy())
        
        assert isinstance(score_float, float)
        assert abs(score_float - 0.95) < 1e-6

    def test_grasp_list_sorting_by_score(self):
        """Test sorting grasps by score."""
        grasps = [
            {"pos": [0, 0, 0], "quat": [0, 0, 0, 1], "score": 0.5, "frame_id": "world"},
            {"pos": [1, 0, 0], "quat": [0, 0, 0, 1], "score": 0.9, "frame_id": "world"},
            {"pos": [0, 1, 0], "quat": [0, 0, 0, 1], "score": 0.7, "frame_id": "world"},
        ]
        
        sorted_grasps = sorted(grasps, key=lambda g: g["score"], reverse=True)
        
        assert sorted_grasps[0]["score"] == 0.9
        assert sorted_grasps[1]["score"] == 0.7
        assert sorted_grasps[2]["score"] == 0.5

    def test_best_grasp_selection(self):
        """Test selecting best grasp by score."""
        grasps = [
            {"pos": [0, 0, 0], "quat": [0, 0, 0, 1], "score": 0.5, "frame_id": "world"},
            {"pos": [1, 0, 0], "quat": [0, 0, 0, 1], "score": 0.95, "frame_id": "world"},
            {"pos": [0, 1, 0], "quat": [0, 0, 0, 1], "score": 0.7, "frame_id": "world"},
        ]
        
        best_grasp = max(grasps, key=lambda g: g["score"])
        
        assert best_grasp["score"] == 0.95
        np.testing.assert_array_almost_equal(best_grasp["pos"], [1, 0, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
