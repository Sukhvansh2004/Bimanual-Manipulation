#!/usr/bin/env python3
"""
Pytest-based test suite for grasp-related functionality.
Tests GraspPredictor transformations, grasp handling, offsets, and service logic.

Note: These tests focus on the transformation and data handling logic,
not on GraspGen model inference (which requires GPU and trained model).
"""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from unittest.mock import patch, MagicMock
import torch

from robosuite_franka_manipulation.transforms import StampedPose, Transform
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.utils import transform_point_cloud
from robosuite_franka_manipulation.schema import GraspSchema
from robosuite_franka_manipulation.perception.grasp_predictor import GraspPredictor
from robosuite_franka_manipulation.services.grasp_service import GraspService

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



class TestGraspSchemaFormat:
    """Test GraspSchema format consistency."""

    def test_grasp_schema_creation(self):
        """Test that GraspSchema has required fields."""
        grasp = GraspSchema(
            pos=[0.5, 0.1, 0.3],
            quat=[0, 0, 0, 1],
            score=0.95,
            id=0,
            frame_id="world"
        )
        
        assert grasp.score == 0.95
        assert grasp.frame_id == "world"
        np.testing.assert_array_almost_equal(grasp.position, [0.5, 0.1, 0.3])

    def test_grasp_schema_serialization(self):
        """Test serializing GraspSchema to dict."""
        grasp = GraspSchema(
            pos=[0.5, 0.1, 0.3],
            quat=[0, 0, 0, 1],
            score=0.95,
            id=0,
            frame_id="world"
        )
        
        d = grasp.model_dump(by_alias=True)
        assert "pos" in d
        assert "quat" in d
        assert "score" in d
        assert d["frame_id"] == "world"


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
        
        transform_matrix = world_to_object.to_matrix()
        point_local = transform_point_cloud(point_world.reshape(1, 3), transform_matrix)[0]
        
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

        pcd_local = transform_point_cloud(pcd, transform_matrix)
        
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
        point_local = transform_point_cloud(pcd, transform_matrix)[0]
        
        # World X should become object-local -Y (inverse of 90 deg rotation)
        np.testing.assert_array_almost_equal(point_local, [0, -1, 0], decimal=6)


class TestGraspPredictorMocked:
    """Test GraspPredictor with mocked GraspGen."""

    @pytest.fixture
    def mock_grasp_predictor(self):
        """Create a mock grasp predictor for testing transformation logic."""
        # We'll test the transformation methods directly without model inference
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
            GraspSchema(pos=[0,0,0], quat=[0,0,0,1], score=0.5, id=1, frame_id="world"),
            GraspSchema(pos=[1,0,0], quat=[0,0,0,1], score=0.9, id=2, frame_id="world"),
            GraspSchema(pos=[0,1,0], quat=[0,0,0,1], score=0.7, id=3, frame_id="world"),
        ]
        
        sorted_grasps = sorted(grasps, key=lambda g: g.score, reverse=True)
        
        assert sorted_grasps[0].score == 0.9
        assert sorted_grasps[1].score == 0.7
        assert sorted_grasps[2].score == 0.5

    def test_best_grasp_selection(self):
        """Test selecting best grasp by score."""
        grasps = [
            GraspSchema(pos=[0,0,0], quat=[0,0,0,1], score=0.5, id=1, frame_id="world"),
            GraspSchema(pos=[1,0,0], quat=[0,0,0,1], score=0.95, id=2, frame_id="world"),
            GraspSchema(pos=[0,1,0], quat=[0,0,0,1], score=0.7, id=3, frame_id="world"),
        ]
        
        best_grasp = max(grasps, key=lambda g: g.score)
        
        assert best_grasp.score == 0.95
        np.testing.assert_array_almost_equal(best_grasp.position, [1, 0, 0])


class TestGraspOffset:
    
    @pytest.fixture
    def predictor(self):
        with patch('robosuite_franka_manipulation.perception.grasp_predictor.GraspGenSampler'), \
             patch('robosuite_franka_manipulation.perception.grasp_predictor.load_grasp_cfg'):
            # Initialize with a dummy gripper type
            predictor = GraspPredictor(gripper_type='test_gripper')
            return predictor

    def test_apply_gripper_offset_rotation_only(self, predictor):
        """Test applying only rotation offset."""
        # Setup offset for test_gripper: 90 deg around Z, no translation
        constants.GRIPPER_GRASP_OFFSETS['test_gripper'] = {
            'position': np.array([0.0, 0.0, 0.0]),
            'quaternion': R.from_euler('z', 90, degrees=True).as_quat()
        }
        
        # Original grasp: identity orientation
        grasp_pose = StampedPose(
            position=[1.0, 2.0, 3.0],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        offset_grasp = predictor._apply_gripper_offset(grasp_pose)
        
        # Position should be unchanged
        np.testing.assert_array_almost_equal(offset_grasp.position, [1.0, 2.0, 3.0])
        
        # Orientation should be rotated 90 deg around Z
        expected_quat = R.from_euler('z', 90, degrees=True).as_quat()
        # Quaternion sign ambiguity check
        dot = np.abs(np.dot(offset_grasp.quaternion, expected_quat))
        assert abs(dot - 1.0) < 1e-6

    def test_apply_gripper_offset_translation_rotated(self, predictor):
        """
        Test that translation offset is correctly rotated by the grasp orientation.
        This verifies the fix: offset should be in grasp frame, not world frame.
        """
        # Setup offset: Move 0.1m along Z (gripper forward direction)
        offset_z = 0.1
        constants.GRIPPER_GRASP_OFFSETS['test_gripper'] = {
            'position': np.array([0.0, 0.0, offset_z]),
            'quaternion': np.array([0, 0, 0, 1]) # Identity rotation
        }
        
        # Case 1: Grasp aligned with world (Identity)
        grasp_pose_identity = StampedPose(
            position=[1.0, 1.0, 1.0],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        offset_grasp_1 = predictor._apply_gripper_offset(grasp_pose_identity)
        # Expected: [1, 1, 1+0.1]
        np.testing.assert_array_almost_equal(offset_grasp_1.position, [1.0, 1.0, 1.1])
        
        # Case 2: Grasp rotated 90 deg around X
        # World Y becomes Grasp Z. 
        # So offset along Grasp Z should be offset along World Y.
        rot_x90 = R.from_euler('x', 90, degrees=True)
        grasp_pose_rotated = StampedPose(
            position=[1.0, 1.0, 1.0],
            quaternion=rot_x90.as_quat(),
            frame_id="world"
        )
        
        offset_grasp_2 = predictor._apply_gripper_offset(grasp_pose_rotated)
        
        # Expected: [1, 1-0.1, 1] because World Y -> Grasp Z is [0, 1, 0] -> [0, 0, 1]?
        # Let's trace:
        # R_x(90) * [0, 0, 1] (offset) 
        # Matrix: [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        # [0, 0, 1] -> [0, -1, 0]
        # So position should change in Y by -0.1
        
        rotated_offset = rot_x90.apply([0, 0, 0.1])
        # [0, -0.1, 0]
        
        expected_pos = np.array([1.0, 1.0, 1.0]) + rotated_offset
        np.testing.assert_array_almost_equal(offset_grasp_2.position, expected_pos)
        
        # Verify it's NOT just adding [0, 0, 0.1] (which would be [1, 1, 1.1])
        assert offset_grasp_2.position[2] == 1.0 
        assert abs(offset_grasp_2.position[1] - 0.9) < 1e-6


# --- Merged content from tests/test_grasp_service_logic.py ---

# Mock constants
@pytest.fixture
def mock_constants():
    # Only patch if not already patched by other fixtures/tests, 
    # but here we scope it to the function use usually.
    # We'll use a context manager inside the test or a fixture.
    # Since this is a merged file, we can define a fixture that applies specifically 
    # to the service logic tests if needed.
    with patch('robosuite_franka_manipulation.constants.GRIPPER_TYPE_MAPPING', 
               {'simple_a': 'gripper_a', 'simple_b': 'gripper_b'}):
        with patch('robosuite_franka_manipulation.constants.GRIPPER_COLLISION_THRESHOLD', 0.005), \
             patch('robosuite_franka_manipulation.constants.GRIPPER_COLLISION_MESH_SAMPLES', 10):
            yield

@pytest.fixture
def mock_grasp_service():
    # Patch dependencies in __init__
    with patch('robosuite_franka_manipulation.services.grasp_service.GraspPredictor'), \
         patch('robosuite_franka_manipulation.services.grasp_service.zmq.Context'), \
         patch('robosuite_franka_manipulation.services.grasp_service.ServiceClient'), \
         patch('robosuite_franka_manipulation.services.grasp_service.TransformManager'):
        service = GraspService(env=None, env_lock=None, tf_manager=None)
        service.grasp_predictors = {} 
        return service

def test_predict_and_filter_sorting_and_collision(mock_grasp_service, mock_constants):
    """
    Test that grasps are collected, sorted by score, and filtered sequentially.
    """
    # Setup mocks
    mock_predictor_a = MagicMock()
    mock_predictor_b = MagicMock()
    
    # Mock collision mesh (dummy)
    mock_mesh = MagicMock()
    mock_predictor_a.get_gripper_collision_mesh.return_value = mock_mesh
    mock_predictor_b.get_gripper_collision_mesh.return_value = mock_mesh
    
    mock_grasp_service.grasp_predictors = {
        'gripper_a': mock_predictor_a,
        'gripper_b': mock_predictor_b
    }
    
    # Define grasps: 
    # A1 (0.9) - Valid
    # B1 (0.8) - Collides with A1
    # A2 (0.7) - Valid (does not collide with A1)
    
    # Note: construct valid StampedPose args
    pose_args = {'position': [0,0,0], 'quaternion': [0,0,0,1], 'frame_id': 'world'}
    
    grasp_a1 = GraspSchema(**pose_args, score=0.9, id=1, gripper_type='gripper_a')
    grasp_b1 = GraspSchema(**pose_args, score=0.8, id=2, gripper_type='gripper_b')
    grasp_a2 = GraspSchema(**pose_args, score=0.7, id=3, gripper_type='gripper_a')
    
    mock_predictor_a.predict_grasps.return_value = [grasp_a1, grasp_a2]
    mock_predictor_b.predict_grasps.return_value = [grasp_b1]
    
    # Mock filter_colliding_grasps and trimesh sampling
    with patch('robosuite_franka_manipulation.services.grasp_service.filter_colliding_grasps') as mock_filter, \
         patch('robosuite_franka_manipulation.services.grasp_service.trimesh.sample.sample_surface') as mock_sample, \
         patch('robosuite_franka_manipulation.services.grasp_service.tra.transform_points') as mock_transform:
        
        # mock_filter return value is [bool] (list of booleans for batch)
        # Sequence:
        # 1. A1: First grasp, occupied is empty -> Code logic: "if occupied_pcd_points: ..." 
        #    so filter_colliding_grasps is NOT called for A1. A1 is auto-added.
        # 2. B1: Checked against A1. We want it to FAIL. Return [False].
        # 3. A2: Checked against A1. We want it to PASS. Return [True].
        
        mock_filter.side_effect = [
            [False], # B1 check
            [True]   # A2 check
        ]
        
        # Mock point sampling for occupied space
        mock_sample.return_value = (np.zeros((10, 3)), None)
        mock_transform.return_value = np.zeros((10, 3))
        
        # Act
        result = mock_grasp_service._predict_and_filter_grasps(
            gripper_types=['simple_a', 'simple_b'], 
            object_pose=StampedPose(**pose_args),
            pcd_np=None,
            mesh_path=None,
            num_grasps=10
        )
        
        # Assert
        # Expected: [A1, A2] because B1 collided
        assert len(result) == 2
        assert result[0].id == 1
        assert result[1].id == 3
        
        # Check sorting happened (A1 > B1 > A2 input order handled by sort)
        # Even though B1 was filtered, the sort order determines check order.
        
        # Verify predict calls
        mock_predictor_a.predict_grasps.assert_called()
        mock_predictor_b.predict_grasps.assert_called()
        
        # Verify filter call count
        assert mock_filter.call_count == 2

def test_intra_gripper_collision(mock_grasp_service, mock_constants):
    """Test collision checking within same gripper type."""
    mock_predictor = MagicMock()
    mock_mesh = MagicMock()
    mock_predictor.get_gripper_collision_mesh.return_value = mock_mesh
    
    mock_grasp_service.grasp_predictors = {'gripper_a': mock_predictor}
    
    pose_args = {'position': [0,0,0], 'quaternion': [0,0,0,1], 'frame_id': 'world'}
    
    # Two grasps same gripper, G1 > G2
    g1 = GraspSchema(**pose_args, score=0.9, id=1, gripper_type='gripper_a')
    g2 = GraspSchema(**pose_args, score=0.8, id=2, gripper_type='gripper_a')
    
    mock_predictor.predict_grasps.return_value = [g2, g1] # Returned unsorted
    
    with patch('robosuite_franka_manipulation.services.grasp_service.filter_colliding_grasps') as mock_filter, \
         patch('robosuite_franka_manipulation.services.grasp_service.trimesh.sample.sample_surface') as mock_sample, \
         patch('robosuite_franka_manipulation.services.grasp_service.tra.transform_points') as mock_transform:
         
         # G1 (0.9) comes first (sorted). Occupied empty -> Auto accepted.
         # G2 (0.8) checked against G1. Let's make it collide -> [False]
         mock_filter.side_effect = [[False]]
         
         mock_sample.return_value = (np.zeros((10, 3)), None)
         mock_transform.return_value = np.zeros((10, 3))
         
         result = mock_grasp_service._predict_and_filter_grasps(
            gripper_types=['simple_a'], 
            object_pose=StampedPose(**pose_args),
            pcd_np=None,
            mesh_path=None,
            num_grasps=10
        )
         
         assert len(result) == 1
         assert result[0].id == 1 # G1 kept