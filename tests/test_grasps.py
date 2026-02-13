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

from robosuite_franka_manipulation.transforms import StampedPose, Transform, TransformManager
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.utils import transform_point_cloud
from robosuite_franka_manipulation.schema import GraspSchema, GraspRequest
from robosuite_franka_manipulation.perception.grasp_predictor import GraspPredictor, maximize_entropy_grasps
from robosuite_franka_manipulation.services.grasp_service import GraspService

from robosuite_franka_manipulation.planning.grasp_generator import GraspGenerator

class TestGraspGenerator:
    """Test GraspGenerator class."""

    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        return env

    @pytest.fixture
    def mock_service_client(self):
        client = MagicMock()
        return client

    def test_grasp_generator_initialization(self, mock_env, mock_service_client):
        """Test GraspGenerator initialization."""
        generator = GraspGenerator(mock_env, service_client=mock_service_client)
        assert generator.env == mock_env
        assert generator.service_client == mock_service_client

    def test_get_object_pose_stamped(self, mock_env, mock_service_client):
        """Test _get_object_pose_stamped method."""
        generator = GraspGenerator(mock_env, service_client=mock_service_client)
        
        # Success case
        mock_env.get_object_pose.return_value = np.eye(4)
        pose = generator._get_object_pose_stamped("test_obj")
        assert pose is not None
        assert pose.frame_id == constants.WORLD_FRAME
        
        # Failure case
        mock_env.get_object_pose.return_value = None
        pose = generator._get_object_pose_stamped("unknown_obj")
        assert pose is None

    def test_generate_grasps_success(self, mock_env, mock_service_client):
        """Test generate_grasps with successful service response."""
        generator = GraspGenerator(mock_env, service_client=mock_service_client)
        mock_env.get_object_pose.return_value = np.eye(4)
        
        mock_service_client.call.return_value = {
            'status': 'success',
            'gripper_type': ['finger'],
            'grasps': [
                {
                    'pos': [0.1, 0.2, 0.3],
                    'quat': [0, 0, 0, 1],
                    'score': 0.95,
                    'id': '0_0',
                    'frame_id': 'world',
                    'gripper_type': 'franka_panda'
                }
            ]
        }
        
        grasps = generator.generate_grasps("test_obj", "finger")
        assert len(grasps) == 1
        assert grasps[0].score == 0.95
        assert grasps[0].id == '0_0'

    def test_generate_grasps_failure(self, mock_env, mock_service_client):
        """Test generate_grasps with failing service response."""
        generator = GraspGenerator(mock_env, service_client=mock_service_client)
        mock_env.get_object_pose.return_value = np.eye(4)
        
        # Error status
        mock_service_client.call.return_value = {
            'status': 'error',
            'message': 'Failed to generate grasps'
        }
        grasps = generator.generate_grasps("test_obj", "finger")
        assert grasps == []

class TestGraspService:
    """Test GraspService class."""

    @pytest.fixture
    def mock_tf_manager(self):
        tfm = MagicMock(spec=TransformManager)
        return tfm

    @pytest.fixture
    def service(self, mock_tf_manager):
        """Create a GraspService with mocked dependencies."""
        with patch('robosuite_franka_manipulation.services.grasp_service.zmq.Context'), \
             patch('robosuite_franka_manipulation.services.grasp_service.ServiceClient'), \
             patch('robosuite_franka_manipulation.services.grasp_service.GraspPredictor'):
            service = GraspService(mock_tf_manager, visualize=True)
            # Inject mock predictors
            service.grasp_predictors = {'franka_panda': MagicMock()}
            return service

    def test_query_fastsam_service(self, service):
        """Test _query_fastsam_service method."""
        service.service_client.call.return_value = {'status': 'success', 'detections': []}
        res = service._query_fastsam_service()
        assert res['status'] == 'success'

    def test_resolve_input_data_with_name(self, service):
        """Test _resolve_input_data with object_name."""
        req = GraspRequest(command='GET_GRASPS', object_name='test_obj', gripper_type=['finger'])
        with patch('robosuite_franka_manipulation.services.grasp_service.resolve_mesh_path') as mock_resolve:
            mock_sample_pose = MagicMock(spec=StampedPose)
            service.service_client.call.return_value = {'status': 'success', 'detections': [{'pose': mock_sample_pose.model_dump()}]}
            pcd, pose, obj_id, mesh = service._resolve_input_data(req)
            assert obj_id == 'test_obj'

    def test_transform_pose_to_world(self, service):
        """Test _transform_pose_to_world method."""
        pose = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="base")
        service.tf_manager.transform_pose.return_value = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
        res = service._transform_pose_to_world(pose)
        assert res.frame_id == "world"

    def test_update_tf_manager(self, service):
        """Test _update_tf_manager method."""
        pose = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
        service._update_tf_manager("obj1", pose)
        assert service.tf_manager.update_object_frame.called

    def test_clear_visualization(self, service):
        """Test _clear_visualization method."""
        service.tf_manager.get_known_frames.return_value = ["grasp_1", "other"]
        service._clear_visualization()
        service.tf_manager.remove_frame.assert_called_with("grasp_1")

    @patch('robosuite_franka_manipulation.services.grasp_service.filter_colliding_grasps')
    def test_filter_batch_against_obstacles(self, mock_filter, service):
        """Test _filter_batch_against_obstacles method."""
        mock_filter.return_value = [True]
        grasp = MagicMock(spec=GraspSchema)
        grasp.gripper_type = 'franka_panda'
        grasp.position = [0,0,0]
        grasp.quaternion = [0,0,0,1]
        grasp.frame_id = "world"
        grasp.score = 1.0
        
        candidates = [(grasp, np.zeros((10, 3)))]
        obstacle = np.zeros((10, 3))
        
        res = service._filter_batch_against_obstacles(candidates, obstacle)
        assert len(res) == 1

    def test_handle_request_get_grasps(self, service):
        """Test _handle_request with GET_GRASPS."""
        req = {
            'command': 'GET_GRASPS', 
            'object_name': 'test_obj', 
            'gripper_type': ['finger'],
            'object_pose': {'position': [0,0,0], 'quaternion': [0,0,0,1], 'frame_id': 'world'}
        }
        
        with patch.object(service, '_get_grasps') as mock_get:
            mock_get.return_value = {'status': 'success'}
            res = service._handle_request(req)
            assert res['status'] == 'success'

    def test_start_stop(self, service):
        """Test start and stop methods."""
        service.service_client.register.return_value = {'status': 'success'}
        with patch('threading.Thread'):
            service.start()
            assert service.running is True
        service.stop()
        assert service.running is False

    def test_resolve_input_data_via_fastsam(self, service):
        """Test _resolve_input_data by querying FastSAM service."""
        req = GraspRequest(command='GET_GRASPS', gripper_type=['finger'])
        
        # Mock FastSAM response
        mock_pose = {'position': [0,0,0], 'quaternion': [0,0,0,1], 'frame_id': 'world'}
        service.service_client.call.return_value = {
            'status': 'success', 
            'detections': [{'id': '123', 'pcd': [[0,0,0]], 'pose': mock_pose}]
        }
        
        pcd, pose, obj_id, mesh = service._resolve_input_data(req)
        assert obj_id == '123'
        assert pcd is not None
        assert pose is not None

    def test_recursive_filter(self, service):
        """Test _recursive_filter method."""
        grasp1 = MagicMock(spec=GraspSchema)
        grasp1.score = 0.9
        grasp2 = MagicMock(spec=GraspSchema)
        grasp2.score = 0.8
        
        candidates = [
            (grasp1, np.zeros((10, 3))),
            (grasp2, np.zeros((10, 3)))
        ]
        
        with patch.object(service, '_filter_batch_against_obstacles') as mock_filter:
            # Mock filter to return second half as is
            mock_filter.side_effect = lambda c, o: c
            res = service._recursive_filter(candidates)
            assert len(res) == 2
            assert mock_filter.called

class TestGraspPredictor:
    """Test GraspPredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create a GraspPredictor instance with mocked sampler."""
        with patch('robosuite_franka_manipulation.perception.grasp_predictor.GraspGenSampler'), \
             patch('robosuite_franka_manipulation.perception.grasp_predictor.load_grasp_cfg'):
            predictor = GraspPredictor(gripper_type='franka_panda', visualize=True, debug=True)
            return predictor

    def test_get_gripper_collision_mesh_finger(self, predictor):
        """Test getting collision mesh for finger gripper."""
        with patch('robosuite_franka_manipulation.perception.grasp_predictor.get_gripper_info') as mock_info:
            mock_mesh = MagicMock()
            mock_info.return_value.collision_mesh = mock_mesh
            mesh = predictor.get_gripper_collision_mesh()
            assert mesh == mock_mesh

    def test_get_gripper_collision_mesh_suction(self, predictor):
        """Test getting collision mesh for suction gripper."""
        predictor.gripper_type = 'suction'
        with patch('pathlib.Path.exists', return_value=True), \
             patch('trimesh.load') as mock_load:
            mock_mesh = MagicMock()
            mock_load.return_value = mock_mesh
            mesh = predictor.get_gripper_collision_mesh()
            assert mesh == mock_mesh
            assert mock_mesh.apply_transform.called

    def test_predict_grasps_invalid_input(self, predictor):
        """Test predict_grasps with invalid input."""
        with pytest.raises(ValueError):
            predictor.predict_grasps(MagicMock(), num_grasps=10, object_pcd=None, mesh_file=None)

    @patch('robosuite_franka_manipulation.perception.grasp_predictor.load_mesh_and_sample')
    @patch('robosuite_franka_manipulation.perception.grasp_predictor.GraspPredictor._predict_grasps_core')
    def test_predict_grasps_mesh(self, mock_core, mock_sample, predictor):
        """Test predict_grasps using a mesh file."""
        mock_sample.return_value = np.zeros((100, 3))
        mock_core.return_value = []
        
        predictor.predict_grasps(MagicMock(), num_grasps=10, mesh_file="test.stl")
        assert mock_sample.called
        assert mock_core.called

    @patch('robosuite_franka_manipulation.perception.grasp_predictor.GraspGenSampler.run_inference')
    def test_predict_grasps_core(self, mock_inference, predictor):
        """Test _predict_grasps_core method."""
        # Setup mock inference to return one grasp
        mock_grasps = torch.zeros((1, 4, 4))
        mock_grasps[0] = torch.eye(4)
        mock_conf = torch.tensor([0.9])
        mock_inference.return_value = (mock_grasps, mock_conf)
        
        # Mock sampler model device
        predictor.sampler.model.parameters.return_value = iter([MagicMock(device=torch.device('cpu'))])
        
        object_pose = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
        pcd = np.zeros((100, 3))
        
        grasps = predictor._predict_grasps_core(pcd, object_pose, num_grasps=10)
        assert len(grasps) >= 1
        assert grasps[0].gripper_type == 'franka_panda'

    @patch('meshcat.Visualizer')
    def test_visualize_gripper(self, mock_vis, predictor):
        """Test visualize_gripper method."""
        mock_vis_instance = MagicMock()
        mock_vis.return_value = mock_vis_instance
        
        with patch.object(predictor, 'get_gripper_collision_mesh') as mock_mesh:
            mock_mesh_obj = MagicMock()
            mock_mesh_obj.vertices = np.zeros((3, 3))
            mock_mesh_obj.faces = np.zeros((1, 3))
            mock_mesh.return_value = mock_mesh_obj
            
            pose = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
            predictor.visualize_gripper(pose)
            
            assert mock_vis_instance.__getitem__.called

    def test_clear_visualizations(self, predictor):
        """Test clear_visualizations method."""
        mock_vis = MagicMock()
        predictor._meshcat_vis = mock_vis
        predictor.clear_visualizations()
        assert mock_vis.__getitem__.called

    @patch('robosuite_franka_manipulation.perception.grasp_predictor.point_cloud_outlier_removal')
    def test_preprocess_point_cloud(self, mock_outlier, predictor):
        """Test _preprocess_point_cloud method."""
        mock_outlier.return_value = (torch.zeros((50, 3)), None)
        
        pcd = np.zeros((100, 3))
        object_pose = StampedPose(position=[0,0,0], quaternion=[0,0,0,1], frame_id="world")
        
        pcd_local = predictor._preprocess_point_cloud(pcd, object_pose)
        assert pcd_local.shape[1] == 3

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
            frame_id="world",
            gripper_type="franka_panda"
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
            frame_id="world",
            gripper_type="franka_panda"
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
            GraspSchema(pos=[0,0,0], quat=[0,0,0,1], score=0.5, id=1, frame_id="world", gripper_type="franka_panda"),
            GraspSchema(pos=[1,0,0], quat=[0,0,0,1], score=0.9, id=2, frame_id="world", gripper_type="franka_panda"),
            GraspSchema(pos=[0,1,0], quat=[0,0,0,1], score=0.7, id=3, frame_id="world", gripper_type="franka_panda"),
        ]
        
        sorted_grasps = sorted(grasps, key=lambda g: g.score, reverse=True)
        
        assert sorted_grasps[0].score == 0.9
        assert sorted_grasps[1].score == 0.7
        assert sorted_grasps[2].score == 0.5

    def test_best_grasp_selection(self):
        """Test selecting best grasp by score."""
        grasps = [
            GraspSchema(pos=[0,0,0], quat=[0,0,0,1], score=0.5, id=1, frame_id="world", gripper_type="franka_panda"),
            GraspSchema(pos=[1,0,0], quat=[0,0,0,1], score=0.95, id=2, frame_id="world", gripper_type="franka_panda"),
            GraspSchema(pos=[0,1,0], quat=[0,0,0,1], score=0.7, id=3, frame_id="world", gripper_type="franka_panda"),
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
        
        offset_grasp = predictor.apply_gripper_offset(grasp_pose)
        
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
        
        offset_grasp_1 = predictor.apply_gripper_offset(grasp_pose_identity)
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
        
        offset_grasp_2 = predictor.apply_gripper_offset(grasp_pose_rotated)
        
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
        service = GraspService(tf_manager=None)
        service.grasp_predictors = {} 
        return service

def test_predict_and_filter_sorting_and_collision(mock_grasp_service, mock_constants):
    """
    Test that grasps are collected, sorted by score, and filtered sequentially.
    Scenario: A1(0.9) valid, A2(0.7) valid. B1(0.8) collides with A1.
    Service processes A first (index 0), then B (index 1).
    A1 -> ID "0_0". A2 -> ID "0_1". B1 -> ID "1_0".
    """
    # Setup mocks
    mock_predictor_a = MagicMock()
    mock_predictor_b = MagicMock()
    mock_mesh = MagicMock()
    mock_predictor_a.get_gripper_collision_mesh.return_value = mock_mesh
    mock_predictor_b.get_gripper_collision_mesh.return_value = mock_mesh
    
    mock_grasp_service.grasp_predictors = {
        'gripper_a': mock_predictor_a,
        'gripper_b': mock_predictor_b
    }
    
    pose_args = {'position': [0,0,0], 'quaternion': [0,0,0,1], 'frame_id': 'world'}
    
    # A1, B1, A2
    # Note: IDs provided here will be overwritten by service
    grasp_a1 = GraspSchema(**pose_args, score=0.9, id=1, gripper_type='gripper_a')
    grasp_b1 = GraspSchema(**pose_args, score=0.8, id=2, gripper_type='gripper_b')
    grasp_a2 = GraspSchema(**pose_args, score=0.7, id=3, gripper_type='gripper_a')
    
    # Predictors return lists. Service sorts them by score.
    # A returns [A1, A2] (already sorted or not, service will sort)
    mock_predictor_a.predict_grasps.return_value = [grasp_a1, grasp_a2]
    mock_predictor_b.predict_grasps.return_value = [grasp_b1]
    
    # Mock apply_gripper_offset to return the input grasp (identity)
    # This ensures return value has valid .position/.quaternion for Pydantic
    mock_predictor_a.apply_gripper_offset.side_effect = lambda g: g
    mock_predictor_b.apply_gripper_offset.side_effect = lambda g: g
    
    # Mock sampling to return points based on SCORE (since ID is overwritten)
    # Return 10 points: [score, 0, 0]
    def sample_side_effect(grasp):
        return np.full((10, 3), [grasp.score, 0, 0], dtype=float)
    
    mock_grasp_service._sample_grasp_points = MagicMock(side_effect=sample_side_effect)
    
    # Mock collision filter logic
    # obstacle_points contains points from previously accepted grasps.
    # We detect collision if obstacle contains points with score 0.9 (A1) and we are checking 0.8 (B1).
    def batch_filter_side_effect(candidates, obstacle_points):
        if obstacle_points.size == 0:
            return candidates
            
        # Extract scores from obstacle points (first column)
        obstacle_scores = np.unique(obstacle_points[:, 0])
        
        survivors = []
        for g, pts in candidates:
            is_valid = True
            # Logic: B1 (score 0.8) collides if A1 (score 0.9) is in obstacles
            # Floating point comparison warning: use close approximation or simple check
            if any(np.isclose(obstacle_scores, 0.9)) and np.isclose(g.score, 0.8):
                is_valid = False
            
            if is_valid:
                survivors.append((g, pts))
        return survivors

    mock_grasp_service._filter_batch_against_obstacles = MagicMock(side_effect=batch_filter_side_effect)

    # Act
    # gripper_types order matters: A then B
    result = mock_grasp_service._predict_and_filter_grasps(
        gripper_types=['simple_a', 'simple_b'], 
        object_pose=StampedPose(**pose_args),
        pcd_np=None,
        mesh_path=None,
        num_grasps=10
    )
    
    # Assert
    # A1 (0.9) and A2 (0.7) should be kept. B1 (0.8) rejected.
    assert len(result) == 2
    # Verify IDs are updated correctly "typeIdx_graspIdx"
    # A1 was first in A list -> "0_0"
    # A2 was second in A list -> "0_1"
    assert result[0].id == "0_0"
    assert result[0].score == 0.9
    assert result[1].id == "0_1"
    assert result[1].score == 0.7

def test_intra_gripper_collision(mock_grasp_service, mock_constants):
    """Test collision checking within same gripper type using recursive logic."""
    mock_predictor = MagicMock()
    mock_mesh = MagicMock()
    mock_predictor.get_gripper_collision_mesh.return_value = mock_mesh
    mock_grasp_service.grasp_predictors = {'gripper_a': mock_predictor}
    
    pose_args = {'position': [0,0,0], 'quaternion': [0,0,0,1], 'frame_id': 'world'}
    
    g_high = GraspSchema(**pose_args, score=0.9, id=1, gripper_type='gripper_a')
    g_low = GraspSchema(**pose_args, score=0.8, id=2, gripper_type='gripper_a')
    
    # Predictor returns SORTED [high, low] to ensure ID assignment is deterministic "0_0", "0_1"
    mock_predictor.predict_grasps.return_value = [g_high, g_low]
    
    # Mock apply_gripper_offset
    mock_predictor.apply_gripper_offset.side_effect = lambda g: g
    
    # Mock sampling (return valid points)
    mock_grasp_service._sample_grasp_points = MagicMock(side_effect=lambda g: np.ones((10, 3)))
    
    # Mock batch filter: If obstacles exist, assume collision (drop everything).
    def batch_filter_side_effect(candidates, obstacle_points):
        print(f"DEBUG: Checking {len(candidates)} candidates against {obstacle_points.size} obstacle points")
        if obstacle_points.size > 0:
            return []
        return candidates

    mock_grasp_service._filter_batch_against_obstacles = MagicMock(side_effect=batch_filter_side_effect)
    
    result = mock_grasp_service._predict_and_filter_grasps(
        gripper_types=['simple_a'], 
        object_pose=StampedPose(**pose_args),
        pcd_np=None,
        mesh_path=None,
        num_grasps=10,
        filter_intra_collision=True
    )
    
    # g_high (0.9) should survive. g_low (0.8) should be filtered.
    assert len(result) == 1
    assert result[0].score == 0.9
    assert result[0].id == "0_0"

def test_maximize_entropy_grasps():
    # Create some dummy grasps
    # Grasp 1 and 2 are close
    # Grasp 3 is far
    
    grasps = [
        GraspSchema(
            pos=np.array([0.0, 0.0, 0.0]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            score=1.0,
            id=1,
            gripper_type="franka_panda"
        ),
        GraspSchema(
            pos=np.array([0.01, 0.0, 0.0]), # Very close to 1
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            score=0.9,
            id=2,
            gripper_type="franka_panda"
        ),
        GraspSchema(
            pos=np.array([0.5, 0.5, 0.5]), # Far from 1 and 2
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
            score=0.8,
            id=3,
            gripper_type="franka_panda"
        ),
        GraspSchema(
            pos=np.array([0.0, 0.0, 0.0]),
            quat=np.array([0.707, 0.0, 0.0, 0.707]), # Different orientation from 1
            score=0.7,
            id=4,
            gripper_type="franka_panda"
        )
    ]
    
    # Select 2 grasps
    # It should pick 1 (highest score) and then 3 (furthest)
    selected = maximize_entropy_grasps(grasps, 2)
    
    selected_ids = [g.id for g in selected]
    assert len(selected) == 2
    assert selected[0].id == 1
    assert selected[1].id == 3

    # Select 3 grasps
    # It should pick 1, 3, and then probably 4 (due to orientation difference)
    selected_3 = maximize_entropy_grasps(grasps, 3)
    assert len(selected_3) == 3
    assert set([g.id for g in selected_3]) == {1, 3, 4}



def test_visualization_flag_handling(mock_grasp_service, mock_constants):
    """Test that visualize_gripper is called when flags are enabled."""
    mock_predictor = MagicMock()
    mock_mesh = MagicMock()
    mock_predictor.get_gripper_collision_mesh.return_value = mock_mesh
    mock_grasp_service.grasp_predictors = {'gripper_a': mock_predictor}
    
    # Mock apply_gripper_offset (identity)
    mock_predictor.apply_gripper_offset.side_effect = lambda g: g
    
    # Enable service-level visualization
    mock_grasp_service.visualize = True
    
    # Predictor returns one grasp
    pose_args = {'position': [0,0,0], 'quaternion': [0,0,0,1], 'frame_id': 'world'}
    grasp = GraspSchema(**pose_args, score=0.9, id=1, gripper_type='gripper_a')
    mock_predictor.predict_grasps.return_value = [grasp]
    
    # Mock sampling/filtering to pass everything
    mock_grasp_service._sample_grasp_points = MagicMock(return_value=np.ones((10, 3)))
    mock_grasp_service._filter_batch_against_obstacles = MagicMock(side_effect=lambda c, o: c)
    mock_grasp_service._recursive_filter = MagicMock(side_effect=lambda c: c)
    
    # Call with visualize_gripper_mesh=True
    mock_grasp_service._predict_and_filter_grasps(
        gripper_types=['simple_a'], 
        object_pose=StampedPose(**pose_args),
        pcd_np=None,
        mesh_path=None,
        num_grasps=1,
        visualize_gripper_mesh=True
    )
    
    # Assert visualize_gripper was called
    mock_predictor.visualize_gripper.assert_called()
    
    # Reset and call with False
    mock_predictor.visualize_gripper.reset_mock()
    mock_grasp_service._predict_and_filter_grasps(
        gripper_types=['simple_a'], 
        object_pose=StampedPose(**pose_args),
        pcd_np=None,
        mesh_path=None,
        num_grasps=1,
        visualize_gripper_mesh=False
    )
    
    # Assert NOT called
    mock_predictor.visualize_gripper.assert_not_called()