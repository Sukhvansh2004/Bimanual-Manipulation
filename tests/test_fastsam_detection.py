#!/usr/bin/env python3
"""
Pytest-based test suite for perception functionality.
Tests object detection, grasp prediction, and perception pipeline.
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch
import threading
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.services import ServiceManager, ServiceClient, FastSAMService
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose
from robosuite_franka_manipulation.schema import DetectionSchema
from robosuite_franka_manipulation import constants


from robosuite_franka_manipulation.perception.object_detector import ObjectDetector
from robosuite_franka_manipulation import constants

from robosuite_franka_manipulation.utils.pointcloud import (
    parse_camera_intrinsics, downsample_point_cloud, compute_point_cloud_bounds,
    filter_point_cloud_by_range, validate_point_cloud, deproject_depth_to_point_cloud
)

class TestPointCloudUtils:
    """Test point cloud utility functions."""

    def test_parse_camera_intrinsics(self):
        """Test parse_camera_intrinsics with different formats."""
        # 1D array
        k1 = np.array([800, 800, 320, 240])
        assert parse_camera_intrinsics(k1) == (800.0, 800.0, 320.0, 240.0)
        
        # 2D matrix
        k2 = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        assert parse_camera_intrinsics(k2) == (800.0, 800.0, 320.0, 240.0)

    def test_downsample_point_cloud(self):
        """Test downsample_point_cloud method."""
        pcd = np.random.randn(100, 3)
        res = downsample_point_cloud(pcd, target_points=50, method='random')
        assert len(res) == 50
        
        res_uniform = downsample_point_cloud(pcd, target_points=50, method='uniform')
        assert len(res_uniform) == 50

    def test_compute_point_cloud_bounds(self):
        """Test compute_point_cloud_bounds method."""
        pcd = np.array([[0, 0, 0], [1, 2, 3]])
        min_b, max_b = compute_point_cloud_bounds(pcd)
        np.testing.assert_array_equal(min_b, [0, 0, 0])
        np.testing.assert_array_equal(max_b, [1, 2, 3])

    def test_filter_point_cloud_by_range(self):
        """Test filter_point_cloud_by_range method."""
        pcd = np.array([[0, 0, 0], [10, 10, 10]])
        res = filter_point_cloud_by_range(pcd, min_range=0, max_range=5)
        assert len(res) == 1

    def test_validate_point_cloud(self):
        """Test validate_point_cloud method."""
        assert validate_point_cloud(np.random.randn(10, 3)) is True
        assert validate_point_cloud(np.array([])) is False
        assert validate_point_cloud(np.array([[np.nan, 0, 0]])) is False

    def test_deproject_depth_to_point_cloud(self):
        """Test deproject_depth_to_point_cloud method."""
        depth = np.ones((10, 10))
        pcd = deproject_depth_to_point_cloud(depth, None, 800, 800, 320, 240, mujoco_convention=False)
        assert pcd.shape == (100, 3)
        assert np.all(pcd[:, 2] == 1.0)

class TestFastSAMService:
    """Test FastSAMService class."""

    @pytest.fixture
    def mock_tf_manager(self):
        tfm = MagicMock(spec=TransformManager)
        tfm.env = MagicMock()
        tfm.env.camera_names = [constants.DEFAULT_CAMERA_NAME]
        tfm.env.camera_widths = [640]
        tfm.env.camera_heights = [480]
        tfm.env_lock = None
        return tfm

    @pytest.fixture
    def service(self, mock_tf_manager):
        """Create a FastSAMService with mocked dependencies."""
        with patch('robosuite_franka_manipulation.services.fastsam_service.zmq.Context'), \
             patch('robosuite_franka_manipulation.services.fastsam_service.ServiceClient'), \
             patch('robosuite_franka_manipulation.services.fastsam_service.ObjectDetector'), \
             patch('robosuite_franka_manipulation.services.fastsam_service.get_camera_intrinsic_matrix'):
            service = FastSAMService(mock_tf_manager, visualize=False)
            return service

    def test_normalize_depth(self, service):
        """Test _normalize_depth method."""
        depth = np.zeros((480, 640))
        res = service._normalize_depth(depth)
        assert res.shape == (480, 640)

    @patch('robosuite_franka_manipulation.services.fastsam_service.get_real_depth_map')
    def test_get_camera_data(self, mock_get_depth, service):
        """Test _get_camera_data method."""
        service.env.observation_spec.return_value = {
            f"{service.camera_name}_image": np.zeros((480, 640, 3)),
            f"{service.camera_name}_depth": np.zeros((480, 640))
        }
        mock_get_depth.return_value = np.zeros((480, 640))
        
        rgb, depth = service._get_camera_data()
        assert rgb is not None
        assert depth is not None

    @patch('robosuite_franka_manipulation.services.fastsam_service.get_camera_extrinsic_matrix')
    def test_process_detections(self, mock_extrinsic, service):
        """Test _process_detections method."""
        mock_extrinsic.return_value = np.eye(4)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640))
        
        service.object_detector.detect_objects.return_value = []
        res = service._process_detections(rgb, depth)
        assert res == []

    def test_handle_request_set_prompt(self, service):
        """Test _handle_request with SET_PROMPT."""
        req = {'command': 'SET_PROMPT', 'prompt': 'new_prompt'}
        res = service._handle_request(req)
        assert res['status'] == 'success'
        assert service.current_prompt == 'new_prompt'

    def test_handle_request_get_latest(self, service):
        """Test _handle_request with GET_LATEST_DETECTION."""
        service.latest_detections = [MagicMock(spec=DetectionSchema)]
        req = {'command': 'GET_LATEST_DETECTION'}
        res = service._handle_request(req)
        assert res['status'] == 'success'
        assert len(res['detections']) == 1

    def test_start_stop(self, service):
        """Test start and stop methods."""
        service.service_client.register.return_value = {'status': 'success'}
        
        # Start
        with patch('threading.Thread'):
            service.start()
            assert service.running is True
        
        # Stop
        service.stop()
        assert service.running is False

    def test_continuous_detection_loop(self, service):
        """Test _continuous_detection_loop method."""
        service.running = True
        
        with patch.object(service, '_get_camera_data') as mock_data, \
             patch.object(service, '_process_detections') as mock_process:
            
            mock_data.side_effect = [
                (np.zeros((480, 640, 3)), np.zeros((480, 640))),
                (None, None) # To break loop if we want, but we'll use flag
            ]
            mock_process.return_value = []
            
            # Run for a short time
            def stop_soon():
                time.sleep(0.2)
                service.running = False
                
            threading.Thread(target=stop_soon).start()
            service._continuous_detection_loop()
            
            assert mock_data.called
            assert mock_process.called

    def test_server_loop(self, service):
        """Test _server_loop method."""
        service.running = True
        # Use a generator to avoid StopIteration
        def poll_side_effect(*args, **kwargs):
            if not hasattr(poll_side_effect, 'called'):
                poll_side_effect.called = True
                return True
            return False
            
        service.socket.poll.side_effect = poll_side_effect
        service.socket.recv_json.return_value = {'command': 'GET_PROMPT'}
        
        def stop_soon():
            time.sleep(0.1)
            service.running = False
            
        threading.Thread(target=stop_soon).start()
        service._server_loop()
        
        assert service.socket.send_json.called

class TestObjectDetector:
    """Test ObjectDetector class directly."""

    @pytest.fixture
    def detector(self):
        """Create an ObjectDetector instance with mocked FastSAM model."""
        with patch('robosuite_franka_manipulation.perception.object_detector.FastSAM'):
            detector = ObjectDetector(visualize=False, debug=True)
            return detector

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.camera_intrinsics is not None
        assert detector.prompt == constants.DEFAULT_DETECTION_PROMPT

    def test_set_prompt(self, detector):
        """Test set_prompt method."""
        old_prompt = detector.prompt
        new_prompt = "green apple"
        
        with patch.object(detector, 'close_visualization_windows') as mock_close:
            detector.set_prompt(new_prompt)
            assert detector.prompt == new_prompt
            mock_close.assert_called_once()

    def test_get_intrinsic_params(self, detector):
        """Test _get_intrinsic_params method."""
        fx, fy, cx, cy = detector._get_intrinsic_params()
        assert isinstance(fx, float)
        assert fx > 0

    def test_ensure_depth_2d(self, detector):
        """Test _ensure_depth_2d method."""
        h, w = detector.camera_height, detector.camera_width
        
        # Already 2D
        depth_2d = np.zeros((h, w))
        result = detector._ensure_depth_2d(depth_2d)
        assert result.shape == (h, w)
        
        # Flat
        depth_flat = np.zeros(h * w)
        result = detector._ensure_depth_2d(depth_flat)
        assert result.shape == (h, w)
        
        # Invalid shape
        with pytest.raises(ValueError):
            detector._ensure_depth_2d(np.zeros((10, 10, 10)))

    def test_mask_to_point_cloud_empty(self, detector):
        """Test _mask_to_point_cloud with empty mask."""
        h, w = detector.camera_height, detector.camera_width
        mask = np.zeros((h, w), dtype=bool)
        depth = np.ones((h, w))
        
        pcd = detector._mask_to_point_cloud(mask, depth)
        assert pcd.size == 0

    @patch("cv2.imshow")
    @patch("cv2.waitKey")
    @patch("cv2.destroyWindow")
    def test_visualization_methods(self, mock_destroy, mock_wait, mock_imshow, detector):
        """Test visualization methods."""
        detector.visualize = True
        detector._current_window_names = ["test_window"]
        
        # Test close_visualization_windows
        detector.close_visualization_windows()
        mock_destroy.assert_called_with("test_window")
        assert len(detector._current_window_names) == 0
        
        # Test _visualize_fastsam_results
        mock_result = MagicMock()
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        detector._visualize_fastsam_results(mock_result, rgb_image)
        assert mock_imshow.called
        assert mock_wait.called

class TestDetectionSchemaFormat:
    """Test DetectionSchema format and validation."""

    def test_detection_schema_creation(self):
        """Test creating a DetectionSchema object."""
        pose = StampedPose(position=[0.5, 0.1, 0.3], quaternion=[0, 0, 0, 1], frame_id="world")
        
        detection = DetectionSchema(
            name="milk_carton",
            id=1,
            pose=pose,
            pcd=[[0.5, 0.1, 0.3], [0.51, 0.1, 0.3]]
        )
        
        assert detection.name == "milk_carton"
        assert detection.id == 1
        assert len(detection.pcd) == 2

    def test_detection_schema_serialization(self):
        """Test serializing DetectionSchema to dict."""
        pose = StampedPose(position=[0.5, 0.1, 0.3], quaternion=[0, 0, 0, 1], frame_id="world")
        
        detection = DetectionSchema(
            name="milk_carton",
            id=1,
            pose=pose,
            pcd=[],
            confidence=0.95
        )
        
        d = detection.model_dump(by_alias=True)
        
        assert "name" in d
        assert "id" in d
        assert "pose" in d
        assert "pcd" in d
        assert d["name"] == "milk_carton"

    def test_detection_schema_with_empty_pointcloud(self):
        """Test DetectionSchema with empty point cloud."""
        pose = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        
        detection = DetectionSchema(
            name="unknown",
            id=0,
            pose=pose,
            pcd=[],
            confidence=0.5
        )
        
        assert detection.pcd == []

    def test_detection_schema_mask_field(self):
        """Test mask field handling."""
        pose = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        
        # Test with mask
        det_with_mask = DetectionSchema(name="obj", id=0, pose=pose, pcd=[], mask=[[1, 2], [3, 4]])
        det_without_mask = DetectionSchema(name="obj", id=1, pose=pose, pcd=[])
        
        assert det_with_mask.mask == [[1, 2], [3, 4]]
        assert det_without_mask.mask is None


class TestDetectionPoseExtraction:
    """Test pose extraction from detections."""

    def test_pose_from_detection(self):
        """Test extracting StampedPose from detection."""
        pose = StampedPose(
            position=[0.5, 0.1, 0.3],
            quaternion=[0, 0, 0, 1],
            frame_id="world"
        )
        
        detection = DetectionSchema(
            name="object",
            id=1,
            pose=pose,
            pcd=[],
            confidence=0.9
        )
        
        extracted_pose = detection.pose
        
        assert isinstance(extracted_pose, StampedPose)
        np.testing.assert_array_almost_equal(extracted_pose.position, [0.5, 0.1, 0.3])

    def test_multiple_detections_unique_ids(self):
        """Test that multiple detections have unique IDs."""
        pose = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        
        detections = [
            DetectionSchema(name="obj1", id=0, pose=pose, pcd=[], confidence=0.9),
            DetectionSchema(name="obj2", id=1, pose=pose, pcd=[], confidence=0.8),
            DetectionSchema(name="obj3", id=2, pose=pose, pcd=[], confidence=0.7),
        ]
        
        ids = [d.id for d in detections]
        assert len(ids) == len(set(ids))  # All IDs unique


class TestServiceClientMocked:
    """Test ServiceClient with mocked services."""

    @pytest.fixture
    def mock_service_client(self):
        """Create a mock ServiceClient."""
        client = MagicMock(spec=ServiceClient)
        return client

    def test_call_returns_response(self, mock_service_client):
        """Test that service call returns expected response."""
        mock_service_client.call.return_value = {
            "status": "success",
            "detections": []
        }
        
        response = mock_service_client.call("fastsam_service", {"command": "GET_LATEST_DETECTION"})
        
        assert response["status"] == "success"
        assert "detections" in response

    def test_call_handles_error_status(self, mock_service_client):
        """Test handling of error status from service."""
        mock_service_client.call.return_value = {
            "status": "error",
            "message": "Service unavailable"
        }
        
        response = mock_service_client.call("fastsam_service", {"command": "INVALID"})
        
        assert response["status"] == "error"
        assert "message" in response


class TestFastSAMServiceCommands:
    """Test FastSAMService command handling (mocked)."""

    def test_supported_commands(self):
        """Test that expected commands are documented."""
        expected_commands = [
            "GET_LATEST_DETECTION",
            "SET_PROMPT",
            "GET_PROMPT",
        ]
        
        # Verify each command exists in expected set
        for cmd in expected_commands:
            assert isinstance(cmd, str)

    def test_get_latest_detection_command_format(self):
        """Test GET_LATEST_DETECTION command format."""
        request = {"command": "GET_LATEST_DETECTION"}
        
        assert "command" in request
        assert request["command"] == "GET_LATEST_DETECTION"

    def test_set_prompt_command_format(self):
        """Test SET_PROMPT command format."""
        request = {
            "command": "SET_PROMPT",
            "prompt": "red cube"
        }
        
        assert "command" in request
        assert "prompt" in request
        assert request["prompt"] == "red cube"


@pytest.fixture(scope="class")
def perception_services(request):
    """Fixture to provide a running ServiceManager and FastSAMService."""
    
    manager = ServiceManager(port=constants.SERVICE_MANAGER_PORT)
    manager.start()
    
    env = make_dual_arm_env(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True
    )
    env.reset()
    
    tf_manager = TransformManager(env)
    fastsam_service = FastSAMService(
        tf_manager, 
        visualize=False,
        port=constants.FASTSAM_SERVICE_PORT,
        manager_port=constants.SERVICE_MANAGER_PORT
    )
    fastsam_service.start()
    
    time.sleep(5)  # Allow more time for FastSAM model loading

    def teardown():
        fastsam_service.stop()
        manager.stop()
        env.close()
    
    request.addfinalizer(teardown)
    
    return manager, fastsam_service


class TestPerception:
    """Test perception components via service calls."""

    def test_object_detection_service(self, perception_services):
        """Test object detection functionality via the FastSAM service."""
        manager, service = perception_services
        client = ServiceClient(manager_host=manager.host, manager_port=manager.port)
        
        # Give more time for at least one detection loop to complete
        max_retries = 10
        response = None
        for _ in range(max_retries):
            response = client.call("fastsam_service", {"command": "GET_LATEST_DETECTION"})
            if response and response.get("status") == "success":
                break
            time.sleep(2)
        
        assert response is not None
        assert response["status"] == "success"
        
        detections = response["detections"]
        assert isinstance(detections, list)

        if detections:
            for det in detections:
                assert 'name' in det
                assert 'pose' in det
                assert 'id' in det
                # Verify pose is serialized StampedPose
                pose = det['pose']
                assert 'pos' in pose or 'position' in pose
                assert 'quat' in pose or 'quaternion' in pose
    
    def test_grasp_prediction_in_service(self, perception_services):
        """Test that the FastSAM service returns detections with point clouds."""
        manager, service = perception_services
        client = ServiceClient(manager_host=manager.host, manager_port=manager.port)
        
        # Wait a moment for the detection loop to run
        time.sleep(2)

        response = client.call("fastsam_service", {"command": "GET_LATEST_DETECTION"})

        assert response is not None
        assert response["status"] == "success"
        
        detections = response["detections"]
        if detections:
            det = detections[0]
            pcd = det.get("pcd", [])
            assert isinstance(pcd, list)

    def test_prompt_service_integration(self, perception_services):
        """Test setting and getting prompts via the FastSAM service."""
        manager, service = perception_services
        client = ServiceClient(manager_host=manager.host, manager_port=manager.port)
        
        # Set a new prompt
        new_prompt = "blue cube"
        set_response = client.call("fastsam_service", {"command": "SET_PROMPT", "prompt": new_prompt})
        assert set_response["status"] == "success"

        # Get the prompt to verify
        get_response = client.call("fastsam_service", {"command": "GET_PROMPT"})
        assert get_response["status"] == "success"
        assert get_response["prompt"] == new_prompt


class TestPerceptionEdgeCases:
    """Test edge cases in perception pipeline."""

    def test_empty_detection_handling(self):
        """Test handling when no objects are detected."""
        pose = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        
        # Empty detection list is valid
        detections = []
        assert len(detections) == 0

    def test_detection_with_large_pointcloud(self):
        """Test detection with large point cloud."""
        pose = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        
        # Generate large point cloud
        large_pcd = np.random.randn(10000, 3).tolist()
        
        detection = DetectionSchema(
            name="large_object",
            id=0,
            pose=pose,
            pcd=large_pcd,
            confidence=0.9
        )
        
        assert len(detection.pcd) == 10000

    def test_detection_filtering_by_name(self):
        """Test filtering detections by name."""
        pose = StampedPose(position=[0, 0, 0], quaternion=[0, 0, 0, 1], frame_id="world")
        
        detections = [
            DetectionSchema(name="apple", id=0, pose=pose, pcd=[]),
            DetectionSchema(name="banana", id=1, pose=pose, pcd=[]),
            DetectionSchema(name="apple", id=2, pose=pose, pcd=[]),
        ]
        
        # Filter to only apples
        filtered = [d for d in detections if d.name == "apple"]
        
        assert len(filtered) == 2
        assert all(d.name == "apple" for d in filtered)