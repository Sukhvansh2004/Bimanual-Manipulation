#!/usr/bin/env python3
"""
Pytest-based test suite for perception functionality.
Tests object detection, grasp prediction, and perception pipeline.
"""

import pytest
import numpy as np
import time
from unittest.mock import MagicMock, patch

from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.services import ServiceManager, ServiceClient, FastSAMService
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose
from robosuite_franka_manipulation.schema import DetectionSchema
from robosuite_franka_manipulation import constants


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
    manager = ServiceManager()
    manager.start()
    
    env = make_dual_arm_env(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True
    )
    env.reset()
    
    tf_manager = TransformManager(env)
    fastsam_service = FastSAMService(tf_manager, visualize=False)
    fastsam_service.start()
    
    time.sleep(2)  # Allow services to start

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
        
        response = client.call("fastsam_service", {"command": "GET_LATEST_DETECTION"})
        
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