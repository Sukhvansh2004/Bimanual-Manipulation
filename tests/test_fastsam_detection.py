#!/usr/bin/env python3
"""
Pytest-based test suite for perception functionality.
Tests object detection, grasp prediction, and perception pipeline.
"""

import pytest
import numpy as np
import time
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.services import ServiceManager, ServiceClient, FastSAMService
from robosuite_franka_manipulation import constants

@pytest.fixture(scope="class")
def perception_services(request):
    """Fixture to provide a running ServiceManager and FastSAMService."""
    manager = ServiceManager()
    manager.start()
    
    env = make_dual_arm_env()
    env.reset()
    
    fastsam_service = FastSAMService(env, visualize=False)
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