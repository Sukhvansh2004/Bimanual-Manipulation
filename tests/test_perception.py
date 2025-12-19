#!/usr/bin/env python3
"""
Pytest-based test suite for perception functionality.
Tests object detection, grasp prediction, and perception pipeline.
"""

import pytest
import numpy as np
import time
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.services import ServiceManager, UnifiedPerceptionService, ServiceClient

@pytest.fixture(scope="class")
def perception_services(request):
    """Fixture to provide a running ServiceManager and UnifiedPerceptionService."""
    manager = ServiceManager()
    manager.start()
    
    env = make_dual_arm_env()
    env.reset()
    
    perception_service = UnifiedPerceptionService(env, visualize=False) # Disable visualization for tests
    perception_service.start()
    
    time.sleep(2) # Allow services to start

    def teardown():
        perception_service.stop()
        manager.stop()
        env.close()
    
    request.addfinalizer(teardown)
    
    return manager, perception_service

class TestPerception:
    """Test perception components via service calls."""

    def test_object_detection_service(self, perception_services):
        """Test object detection functionality via the perception service."""
        manager, service = perception_services
        client = ServiceClient(manager_host=manager.host, manager_port=manager.port)
        
        response = client.call("perception_service", {"command": "GET_LATEST_OBJECTS"})
        
        assert response is not None
        assert response["status"] == "success"
        
        objects = response["objects"]
        assert isinstance(objects, list)

        if objects:
            for obj in objects:
                assert 'name' in obj
                assert 'pose' in obj
                assert 'grasps' in obj
                assert 'id' in obj
    
    def test_grasp_prediction_in_service(self, perception_services):
        """Test that the perception service returns grasp predictions."""
        manager, service = perception_services
        client = ServiceClient(manager_host=manager.host, manager_port=manager.port)
        
        # Wait a moment for the perception loop to run
        time.sleep(2)

        response = client.call("perception_service", {"command": "GET_LATEST_OBJECTS"})

        assert response is not None
        assert response["status"] == "success"
        
        objects = response["objects"]
        if objects:
            obj = objects[0]
            grasps = obj.get("grasps", [])
            assert isinstance(grasps, list)
            
            if grasps:
                for grasp in grasps:
                    assert 'pos' in grasp
                    assert 'quat' in grasp
                    assert isinstance(grasp['pos'], list)
                    assert isinstance(grasp['quat'], list)

    def test_prompt_service_integration(self, perception_services):
        """Test setting and getting prompts via the unified service."""
        manager, service = perception_services
        client = ServiceClient(manager_host=manager.host, manager_port=manager.port)
        
        # Set a new prompt
        new_prompt = "blue cube"
        set_response = client.call("perception_service", {"command": "SET_PROMPT", "prompt": new_prompt})
        assert set_response["status"] == "success"

        # Get the prompt to verify
        get_response = client.call("perception_service", {"command": "GET_PROMPT"})
        assert get_response["status"] == "success"
        assert get_response["prompt"] == new_prompt