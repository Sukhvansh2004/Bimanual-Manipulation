#!/usr/bin/env python3
"""
Pytest-based test suite for microservices.
Tests Prompt Service, Grasp Service, and EEF Service functionality.
"""

import pytest
import time
import threading
import numpy as np
import zmq

from robosuite_franka_manipulation.services.service_manager import EnhancedServiceManager, ServiceClient
from robosuite_franka_manipulation.services.eef_service import EEFService
from robosuite_franka_manipulation.services.fastsam_service import FastSAMService
from robosuite_franka_manipulation.services.grasp_service import GraspService
from robosuite_franka_manipulation.environments import make_dual_arm_env

@pytest.fixture(scope="class")
def service_manager(request):
    """Fixture to provide a running EnhancedServiceManager."""
    manager = EnhancedServiceManager()
    manager.start()
    time.sleep(0.5)  # Allow manager to start

    def teardown():
        manager.stop()

    request.addfinalizer(teardown)
    return manager


@pytest.fixture(scope="class")
def eef_service_fixture(request, service_manager):
    """Fixture to provide a running EEFService."""
    env = make_dual_arm_env()
    env.reset()

    eef_service = EEFService(env, manager_host=service_manager.host, manager_port=service_manager.port)
    eef_service.start()
    time.sleep(1)  # Allow service to start and register

    def teardown():
        eef_service.stop()
        env.close()

    request.addfinalizer(teardown)
    return eef_service


@pytest.fixture(scope="class")
def fastsam_service_fixture(request, service_manager):
    """Fixture to provide a running FastSAMService with mock model."""
    # Use a mock model path for testing
    fastsam_service = FastSAMService(
        host='localhost',
        port=5551,
        manager_host=service_manager.host,
        manager_port=service_manager.port,
        model_path='/tmp/mock_fastsam.pt',  # Mock path for testing
        visualization=False,
        max_fps=1
    )
    fastsam_service.start()
    time.sleep(1)  # Allow service to start and register

    def teardown():
        fastsam_service.stop()

    request.addfinalizer(teardown)
    return fastsam_service


@pytest.fixture(scope="class")
def grasp_service_fixture(request, service_manager):
    """Fixture to provide a running GraspService with mock model."""
    grasp_service = GraspService(
        host='localhost',
        port=5552,
        manager_host=service_manager.host,
        manager_port=service_manager.port,
        model_path='/tmp/mock_grasp',  # Mock path for testing
        batch_size=1,
        num_workers=1
    )
    grasp_service.start()
    time.sleep(1)  # Allow service to start and register

    def teardown():
        grasp_service.stop()

    request.addfinalizer(teardown)
    return grasp_service

class TestEnhancedServiceManager:
    """Test cases for the EnhancedServiceManager."""

    def test_service_registration_and_lookup(self, service_manager, eef_service_fixture):
        """Test that services can register and be looked up."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)

        # Check for EEF service
        eef_info = client.lookup("eef_service")
        assert eef_info is not None
        assert eef_info["type"] == "eef"

        # Check for a non-existent service
        non_existent_info = client.lookup("non_existent_service")
        assert non_existent_info is None

    def test_service_status(self, service_manager, eef_service_fixture):
        """Test getting service status."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        status = client.get_service_status()
        assert status["status"] == "success"
        assert len(status["services"]) >= 1
        assert "eef_service" in status["services"]


class TestEEFService:
    """Test cases for the EEFService."""

    def test_get_eef_coordinates(self, service_manager, eef_service_fixture):
        """Test getting end-effector coordinates."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        response = client.call("eef_service", {"command": "GET"})
        assert response is not None
        assert response["success"] is True
        assert "left_eef_pos" in response
        assert "right_eef_pos" in response


class TestFastSAMService:
    """Test cases for the FastSAMService."""

    def test_service_registration(self, service_manager, fastsam_service_fixture):
        """Test that FastSAM service registers correctly."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        fastsam_info = client.lookup("fastsam_service")
        assert fastsam_info is not None
        assert fastsam_info["type"] == "fastsam"

    def test_get_status(self, service_manager, fastsam_service_fixture):
        """Test getting FastSAM service status."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        response = client.call("fastsam_service", {"command": "GET_STATUS"})
        assert response["status"] == "success"
        assert "model_loaded" in response


class TestGraspService:
    """Test cases for the GraspService."""

    def test_service_registration(self, service_manager, grasp_service_fixture):
        """Test that Grasp service registers correctly."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        grasp_info = client.lookup("grasp_service")
        assert grasp_info is not None
        assert grasp_info["type"] == "grasp"

    def test_get_status(self, service_manager, grasp_service_fixture):
        """Test getting Grasp service status."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        response = client.call("grasp_service", {"command": "GET_STATUS"})
        assert response["status"] == "success"


class TestPerceptionPipeline:
    """Test cases for the orchestrated perception pipeline."""

    def test_perception_pipeline_orchestration(self, service_manager, fastsam_service_fixture, grasp_service_fixture):
        """Test the complete perception pipeline orchestration."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)

        # Test pipeline execution (may fail due to mock models, but should not crash)
        result = client.get_perception_pipeline(num_grasps=1)
        assert result is not None
        assert "status" in result
        # Note: Pipeline may fail due to mock models, but structure should be correct
        assert "objects" in response

    def test_set_get_prompt(self, perception_service_fixture):
        """Test setting and getting the prompt."""
        manager, _ = perception_service_fixture
        client = ServiceClient(manager_host=manager.host, manager_port=manager.port)
        
        # Set a new prompt
        new_prompt = "test prompt"
        response = client.call("perception_service", {"command": "SET_PROMPT", "prompt": new_prompt})
        assert response["status"] == "success"

        # Get the prompt to verify
        response = client.call("perception_service", {"command": "GET_PROMPT"})
        assert response["status"] == "success"
        assert response["prompt"] == new_prompt

class TestEEFService:
    """Test cases for the EEFService."""
    def test_get_eef_coordinates(self, manager_and_eef_service):
        """Test getting EEF coordinates."""
        manager, _ = manager_and_eef_service
        client = ServiceClient(manager_host=manager.host, manager_port=manager.port)
        
        response = client.call("eef_service", {"command": "GET"})
        assert response is not None
        assert response["success"] is True
        assert "eef_coordinates" in response
        assert "robot_0" in response["eef_coordinates"]
        assert "robot_1" in response["eef_coordinates"]