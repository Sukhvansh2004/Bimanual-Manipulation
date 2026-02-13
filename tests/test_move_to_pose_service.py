
import pytest
import time
import threading
import numpy as np
from unittest.mock import MagicMock, ANY

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.services.move_to_pose_service import MoveToPoseService
from robosuite_franka_manipulation.services.services_core import ServiceManager, ServiceClient
from robosuite_franka_manipulation.transforms import StampedPose


class MockController:
    def __init__(self):
        self.move_called = False
        self.follow_trajectory_called = False
        self.cancel_called = False
        self.last_target = None
        self.moving = False
        self.last_status = 1
        self._stop_event = threading.Event()
        
    def move_to_pose(self, target_pose, **kwargs):
        self.move_called = True
        self.last_target = target_pose
        self.moving = True
        self._stop_event.clear()
        # Simulate movement
        start_time = time.time()
        while time.time() - start_time < 0.5:
            if self._stop_event.is_set():
                self.moving = False
                return False
            time.sleep(0.05)
        self.moving = False
        return True

    def follow_trajectory(self, trajectory, **kwargs):
        self.follow_trajectory_called = True
        self.moving = True
        self._stop_event.clear()
        time.sleep(0.2)
        self.moving = False
        return True
        
    def cancel_movement(self):
        self.cancel_called = True
        self._stop_event.set()
    
    def is_moving(self):
        """Return whether the controller is currently moving."""
        return self.moving
    
    def get_eef_pose_in_base(self):
        """Return current EEF pose."""
        return np.array([0.5, 0.0, 0.5]), np.array([0, 0, 0, 1])
    
    def reset(self):
        """Reset mock state for next test."""
        self.move_called = False
        self.follow_trajectory_called = False
        self.cancel_called = False
        self.last_target = None
        self.moving = False
        self._stop_event.clear()


@pytest.fixture(scope="module")
def service_manager():
    # Using constants.SERVICE_MANAGER_PORT which is patched by conftest.py
    manager = ServiceManager(port=constants.SERVICE_MANAGER_PORT)
    manager.start()
    time.sleep(0.5)
    yield manager
    manager.stop()


@pytest.fixture(scope="module")
def move_service_fixture(service_manager):
    service = MoveToPoseService(
        port=constants.MOVE_TO_POSE_SERVICE_PORT,
        manager_host=service_manager.host, 
        manager_port=service_manager.port
    )
    
    # Register mock controllers
    left_controller = MockController()
    right_controller = MockController()
    
    service.register_controller(0, "right", left_controller)
    service.register_controller(1, "right", right_controller)
    
    service.start()
    time.sleep(1)
    
    yield service, left_controller, right_controller
    
    service.stop()


class TestMoveToPoseService:
    """Tests for MoveToPoseService functionality."""

    def test_move_command(self, service_manager, move_service_fixture):
        """Test basic move command."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        # Reset mock
        left_ctrl.reset()
        
        pose = {"position": [0.5, 0.0, 0.5], "quaternion": [0, 0, 0, 1], "frame_id": "world"}
        
        response = client.call("move_to_pose_service", {
            "command": "MOVE_TO_POSE",
            "robot_id": 0,
            "arm": "right",
            "pose": pose,
            "gripper_action": 1.0,
            "approach": True
        })
        
        assert response["status"] == "success"
        
        # Wait for thread to start
        time.sleep(0.2)
        assert left_ctrl.move_called
        # Convert numpy array to list for comparison or use np.allclose
        if isinstance(left_ctrl.last_target.position, np.ndarray):
            assert np.allclose(left_ctrl.last_target.position, [0.5, 0.0, 0.5])
        else:
            assert left_ctrl.last_target.position == [0.5, 0.0, 0.5]

    def test_cancel_command(self, service_manager, move_service_fixture):
        """Test cancel move command."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        # Reset mock
        right_ctrl.reset()
        
        # Start a move
        pose = {"position": [0.6, 0.0, 0.6], "quaternion": [0, 0, 0, 1]}
        client.call("move_to_pose_service", {
            "command": "MOVE_TO_POSE",
            "robot_id": 1,
            "arm": "right",
            "pose": pose
        })
        
        time.sleep(0.1)
        assert right_ctrl.move_called
        assert right_ctrl.moving
        
        # Cancel it
        response = client.call("move_to_pose_service", {
            "command": "CANCEL_MOVE",
            "robot_id": 1,
            "arm": "right"
        })
        
        assert response["status"] == "success"
        assert right_ctrl.cancel_called
        
        # Wait for stop
        time.sleep(0.2)
        assert not right_ctrl.moving


class TestMoveToPoseServiceEdgeCases:
    """Edge case tests for MoveToPoseService."""

    def test_get_arm_status_command(self, service_manager, move_service_fixture):
        """Test GET_ARM_STATUS command."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        response = client.call("move_to_pose_service", {
            "command": "GET_ARM_STATUS",
            "robot_id": 0,
            "arm": "right"
        })
        
        assert response["status"] == "success"
        # status_code: -1 = moving, 1 = success, 0 = failure
        assert "status_code" in response

    def test_invalid_command(self, service_manager, move_service_fixture):
        """Test handling of invalid command."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        response = client.call("move_to_pose_service", {
            "command": "INVALID_COMMAND",
            "robot_id": 0,
            "arm": "right"
        })
        
        assert response["status"] == "error"

    def test_unregistered_controller(self, service_manager, move_service_fixture):
        """Test handling of request for unregistered controller."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        response = client.call("move_to_pose_service", {
            "command": "MOVE_TO_POSE",
            "robot_id": 99,  # Non-existent robot
            "arm": "right",
            "pose": {"position": [0.5, 0, 0.5], "quaternion": [0, 0, 0, 1]}
        })
        
        assert response["status"] == "error"

    def test_move_with_timeout(self, service_manager, move_service_fixture):
        """Test move command with timeout parameter."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        left_ctrl.reset()
        
        pose = {"position": [0.5, 0.1, 0.5], "quaternion": [0, 0, 0, 1], "frame_id": "world"}
        
        response = client.call("move_to_pose_service", {
            "command": "MOVE_TO_POSE",
            "robot_id": 0,
            "arm": "right",
            "pose": pose,
            "timeout": 5.0  # 5 second timeout
        })
        
        assert response["status"] == "success"

    def test_concurrent_moves_different_robots(self, service_manager, move_service_fixture):
        """Test concurrent moves on different robots."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        left_ctrl.reset()
        right_ctrl.reset()
        
        # Start moves on both robots
        pose1 = {"position": [0.5, 0, 0.5], "quaternion": [0, 0, 0, 1]}
        pose2 = {"position": [0.6, 0, 0.6], "quaternion": [0, 0, 0, 1]}
        
        response1 = client.call("move_to_pose_service", {
            "command": "MOVE_TO_POSE",
            "robot_id": 0,
            "arm": "right",
            "pose": pose1
        })
        
        response2 = client.call("move_to_pose_service", {
            "command": "MOVE_TO_POSE",
            "robot_id": 1,
            "arm": "right",
            "pose": pose2
        })
        
        assert response1["status"] == "success"
        assert response2["status"] == "success"
        
        # Both should be called
        time.sleep(0.2)
        assert left_ctrl.move_called
        assert right_ctrl.move_called


    def test_follow_trajectory_command(self, service_manager, move_service_fixture):
        """Test FOLLOW_TRAJECTORY command."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        left_ctrl.reset()
        
        trajectory = [
            {"position": [0.5, 0.0, 0.5], "quaternion": [0, 0, 0, 1], "frame_id": "world"},
            {"position": [0.6, 0.0, 0.6], "quaternion": [0, 0, 0, 1], "frame_id": "world"}
        ]
        
        response = client.call("move_to_pose_service", {
            "command": "FOLLOW_TRAJECTORY",
            "robot_id": 0,
            "arm": "right",
            "trajectory": trajectory
        })
        
        assert response["status"] == "success"
        time.sleep(0.2)
        assert left_ctrl.follow_trajectory_called

    def test_concurrent_move_same_robot(self, service_manager, move_service_fixture):
        """Test concurrent moves on same robot triggers cancellation of previous."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        left_ctrl.reset()
        
        pose1 = {"position": [0.5, 0, 0.5], "quaternion": [0, 0, 0, 1]}
        pose2 = {"position": [0.6, 0, 0.6], "quaternion": [0, 0, 0, 1]}
        
        client.call("move_to_pose_service", {
            "command": "MOVE_TO_POSE",
            "robot_id": 0,
            "arm": "right",
            "pose": pose1
        })
        time.sleep(0.1)
        
        response = client.call("move_to_pose_service", {
            "command": "MOVE_TO_POSE",
            "robot_id": 0,
            "arm": "right",
            "pose": pose2
        })
        
        assert response["status"] == "success"
        assert left_ctrl.cancel_called

class TestMoveToPoseServicePoseFormats:
    """Test various pose input formats."""

    def test_pose_with_dict_format(self, service_manager, move_service_fixture):
        """Test move with dict pose format."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        left_ctrl.reset()
        
        pose = {
            "position": [0.5, 0.1, 0.5],
            "quaternion": [0, 0, 0, 1],
            "frame_id": "world"
        }
        
        response = client.call("move_to_pose_service", {
            "command": "MOVE_TO_POSE",
            "robot_id": 0,
            "arm": "right",
            "pose": pose
        })
        
        assert response["status"] == "success"

    def test_pose_with_alias_format(self, service_manager, move_service_fixture):
        """Test move with aliased pose format (pos/quat)."""
        service, left_ctrl, right_ctrl = move_service_fixture
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        left_ctrl.reset()
        
        pose = {
            "pos": [0.5, 0.2, 0.5],
            "quat": [0, 0, 0, 1],
            "frame_id": "world"
        }
        
        response = client.call("move_to_pose_service", {
            "command": "MOVE_TO_POSE",
            "robot_id": 0,
            "arm": "right",
            "pose": pose
        })
        
        assert response["status"] == "success"
