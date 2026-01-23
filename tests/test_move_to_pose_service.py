
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
        self.cancel_called = False
        self.last_target = None
        self.moving = False
        self._stop_event = threading.Event()
        
    def move_to_pose(self, target_pose, gripper_action=0.0, approach=False):
        self.move_called = True
        self.last_target = target_pose
        self.moving = True
        # Simulate movement
        start_time = time.time()
        while time.time() - start_time < 0.5:
            if self._stop_event.is_set():
                self.moving = False
                return False
            time.sleep(0.05)
        self.moving = False
        return True
        
    def cancel_movement(self):
        self.cancel_called = True
        self._stop_event.set()

@pytest.fixture(scope="module")
def service_manager():
    manager = ServiceManager()
    manager.start()
    time.sleep(0.5)
    yield manager
    manager.stop()

@pytest.fixture(scope="module")
def move_service_fixture(service_manager):
    service = MoveToPoseService(manager_host=service_manager.host, manager_port=service_manager.port)
    
    # Register mock controllers
    left_controller = MockController()
    right_controller = MockController()
    
    service.register_controller(0, "right", left_controller)
    service.register_controller(1, "right", right_controller)
    
    service.start()
    time.sleep(1)
    
    yield service, left_controller, right_controller
    
    service.stop()

def test_move_command(service_manager, move_service_fixture):
    service, left_ctrl, right_ctrl = move_service_fixture
    client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
    
    # Reset mock
    left_ctrl.move_called = False
    
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

def test_cancel_command(service_manager, move_service_fixture):
    service, left_ctrl, right_ctrl = move_service_fixture
    client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
    
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
