
import pytest
import threading
import time
import zmq
from robosuite_franka_manipulation.services.eef_service import EEFService
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.services.services_core import ServiceClient

class TestEEFService:
    """Test cases for the EEFService."""

    def test_get_eef_coordinates(self, service_manager, eef_service_fixture):
        """Test getting end-effector coordinates."""
        # Test direct service call (service manager forwarding has issues)
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{constants.EEF_SERVICE_PORT}")
        socket.send_json({"command": "GET"})
        
        if socket.poll(timeout=5000):
            response = socket.recv_json()
            socket.close()
            context.term()
            
            assert response is not None
            assert response["success"] is True
            assert "eef_coordinates" in response
            
            # The service returns keys in format: robot_{id}_{arm_id}
            # e.g., robot_0_right_hand
            eef_coords = response["eef_coordinates"]
            assert len(eef_coords) > 0, "No EEF coordinates returned"
            
            # Check that we have at least one entry for robot 0
            robot0_keys = [k for k in eef_coords.keys() if k.startswith("robot_0")]
            assert len(robot0_keys) > 0, f"No coordinates found for robot 0. Keys: {list(eef_coords.keys())}"
            
            # Check serialized StampedPose format for the first found arm
            first_key = robot0_keys[0]
            robot0_pose = eef_coords[first_key]
            
            assert "pos" in robot0_pose or "position" in robot0_pose
            assert "quat" in robot0_pose or "quaternion" in robot0_pose
            assert "frame_id" in robot0_pose
        else:
            socket.close()
            context.term()
            assert False, "Timeout waiting for EEF service response"

    def test_service_registration(self, service_manager, eef_service_fixture):
        """Test that EEF service registers correctly."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        eef_info = client.lookup("eef_service")
        assert eef_info is not None
        assert eef_info["port"] == 5559


class TestEEFConcurrency:
    """Test cases for EEFService concurrency/deadlock scenarios."""

    def test_get_eef_coordinates_with_lock(self):
        """
        Test getting end-effector coordinates when the service is initialized with a lock.
        This reproduces the potential deadlock scenario where TransformManager and EEFService
        might compete for a non-reentrant lock.
        """
        env = make_dual_arm_env()
        env.reset()
        
        # Create a non-reentrant lock (standard Lock) to strict test deadlock
        env_lock = threading.Lock()
        
        # Initialize service with the lock
        # Note: We don't need a real ServiceManager for this unit test of the method
        eef_service = EEFService(
            env, 
            manager_host="localhost", 
            manager_port=9999,
            env_lock=env_lock
        )
        
        # We don't need to start the full ZMQ server, just call the method directly
        # because the deadlock happens inside get_eef_coordinates logic.
        
        # Create a thread to call the method, so we can timeout if it deadlocks
        result_container = {}
        
        def call_service():
            print("Calling get_eef_coordinates...")
            res = eef_service.get_eef_coordinates()
            print("Returned from get_eef_coordinates")
            result_container['response'] = res

        t = threading.Thread(target=call_service)
        t.start()
        
        # Wait for result with timeout
        t.join(timeout=5.0)
        
        if t.is_alive():
            # If thread is still alive, it's deadlocked
            pytest.fail("Deadlock detected! get_eef_coordinates hung when using env_lock.")
        
        # Check success
        response = result_container.get('response')
        assert response is not None
        assert response["success"] is True
        
        # Clean up
        env.close()

