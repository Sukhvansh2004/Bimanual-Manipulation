#!/usr/bin/env python3
"""
Pytest configuration and fixtures for the test suite.
Handles port isolation to prevent conflicts with running instances.
"""

import pytest
import random
import time
from robosuite_franka_manipulation import constants

# Generate unique port offset for this test session (10000-40000 range)
# This ensures tests don't conflict with production ports (5550-5560) or other test runs
_PORT_OFFSET = random.randint(10000, 40000)


def get_test_port(base_port: int) -> int:
    """Get a test port by applying the session offset to a base port."""
    return base_port + _PORT_OFFSET


@pytest.fixture(scope="session", autouse=True)
def patch_service_ports():
    """
    Patch all service port constants to use isolated test ports.
    This fixture runs automatically for every test session.
    """
    
    # Store original values
    original_manager_port = constants.SERVICE_MANAGER_PORT
    original_fastsam_port = constants.FASTSAM_SERVICE_PORT
    original_grasp_port = constants.GRASP_SERVICE_PORT
    original_move_port = constants.MOVE_TO_POSE_SERVICE_PORT
    original_zmq_timeout = constants.ZMQ_MANAGER_TIMEOUT
    
    # Apply offset to create isolated test ports
    constants.SERVICE_MANAGER_PORT = get_test_port(original_manager_port)
    constants.FASTSAM_SERVICE_PORT = get_test_port(original_fastsam_port)
    constants.GRASP_SERVICE_PORT = get_test_port(original_grasp_port)
    constants.MOVE_TO_POSE_SERVICE_PORT = get_test_port(original_move_port)
    # Use a short timeout for tests (2 seconds instead of 5.5 minutes)
    constants.ZMQ_MANAGER_TIMEOUT = 2000
    
    print(f"\n[Test Config] Using port offset: {_PORT_OFFSET}")
    print(f"[Test Config] SERVICE_MANAGER_PORT: {constants.SERVICE_MANAGER_PORT}")
    print(f"[Test Config] FASTSAM_SERVICE_PORT: {constants.FASTSAM_SERVICE_PORT}")
    print(f"[Test Config] GRASP_SERVICE_PORT: {constants.GRASP_SERVICE_PORT}")
    print(f"[Test Config] MOVE_TO_POSE_SERVICE_PORT: {constants.MOVE_TO_POSE_SERVICE_PORT}")
    print(f"[Test Config] ZMQ_MANAGER_TIMEOUT: {constants.ZMQ_MANAGER_TIMEOUT}ms")
    
    yield
    
    # Restore original values after tests complete
    constants.SERVICE_MANAGER_PORT = original_manager_port
    constants.FASTSAM_SERVICE_PORT = original_fastsam_port
    constants.GRASP_SERVICE_PORT = original_grasp_port
    constants.MOVE_TO_POSE_SERVICE_PORT = original_move_port
    constants.ZMQ_MANAGER_TIMEOUT = original_zmq_timeout


@pytest.fixture(scope="session")
def test_port_offset():
    """Provide the port offset for tests that need to create custom ports."""
    return _PORT_OFFSET


@pytest.fixture
def unique_port(test_port_offset):
    """
    Generate a unique port for tests that need one-off ports.
    Uses random within a range to avoid collisions within the same test run.
    """
    base = random.randint(6000, 9000)
    return base + test_port_offset


# Environment fixtures that prevent rendering during tests
@pytest.fixture(scope="session")
def env_kwargs():
    """Default environment kwargs for tests (no rendering)."""
    return {
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
    }


@pytest.fixture(scope="session")
def env_kwargs_with_camera():
    """Environment kwargs with camera for perception tests."""
    return {
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
    }
