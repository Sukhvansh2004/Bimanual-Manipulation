#!/usr/bin/env python3
"""
Pytest-based test suite for microservices.
Tests ServiceManager, ServiceClient, and request/response handling.
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
import zmq

from robosuite_franka_manipulation.services.services_core import ServiceManager, ServiceClient
from robosuite_franka_manipulation import constants


class MockService:
    """Mock service for testing ServiceManager."""
    def __init__(self, manager_host, manager_port, service_port=15555):
        self.service_client = ServiceClient(manager_host, manager_port)
        self.service_port = service_port
        self.running = False
        
    def start(self):
        self.running = True
        self.service_client.register("mock_service", "localhost", self.service_port, "mock", "Mock Service")

    def stop(self):
        self.running = False


@pytest.fixture(scope="class")
def service_manager(request):
    """Fixture to provide a running ServiceManager."""
    manager = ServiceManager()
    manager.start()
    time.sleep(0.5)  # Allow manager to start

    def teardown():
        manager.stop()

    request.addfinalizer(teardown)
    return manager


@pytest.fixture(scope="class")
def mock_service_fixture(request, service_manager, test_port_offset):
    """Fixture to provide a running MockService."""
    service_port = 15555 + test_port_offset
    service = MockService(
        manager_host=service_manager.host, 
        manager_port=service_manager.port,
        service_port=service_port
    )
    service.start()
    time.sleep(1)  # Allow service to start and register

    def teardown():
        service.stop()

    request.addfinalizer(teardown)
    return service


class TestServiceManager:
    """Test cases for the ServiceManager."""

    def test_service_registration_and_lookup(self, service_manager, mock_service_fixture):
        """Test that services can register and be looked up."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)

        # Check for Mock service
        info = client.lookup("mock_service")
        assert info is not None
        assert "host" in info
        assert "port" in info
        assert info["port"] == mock_service_fixture.service_port

        # Check for a non-existent service
        non_existent_info = client.lookup("non_existent_service")
        assert non_existent_info is None

    def test_service_status(self, service_manager, mock_service_fixture):
        """Test that services are properly registered."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        info = client.lookup("mock_service")
        assert info is not None

    def test_list_services(self, service_manager, mock_service_fixture):
        """Test that list_services returns all registered services."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        response = client.list_services()
        
        assert response['status'] == 'success'
        assert 'services' in response
        assert 'mock_service' in response['services']
        assert response['services']['mock_service']['name'] == 'mock_service'


class TestServiceManagerEdgeCases:
    """Test edge cases for ServiceManager."""

    def test_manager_double_start(self, test_port_offset):
        """Test that starting manager twice doesn't cause issues."""
        manager = ServiceManager(port=5560 + test_port_offset)
        manager.start()
        time.sleep(0.3)
        
        # Second start should be ignored
        manager.start()
        
        assert manager.running is True
        manager.stop()

    def test_manager_stop_when_not_running(self, test_port_offset):
        """Test stopping manager when not running."""
        manager = ServiceManager(port=5561 + test_port_offset)
        # Should not raise any errors
        manager.stop()

    def test_manager_double_stop(self, test_port_offset):
        """Test stopping manager twice doesn't cause issues."""
        manager = ServiceManager(port=5562 + test_port_offset)
        manager.start()
        time.sleep(0.3)
        
        manager.stop()
        # Second stop should be safe
        manager.stop()


class TestServiceClient:
    """Test cases for ServiceClient."""

    @pytest.fixture
    def client(self, service_manager):
        """Create a service client for testing."""
        return ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)

    def test_client_close(self, service_manager):
        """Test that client can be closed properly."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        client.close()
        # Should not raise any errors

    def test_register_service(self, client, test_port_offset):
        """Test service registration via client."""
        response = client.register(
            name="test_service",
            host="localhost",
            port=5570 + test_port_offset,
            service_type="test",
            description="Test service"
        )
        
        assert response.get('status') == 'success'

    def test_lookup_nonexistent_service(self, client):
        """Test looking up a service that doesn't exist."""
        result = client.lookup("definitely_not_a_service")
        assert result is None


class TestServiceClientErrorHandling:
    """Test error handling in ServiceClient."""

    def test_call_manager_timeout(self):
        """Test client timeout when manager is not available."""
        # Use a port that is definitely not running a manager
        client = ServiceClient(manager_host="localhost", manager_port=59999)
        
        response = client.register("test", "localhost", 5000, "test", "test")
        
        # Should return error status (timeout or connection refused)
        assert response.get('status') == 'error'
        client.close()

    def test_call_service_with_invalid_data(self, service_manager, mock_service_fixture):
        """Test calling a service with invalid data."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        # Call with bad request data
        response = client.call("mock_service", {"invalid": "data"})
        
        # Should handle gracefully (service may return error or not respond)
        assert isinstance(response, dict)


class TestServiceRegistration:
    """Test service registration details."""

    def test_duplicate_service_registration(self, service_manager, test_port_offset):
        """Test registering the same service twice returns error (duplicates rejected)."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        # Register first time
        response1 = client.register("dup_service", "localhost", 5580 + test_port_offset, "test", "First")
        assert response1.get('status') == 'success'
        
        # Register again - should return error (duplicate rejected)
        response2 = client.register("dup_service", "localhost", 5581 + test_port_offset, "test", "Second")
        assert response2.get('status') == 'error'
        assert 'already registered' in response2.get('message', '').lower()
        
        # Lookup should return original info (not updated)
        info = client.lookup("dup_service")
        assert info is not None
        assert info['port'] == 5580 + test_port_offset

    def test_service_registration_with_empty_description(self, service_manager):
        """Test service registration with empty description."""
        client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
        
        response = client.register("nodesc_service", "localhost", 5582, "test", "")
        assert response.get('status') == 'success'


class TestServiceConcurrency:
    """Test concurrent access to services."""

    def test_concurrent_lookups(self, service_manager, mock_service_fixture):
        """Test multiple concurrent lookups."""
        results = []
        errors = []
        
        def lookup_task():
            try:
                client = ServiceClient(manager_host=service_manager.host, manager_port=service_manager.port)
                info = client.lookup("mock_service")
                results.append(info)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=lookup_task) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        
        assert len(errors) == 0
        assert len(results) == 5
        for info in results:
            assert info is not None
