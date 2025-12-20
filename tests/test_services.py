#!/usr/bin/env python3
"""
Pytest-based test suite for microservices.
Tests Prompt Service, Grasp Service, and EEF Service functionality.
"""

import pytest
import socket
import json
import time
import threading
import numpy as np
import zmq

from robosuite_franka_manipulation.services.prompt_service import PromptService
from robosuite_franka_manipulation.services.grasp_service import GraspService
from robosuite_franka_manipulation.services.eef_service import EEFService
from robosuite_franka_manipulation.environments import make_dual_arm_env


class TestPromptService:
    """Test cases for Prompt Service (TCP socket on port 5557)."""

    @pytest.fixture(scope="class")
    def prompt_service(self):
        """Fixture to provide a running PromptService instance."""
        service = PromptService(port=5557)
        service_thread = threading.Thread(target=service.run, daemon=True)
        service_thread.start()
        time.sleep(0.1)  # Allow service to start
        yield service
        service.running = False  # Stop the service

    def test_prompt_service_initialization(self):
        """Test that PromptService can be initialized."""
        service = PromptService(port=5557)
        assert service is not None
        assert service.port == 5557
        assert hasattr(service, 'run')
        assert hasattr(service, 'handle_client')

    def test_get_command(self, prompt_service):
        """Test GET command to retrieve current prompts."""
        # Connect to service
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5557))

        # Send GET command
        sock.send(b'GET\n')
        response = sock.recv(1024).decode().strip()
        sock.close()

        # Parse response
        response_data = json.loads(response)
        assert 'status' in response_data
        assert 'prompts' in response_data
        assert response_data['status'] == 'success'
        assert isinstance(response_data['prompts'], list)

    def test_set_command(self, prompt_service):
        """Test SET command to update prompts."""
        # Connect to service
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5557))

        # Send SET command with new prompts
        new_prompts = ['bottle', 'cup', 'box']
        command = f'SET {json.dumps(new_prompts)}\n'
        sock.send(command.encode())

        response = sock.recv(1024).decode().strip()
        sock.close()

        # Parse response
        response_data = json.loads(response)
        assert response_data['status'] == 'success'

        # Verify prompts were set by getting them back
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5557))
        sock.send(b'GET\n')
        response = sock.recv(1024).decode().strip()
        sock.close()

        response_data = json.loads(response)
        assert response_data['prompts'] == new_prompts

    def test_invalid_command(self, prompt_service):
        """Test handling of invalid commands."""
        # Connect to service
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5557))

        # Send invalid command
        sock.send(b'INVALID_COMMAND\n')
        response = sock.recv(1024).decode().strip()
        sock.close()

        # Parse response
        response_data = json.loads(response)
        assert response_data['status'] == 'error'
        assert 'message' in response_data

    def test_multiple_connections(self, prompt_service):
        """Test handling multiple simultaneous connections."""
        def client_worker(client_id):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', 5557))
            sock.send(b'GET\n')
            response = sock.recv(1024).decode().strip()
            sock.close()
            return json.loads(response)

        # Start multiple clients
        threads = []
        results = []

        for i in range(5):
            def worker(client_id=i):
                result = client_worker(client_id)
                results.append(result)

            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Verify all responses were successful
        assert len(results) == 5
        for result in results:
            assert result['status'] == 'success'


class TestGraspService:
    """Test cases for Grasp Service (ZMQ REP on port 5558)."""

    @pytest.fixture(scope="class")
    def grasp_service(self):
        """Fixture to provide a running GraspService instance."""
        service = GraspService(port=5558)
        service_thread = threading.Thread(target=service.run, daemon=True)
        service_thread.start()
        time.sleep(0.1)  # Allow service to start
        yield service
        service.running = False  # Stop the service

    def test_grasp_service_initialization(self):
        """Test that GraspService can be initialized."""
        service = GraspService(port=5558)
        assert service is not None
        assert service.port == 5558
        assert hasattr(service, 'run')
        assert hasattr(service, 'handle_request')

    def test_grasp_prediction_request(self, grasp_service):
        """Test grasp prediction with valid point cloud and pose."""
        # Create ZMQ client
        context = zmq.Context()
        socket_client = context.socket(zmq.REQ)
        socket_client.connect("tcp://localhost:5558")

        # Create test data
        # Simple point cloud (cube)
        points = np.random.rand(100, 3) * 0.1 + [0.5, 0, 0.1]
        pose = {
            'pos': [0.5, 0, 0.1],
            'quat': [0, 0, 0, 1]
        }

        request = {
            'point_cloud': points.tolist(),
            'object_pose': pose,
            'num_grasps': 3
        }

        # Send request
        socket_client.send_json(request)

        # Receive response
        response = socket_client.recv_json()
        context.term()

        # Verify response structure
        assert 'status' in response
        assert response['status'] == 'success'
        assert 'grasps' in response
        assert isinstance(response['grasps'], list)

        if response['grasps']:
            for grasp in response['grasps']:
                assert 'pos' in grasp
                assert 'quat' in grasp
                assert 'score' in grasp
                assert isinstance(grasp['pos'], list)
                assert isinstance(grasp['quat'], list)
                assert isinstance(grasp['score'], (int, float))

    def test_invalid_request_format(self, grasp_service):
        """Test handling of malformed requests."""
        context = zmq.Context()
        socket_client = context.socket(zmq.REQ)
        socket_client.connect("tcp://localhost:5558")

        # Send invalid request (missing required fields)
        invalid_request = {'invalid': 'data'}
        socket_client.send_json(invalid_request)

        response = socket_client.recv_json()
        context.term()

        assert response['status'] == 'error'
        assert 'message' in response

    def test_empty_point_cloud(self, grasp_service):
        """Test handling of empty point cloud."""
        context = zmq.Context()
        socket_client = context.socket(zmq.REQ)
        socket_client.connect("tcp://localhost:5558")

        request = {
            'point_cloud': [],
            'object_pose': {'pos': [0, 0, 0], 'quat': [0, 0, 0, 1]},
            'num_grasps': 1
        }

        socket_client.send_json(request)
        response = socket_client.recv_json()
        context.term()

        # Should handle gracefully (may return empty grasps or error)
        assert 'status' in response
        assert 'grasps' in response or 'message' in response


class TestEEFService:
    """Test cases for EEF Service (TCP socket on port 5559)."""

    @pytest.fixture(scope="class")
    def env_and_service(self):
        """Fixture to provide environment and running EEFService."""
        env = make_dual_arm_env()
        env.reset()

        service = EEFService(env, port=5559)
        service_thread = threading.Thread(target=service.run, daemon=True)
        service_thread.start()
        time.sleep(0.1)  # Allow service to start

        yield env, service

        service.running = False  # Stop the service
        env.close()

    def test_eef_service_initialization(self):
        """Test that EEFService can be initialized."""
        env = make_dual_arm_env()
        service = EEFService(env, port=5559)
        assert service is not None
        assert service.port == 5559
        assert hasattr(service, 'run')
        assert hasattr(service, 'handle_client')
        env.close()

    def test_get_eef_coordinates(self, env_and_service):
        """Test retrieving EEF coordinates for both arms."""
        env, service = env_and_service

        # Connect to service
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5559))

        # Send request for left arm
        sock.send(b'GET left\n')
        response = sock.recv(1024).decode().strip()
        sock.close()

        # Parse response
        response_data = json.loads(response)
        assert response_data['status'] == 'success'
        assert 'eef_position' in response_data
        assert 'eef_quaternion' in response_data
        assert isinstance(response_data['eef_position'], list)
        assert isinstance(response_data['eef_quaternion'], list)
        assert len(response_data['eef_position']) == 3
        assert len(response_data['eef_quaternion']) == 4

        # Test right arm
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5559))
        sock.send(b'GET right\n')
        response = sock.recv(1024).decode().strip()
        sock.close()

        response_data = json.loads(response)
        assert response_data['status'] == 'success'

    def test_invalid_arm_request(self, env_and_service):
        """Test handling of invalid arm requests."""
        env, service = env_and_service

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5559))

        # Send request for invalid arm
        sock.send(b'GET invalid_arm\n')
        response = sock.recv(1024).decode().strip()
        sock.close()

        response_data = json.loads(response)
        assert response_data['status'] == 'error'
        assert 'message' in response_data

    def test_invalid_command(self, env_and_service):
        """Test handling of invalid commands."""
        env, service = env_and_service

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5559))

        # Send invalid command
        sock.send(b'INVALID_COMMAND\n')
        response = sock.recv(1024).decode().strip()
        sock.close()

        response_data = json.loads(response)
        assert response_data['status'] == 'error'
        assert 'message' in response_data


class TestServiceIntegration:
    """Integration tests for all services running together."""

    @pytest.fixture(scope="class")
    def all_services(self):
        """Fixture to provide all services running simultaneously."""
        env = make_dual_arm_env()
        env.reset()

        # Start all services
        prompt_service = PromptService(port=5557)
        grasp_service = GraspService(port=5558)
        eef_service = EEFService(env, port=5559)

        services = [prompt_service, grasp_service, eef_service]
        threads = []

        for service in services:
            thread = threading.Thread(target=service.run, daemon=True)
            thread.start()
            threads.append(thread)

        time.sleep(0.2)  # Allow all services to start

        yield services, env

        # Stop all services
        for service in services:
            service.running = False

        env.close()

    def test_all_services_running(self, all_services):
        """Test that all services can run simultaneously without conflicts."""
        services, env = all_services

        # Test prompt service
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5557))
        sock.send(b'GET\n')
        response = sock.recv(1024).decode().strip()
        sock.close()
        assert json.loads(response)['status'] == 'success'

        # Test EEF service
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('localhost', 5559))
        sock.send(b'GET left\n')
        response = sock.recv(1024).decode().strip()
        sock.close()
        assert json.loads(response)['status'] == 'success'

        # Test grasp service
        context = zmq.Context()
        socket_client = context.socket(zmq.REQ)
        socket_client.connect("tcp://localhost:5558")

        request = {
            'point_cloud': [[0.5, 0, 0.1], [0.5, 0.01, 0.1]],
            'object_pose': {'pos': [0.5, 0, 0.1], 'quat': [0, 0, 0, 1]},
            'num_grasps': 1
        }

        socket_client.send_json(request)
        response = socket_client.recv_json()
        context.term()

        assert 'status' in response
        # May succeed or fail depending on grasp predictor, but should not crash

    def test_service_isolation(self, all_services):
        """Test that services operate independently."""
        services, env = all_services

        # Stop one service and verify others still work
        services[0].running = False  # Stop prompt service
        time.sleep(0.1)

        # Test that EEF service still works
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', 5559))
            sock.send(b'GET left\n')
            response = sock.recv(1024).decode().strip()
            sock.close()
            assert json.loads(response)['status'] == 'success'
        except ConnectionRefusedError:
            pytest.fail("EEF service should still be running")

        # Test that grasp service still works
        context = zmq.Context()
        socket_client = context.socket(zmq.REQ)
        try:
            socket_client.connect("tcp://localhost:5558")
            socket_client.send_json({'test': 'data'})
            response = socket_client.recv_json()
            context.term()
            assert 'status' in response
        except zmq.ZMQError:
            pytest.fail("Grasp service should still be running")