#!/usr/bin/env python3
"""
Service Manager - Central orchestrator for distributed services.
"""

import zmq
import threading
import time
import logging
from typing import Dict, Any, Optional, List

from robosuite_franka_manipulation import constants

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceClient:
    """
    Client for interacting with registered services through the service manager.
    """

    def __init__(self, manager_host: str = None, manager_port: int = None):
        self.manager_host = manager_host or constants.DEFAULT_HOST
        self.manager_port = manager_port or constants.SERVICE_MANAGER_PORT
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)

    def _call_manager(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Make a call to the service manager."""
        try:
            self.socket.connect(f"tcp://{self.manager_host}:{self.manager_port}")
            self.socket.send_json(request)

            # Set timeout
            if self.socket.poll(timeout=constants.ZMQ_MANAGER_TIMEOUT):
                response = self.socket.recv_json()
                return response
            else:
                return {'status': 'error', 'message': 'Request timeout'}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}
        finally:
            self.socket.disconnect(f"tcp://{self.manager_host}:{self.manager_port}")

    def register(self, name: str, host: str, port: int,
                service_type: str, description: str = "") -> bool:
        """Register a service with the manager."""
        request = {
            'command': 'REGISTER',
            'name': name,
            'host': host,
            'port': port,
            'type': service_type,
            'description': description
        }
        response = self._call_manager(request)
        return response.get('status') == 'success'

    def lookup(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Look up a service by name."""
        request = {
            'command': 'LOOKUP',
            'name': service_name
        }
        response = self._call_manager(request)
        if response.get('status') == 'success':
            return response.get('service')
        return None

    def call(self, service_name: str, service_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call a service through the manager."""
        request = {
            'command': 'CALL_SERVICE',
            'service_name': service_name,
            'request': service_request
        }
        return self._call_manager(request)


class ServiceManager:
    """
    Simple Service Manager for service registration, lookup, and calling.
    """

    def __init__(self, host: str = None, port: int = None):
        self.host = host or constants.DEFAULT_HOST
        self.port = port or constants.SERVICE_MANAGER_PORT

        # Service registry
        self.services: Dict[str, Dict[str, Any]] = {}

        # State
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        logger.info(f"Service Manager initialized on {host}:{port}")

    def start(self) -> None:
        """Start the service manager."""
        if self.running:
            logger.warning("Service Manager already running")
            return

        logger.info("Starting Service Manager...")
        self.running = True

        # Start main server thread
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()

        logger.info("Service Manager started successfully")

    def stop(self) -> None:
        """Stop the service manager."""
        if not self.running:
            return

        logger.info("Stopping Service Manager...")
        self.running = False

        # Wait for thread to exit first (it will stop when self.running = False)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # Socket is closed in _server_loop finally block, but close context here
        if self.context:
            try:
                self.context.term()
            except Exception:
                pass

        logger.info("Service Manager stopped")

    def _server_loop(self) -> None:
        """Main ZMQ server loop."""
        logger.info(f"Starting ZMQ server on tcp://{self.host}:{self.port}")

        try:
            self.socket.bind(f"tcp://{self.host}:{self.port}")

            while self.running:
                try:
                    if self.socket.poll(timeout=constants.ZMQ_SERVER_RECV_TIMEOUT):
                        message = self.socket.recv_json()
                        response = self._handle_request(message)
                        self.socket.send_json(response)

                except zmq.ZMQError as e:
                    if e.errno == zmq.ETERM or not self.running:
                        break
                    logger.error(f"ZMQ error: {e}")

        except Exception as e:
            if self.running:
                logger.error(f"Server loop error: {e}")
        finally:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass

        logger.info("Server loop stopped")

    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming requests."""
        try:
            command = request.get('command', '').upper()

            if command == 'REGISTER':
                return self._handle_register(request)
            elif command == 'LOOKUP':
                return self._handle_lookup(request)
            elif command == 'CALL_SERVICE':
                return self._handle_call_service(request)
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown command: {command}'
                }

        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _handle_register(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle service registration."""
        try:
            name = request['name']
            host = request['host']
            port = request['port']
            service_type = request['type']
            description = request.get('description', '')

            service_info = {
                'name': name,
                'host': host,
                'port': port,
                'type': service_type,
                'description': description,
                'registered_at': time.time()
            }

            self.services[name] = service_info

            logger.info(f"Registered service: {name} ({service_type}) at {host}:{port}")

            return {
                'status': 'success',
                'message': f'Service {name} registered successfully'
            }

        except KeyError as e:
            return {
                'status': 'error',
                'message': f'Missing required field: {e}'
            }

    def _handle_lookup(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle service lookup."""
        name = request.get('name')
        if not name or name not in self.services:
            return {
                'status': 'error',
                'message': f'Service {name} not found'
            }

        return {
            'status': 'success',
            'service': self.services[name]
        }

    def _handle_call_service(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle direct service calls."""
        service_name = request.get('service_name')
        service_request = request.get('request', {})

        if not service_name or service_name not in self.services:
            return {
                'status': 'error',
                'message': f'Service {service_name} not found'
            }

        service_info = self.services[service_name]

        # Make direct ZMQ call to service
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect(f"tcp://{service_info['host']}:{service_info['port']}")

            socket.send_json(service_request)

            if socket.poll(timeout=constants.ZMQ_SERVICE_CALL_TIMEOUT):
                response = socket.recv_json()
                socket.close()
                context.term()
                return response
            else:
                socket.close()
                context.term()
                return {
                    'status': 'error',
                    'message': f'Service {service_name} timeout'
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Service call failed: {e}'
            }

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
