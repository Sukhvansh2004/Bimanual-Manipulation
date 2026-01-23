#!/usr/bin/env python3
"""
Service Manager - Central orchestrator for distributed services.
"""

import zmq
import threading
import logging
from typing import Dict, Any, Optional, cast

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.schema import (
    RegisterServiceRequest, LookupServiceRequest, CallServiceRequest, ListServicesRequest,
    ServiceListResponse, ServiceLookupResponse, ResponseStatus, ServiceInfo
)

# Configure logging
logger = logging.getLogger(__name__)


class ServiceClient:
    """
    Client for interacting with registered services through the service manager.
    """

    def __init__(
        self,
        manager_host: str = constants.DEFAULT_HOST,
        manager_port: int = constants.SERVICE_MANAGER_PORT
    ) -> None:
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.context = zmq.Context()
        # NOTE: Do not keep a persistent REQ socket here.
        # If a request times out, a REQ socket is left in an invalid state
        # (must recv before next send), which causes ZMQ EFSM errors.
        self.socket = None

    def close(self) -> None:
        """Close underlying ZMQ resources immediately."""
        try:
            if self.context is not None:
                self.context.term()
        except Exception:
            pass

    def _call_manager(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Make a call to the service manager."""
        endpoint = f"tcp://{self.manager_host}:{self.manager_port}"
        socket: Optional[zmq.Socket] = None
        connected = False
        try:
            socket = self.context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)

            socket.connect(endpoint)
            connected = True
            socket.send_json(request)

            # Set timeout
            if socket.poll(timeout=constants.ZMQ_MANAGER_TIMEOUT):
                response = cast(Dict[str, Any], socket.recv_json())
                return response
            else:
                return ResponseStatus(status='error', message='Request timeout').model_dump()

        except Exception as e:
            return ResponseStatus(status='error', message=str(e)).model_dump()
        finally:
            try:
                if socket is not None:
                    if connected:
                        try:
                            socket.disconnect(endpoint)
                        except Exception:
                            pass
                    socket.close(linger=0)
            except Exception:
                pass

    def register(self, name: str, host: str, port: int,
                service_type: str, description: str = "") -> Dict[str, Any]:
        """Register a service with the manager."""
        try:
            req = RegisterServiceRequest(
                name=name,
                host=host,
                port=port,
                type=service_type,
                description=description
            )
            response = self._call_manager(req.model_dump())
            return response
        except Exception as e:
            return ResponseStatus(status='error', message=f"Registration failed: {e}").model_dump()

    def lookup(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Look up a service by name."""
        try:
            req = LookupServiceRequest(name=service_name)
            response = self._call_manager(req.model_dump())
            if response.get('status') == 'success':
                return response.get('service')
            return None
        except Exception:
            return ResponseStatus(status='error', message=f"Lookup failed for {service_name}").model_dump()

    def list_services(self) -> Dict[str, Any]:
        """List all registered services."""
        try:
            req = ListServicesRequest()
            return self._call_manager(req.model_dump())
        except Exception as e:
            return ResponseStatus(status='error', message=f"List services failed: {e}").model_dump()

    def call(self, service_name: str, service_request: Dict[str, Any]) -> Dict[str, Any]:
        """Call a service through the manager."""
        try:
            req = CallServiceRequest(
                service_name=service_name,
                request=service_request
            )
            return self._call_manager(req.model_dump())
        except Exception as e:
            return ResponseStatus(status='error', message=f"Service call prep failed: {e}").model_dump()


class ServiceManager:
    """
    Simple Service Manager for service registration, lookup, and calling.
    """

    def __init__(
        self,
        host: str = constants.DEFAULT_HOST,
        port: int = constants.SERVICE_MANAGER_PORT
    ) -> None:
        self.host = host
        self.port = port

        # Service registry
        self.services: Dict[str, ServiceInfo] = {}

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
                        message = cast(Dict[str, Any], self.socket.recv_json())
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
                req = RegisterServiceRequest(**request)
                return self._handle_register(req)
            elif command == 'LOOKUP':
                req = LookupServiceRequest(**request)
                return self._handle_lookup(req)
            elif command == 'LIST_SERVICES':
                return self._handle_list_services()
            elif command == 'CALL_SERVICE':
                req = CallServiceRequest(**request)
                return self._handle_call_service(req)
            else:
                return ResponseStatus(
                    status='error',
                    message=f'Unknown command: {command}'
                ).model_dump()

        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return ResponseStatus(
                status='error',
                message=str(e)
            ).model_dump()

    def _handle_register(self, request: RegisterServiceRequest) -> Dict[str, Any]:
        """Handle service registration."""
        try:
            name = request.name
            
            if name in self.services:
                logger.warning(f"Service {name} is already registered.")
                return ResponseStatus(
                    status='error',
                    message=f'Service {name} is already registered'
                ).model_dump()

            service_info = ServiceInfo(
                name=request.name,
                host=request.host,
                port=request.port,
                type=request.type,
                description=request.description
            )
            
            self.services[name] = service_info

            logger.info(f"Registered service: {name} ({request.type}) at {request.host}:{request.port}")

            return ResponseStatus(
                status='success',
                message=f'Service {name} registered successfully'
            ).model_dump()

        except Exception as e:
            return ResponseStatus(
                status='error',
                message=f'Registration failed: {e}'
            ).model_dump()

    def _handle_lookup(self, request: LookupServiceRequest) -> Dict[str, Any]:
        """Handle service lookup."""
        name = request.name
        if not name or name not in self.services:
            return ResponseStatus(
                status='error',
                message=f'Service {name} not found'
            ).model_dump()

        return ServiceLookupResponse(
            status='success',
            service=self.services[name]
        ).model_dump()

    def _handle_list_services(self) -> Dict[str, Any]:
        """Handle list services request."""
        return ServiceListResponse(
            status='success',
            services=self.services
        ).model_dump()

    def _handle_call_service(self, request: CallServiceRequest) -> Dict[str, Any]:
        """Handle direct service calls."""
        service_name = request.service_name
        service_request = request.request

        if not service_name or service_name not in self.services:
            return ResponseStatus(
                status='error',
                message=f'Service {service_name} not found'
            ).model_dump()

        service_info = self.services[service_name]

        # Make direct ZMQ call to service
        try:
            # Reuse existing context
            socket = self.context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(f"tcp://{service_info.host}:{service_info.port}")

            socket.send_json(service_request)

            if socket.poll(timeout=constants.ZMQ_SERVICE_CALL_TIMEOUT):
                response = cast(Dict[str, Any], socket.recv_json())
                socket.close()
                return response
            else:
                socket.close()
                return ResponseStatus(
                    status='error',
                    message=f'Service {service_name} timeout'
                ).model_dump()

        except Exception as e:
            return ResponseStatus(
                status='error',
                message=f'Service call failed: {e}'
            ).model_dump()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop()
