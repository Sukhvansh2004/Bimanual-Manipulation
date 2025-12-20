#!/usr/bin/env python3
"""
Enhanced Service Manager - Central orchestrator for distributed perception services.

This service manager provides:
- Service discovery and registration
- Orchestrated perception pipeline (FastSAM → Grasp prediction)
- Unified API for complex multi-service operations
- Health monitoring and automatic failover
- REST-like interface for easy integration
"""

import zmq
import threading
import time
import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceClient:
    """
    Client for interacting with registered services through the service manager.
    """

    def __init__(self, manager_host: str = 'localhost', manager_port: int = 5555):
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)

    def _call_manager(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Make a call to the service manager."""
        try:
            self.socket.connect(f"tcp://{self.manager_host}:{self.manager_port}")
            self.socket.send_json(request)

            # Set timeout
            if self.socket.poll(timeout=15000):  # 15 second timeout
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

    def get_perception_pipeline(self, num_grasps: int = 5) -> Dict[str, Any]:
        """Get complete perception pipeline results (detections + grasps)."""
        request = {
            'command': 'GET_PERCEPTION_PIPELINE',
            'num_grasps': num_grasps
        }
        return self._call_manager(request)

    def set_detection_prompt(self, prompt: str) -> Dict[str, Any]:
        """Set the detection prompt for FastSAM service."""
        request = {
            'command': 'SET_DETECTION_PROMPT',
            'prompt': prompt
        }
        return self._call_manager(request)

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all registered services."""
        request = {'command': 'GET_STATUS'}
        return self._call_manager(request)


class EnhancedServiceManager:
    """
    Enhanced Service Manager with orchestration capabilities.

    This manager not only handles service registration and discovery,
    but also provides high-level orchestration for complex multi-service
    operations like the complete perception pipeline.
    """

    def __init__(self, host: str = 'localhost', port: int = 5555):
        self.host = host
        self.port = port

        # Service registry
        self.services: Dict[str, Dict[str, Any]] = {}
        self.service_types: Dict[str, List[str]] = defaultdict(list)

        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.last_health_check = {}

        # Orchestration state
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.health_thread: Optional[threading.Thread] = None

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        logger.info(f"Enhanced Service Manager initialized on {host}:{port}")

    def start(self) -> None:
        """Start the service manager."""
        if self.running:
            logger.warning("Service Manager already running")
            return

        logger.info("Starting Enhanced Service Manager...")
        self.running = True

        # Start main server thread
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()

        # Start health monitoring thread
        self.health_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self.health_thread.start()

        logger.info("Enhanced Service Manager started successfully")

    def stop(self) -> None:
        """Stop the service manager."""
        if not self.running:
            return

        logger.info("Stopping Enhanced Service Manager...")
        self.running = False

        # Wait for threads
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        if self.health_thread and self.health_thread.is_alive():
            self.health_thread.join(timeout=2.0)

        # Close ZMQ
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

        logger.info("Enhanced Service Manager stopped")

    def _server_loop(self) -> None:
        """Main ZMQ server loop."""
        logger.info(f"Starting ZMQ server on tcp://{self.host}:{self.port}")

        try:
            self.socket.bind(f"tcp://{self.host}:{self.port}")

            while self.running:
                try:
                    if self.socket.poll(timeout=1000):
                        message = self.socket.recv_json()
                        response = self._handle_request(message)
                        self.socket.send_json(response)

                except zmq.ZMQError as e:
                    if self.running:
                        logger.error(f"ZMQ error: {e}")

        except Exception as e:
            logger.error(f"Server loop error: {e}")
        finally:
            self.socket.close()

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
            elif command == 'GET_PERCEPTION_PIPELINE':
                return self._handle_perception_pipeline(request)
            elif command == 'SET_DETECTION_PROMPT':
                return self._handle_set_prompt(request)
            elif command == 'GET_STATUS':
                return self._handle_get_status()
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
                'registered_at': time.time(),
                'status': 'active'
            }

            self.services[name] = service_info
            self.service_types[service_type].append(name)

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

            if socket.poll(timeout=10000):  # 10 second timeout
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

    def _handle_perception_pipeline(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle orchestrated perception pipeline requests."""
        num_grasps = request.get('num_grasps', 5)

        try:
            # Step 1: Get detections from FastSAM service
            fastsam_response = self._handle_call_service({
                'service_name': 'fastsam_service',
                'request': {'command': 'GET_DETECTIONS'}
            })

            if fastsam_response.get('status') != 'success':
                return {
                    'status': 'error',
                    'message': 'Failed to get detections from FastSAM service',
                    'details': fastsam_response
                }

            detections = fastsam_response.get('detections', [])

            if not detections:
                return {
                    'status': 'success',
                    'message': 'No objects detected',
                    'detections': [],
                    'grasps': []
                }

            # Step 2: Get point clouds
            pointclouds_response = self._handle_call_service({
                'service_name': 'fastsam_service',
                'request': {'command': 'GET_POINTCLOUDS'}
            })

            pointclouds = []
            if pointclouds_response.get('status') == 'success':
                pointclouds = pointclouds_response.get('pointclouds', [])

            # Step 3: Generate grasps for each detection
            pipeline_results = []

            for i, detection in enumerate(detections):
                pointcloud = pointclouds[i] if i < len(pointclouds) else []

                if not pointcloud:
                    pipeline_results.append({
                        'detection': detection,
                        'grasps': [],
                        'message': 'No point cloud available'
                    })
                    continue

                # Create object pose (simplified - in practice this would come from detection)
                object_pose = {
                    'pos': [0.5, 0, 0.1],  # Default position
                    'quat': [0, 0, 0, 1]   # Default orientation
                }

                # Get grasps from Grasp service
                grasp_response = self._handle_call_service({
                    'service_name': 'grasp_service',
                    'request': {
                        'command': 'PREDICT_GRASPS',
                        'pointcloud': pointcloud,
                        'object_pose': object_pose,
                        'num_grasps': num_grasps
                    }
                })

                if grasp_response.get('status') == 'success':
                    grasps = grasp_response.get('grasps', [])
                else:
                    grasps = []

                pipeline_results.append({
                    'detection': detection,
                    'grasps': grasps,
                    'pointcloud_size': len(pointcloud),
                    'num_grasps': len(grasps)
                })

            return {
                'status': 'success',
                'pipeline_results': pipeline_results,
                'total_detections': len(detections),
                'total_grasps': sum(len(r.get('grasps', [])) for r in pipeline_results)
            }

        except Exception as e:
            logger.error(f"Perception pipeline error: {e}")
            return {
                'status': 'error',
                'message': f'Perception pipeline failed: {e}'
            }

    def _handle_set_prompt(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompt setting requests."""
        prompt = request.get('prompt')
        if not prompt:
            return {
                'status': 'error',
                'message': 'No prompt provided'
            }

        # Forward to FastSAM service
        return self._handle_call_service({
            'service_name': 'fastsam_service',
            'request': {
                'command': 'SET_PROMPT',
                'prompt': prompt
            }
        })

    def _handle_get_status(self) -> Dict[str, Any]:
        """Handle status requests."""
        return {
            'status': 'success',
            'manager_info': {
                'host': self.host,
                'port': self.port,
                'running': self.running
            },
            'services': self.services,
            'service_types': dict(self.service_types),
            'total_services': len(self.services)
        }

    def _health_monitor_loop(self) -> None:
        """Monitor health of registered services."""
        logger.info("Starting health monitoring")

        while self.running:
            try:
                current_time = time.time()

                for service_name, service_info in self.services.items():
                    # Check if we need to run health check
                    last_check = self.last_health_check.get(service_name, 0)
                    if current_time - last_check < self.health_check_interval:
                        continue

                    # Perform health check
                    try:
                        context = zmq.Context()
                        socket = context.socket(zmq.REQ)
                        socket.connect(f"tcp://{service_info['host']}:{service_info['port']}")

                        socket.send_json({'command': 'GET_STATUS'})

                        if socket.poll(timeout=2000):  # 2 second timeout
                            response = socket.recv_json()
                            if response.get('status') == 'success':
                                service_info['status'] = 'healthy'
                                service_info['last_health_check'] = current_time
                            else:
                                service_info['status'] = 'unhealthy'
                        else:
                            service_info['status'] = 'unreachable'

                        socket.close()
                        context.term()

                    except Exception as e:
                        service_info['status'] = 'error'
                        logger.warning(f"Health check failed for {service_name}: {e}")

                    self.last_health_check[service_name] = current_time

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(10)

        logger.info("Health monitoring stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()

    def _handle_request(self, request: Dict) -> Dict:
        """Handles incoming requests for registration and discovery."""
        command = request.get("command")
        if command == "REGISTER":
            name = request.get("name")
            host = request.get("host")
            port = request.get("port")
            if name and host and port:
                self.services[name] = {"host": host, "port": port}
                print(f"Registered service '{name}' at {host}:{port}")
                return {"status": "success", "message": f"Service '{name}' registered."}
            else:
                return {"status": "error", "message": "Missing name, host, or port for registration."}

        elif command == "LOOKUP":
            name = request.get("name")
            if name in self.services:
                return {"status": "success", "service": self.services[name]}
            else:
                return {"status": "error", "message": f"Service '{name}' not found."}
        
        elif command == "LIST":
            return {"status": "success", "services": self.services}

        else:
            return {"status": "error", "message": f"Unknown command: {command}"}
