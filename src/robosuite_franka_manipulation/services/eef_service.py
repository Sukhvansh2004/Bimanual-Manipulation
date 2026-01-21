#!/usr/bin/env python3
"""
End-Effector Coordinates Service.
Provides current end-effector positions and orientations via TCP socket.
"""

import zmq
import threading
import time
import logging
from typing import Dict, Any, Optional, cast, Union
from robosuite_franka_manipulation.services.service_manager import ServiceClient
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.schema import EEFRequest, EEFResponse
from robosuite_franka_manipulation.transforms import StampedPose

logger = logging.getLogger(__name__)

class EEFService:
    """
    ZMQ socket service for getting end-effector coordinates.
    Provides current positions and orientations of robot end-effectors.
    """

    env_lock: Optional[threading.Lock] = None

    def __init__(self, env: DualArmEnvironment, host: str = constants.DEFAULT_HOST,
                 port: int = constants.EEF_SERVICE_PORT,
                 manager_host: str = constants.DEFAULT_HOST,
                 manager_port: int = constants.SERVICE_MANAGER_PORT,
                 env_lock: Optional[threading.Lock] = None):
        self.env = env
        self.host = host
        self.port = port
        self.env_lock = env_lock
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.service_client = ServiceClient(manager_host, manager_port)

    def start(self) -> None:
        """Start the EEF coordinates service."""
        if self.running:
            logger.warning("EEF service already running")
            return

        self.running = True
        
        # Register with Service Manager
        response = self.service_client.register(
            "eef_service",
            self.host,
            self.port,
            "eef",
            "End-effector coordinates service"
        )
        if response.get('status') != 'success':
            logger.warning("Failed to register EEF service with Service Manager: %s", response.get('message'))
            self.running = False
            return

        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger.info(f"EEF service started on tcp://{self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the EEF coordinates service."""
        self.running = False
        
        # Wait for thread to exit first (it will stop when self.running = False)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Now safe to close socket and context
        if self.socket:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
        if self.context:
            try:
                self.context.term()
            except Exception:
                pass
        logger.info("EEF service stopped")

    def get_eef_coordinates(self) -> Dict[str, Any]:
        """
        Get current end-effector coordinates for all robots.

        Returns:
            Dict with positions and orientations in world frame
        """
        lock = self.env_lock
        try:
            if lock:
                lock.acquire()
            # Get current observation
            # robosuite exposes observation_spec() publicly; it returns the current
            # observation dict (and internally handles viewer_get_obs if enabled).
            obs = self.env.observation_spec()

            eef_data = {}

            # Extract EEF data for each robot
            for robot_id in [0, 1]:  # Two robots
                robot_key = f"robot{robot_id}"

                if f"{robot_key}_eef_pos" in obs:
                    pos = obs[f"{robot_key}_eef_pos"]
                    quat = obs[f"{robot_key}_eef_quat"]

                    # Use StampedPose object directly
                    eef_data[f"robot_{robot_id}"] = StampedPose(
                        position=pos,
                        quaternion=quat,
                        frame_id=constants.WORLD_FRAME,
                        timestamp=time.time()
                    )

            return {
                "success": True,
                "eef_coordinates": eef_data,
                "timestamp": time.time(),
                "frame": constants.WORLD_FRAME
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get EEF coordinates: {str(e)}"
            }
        finally:
            if lock:
                lock.release()

    def _run_server(self) -> None:
        """Run the ZMQ server loop."""
        self.socket.bind(f"tcp://{self.host}:{self.port}")
        while self.running:
            try:
                # Use poll with timeout to allow checking self.running flag
                if self.socket.poll(timeout=100):  # 100ms timeout
                    message = cast(Dict[str, Any], self.socket.recv_json())
                    response = self._handle_request(message)
                    self.socket.send_json(response)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM or not self.running:
                    break
                logger.error(f"ZMQ error in EEF service: {e}")
            except Exception as e:
                if not self.running:
                    break
                logger.error(f"Error in EEF service server: {e}")
                try:
                    self.socket.send_json({"success": False, "error": str(e)})
                except Exception:
                    pass

    def _handle_request(self, request: Dict) -> Dict:
        """Handle a client request."""
        try:
            req = EEFRequest(**request)
        except Exception as e:
            return EEFResponse(success=False, error=str(e)).model_dump()

        if req.command == "GET":
            response_data = self.get_eef_coordinates()
            return EEFResponse(**response_data).model_dump()
        else:
            return EEFResponse(success=False, error="Invalid command. Use 'GET'.").model_dump()