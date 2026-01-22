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
from robosuite_franka_manipulation.services.services_core import ServiceClient
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.schema import EEFRequest, EEFResponse
import numpy as np
from scipy.spatial.transform import Rotation as R
from contextlib import nullcontext
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose

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
        self.transform_manager = TransformManager(env, env_lock=self.env_lock)

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
        Get current end-effector coordinates for all arms of all robots.

        Returns:
            Dict with StampedPose objects in world frame
        """
        with self.env_lock if self.env_lock else nullcontext():
            try:
                # Force MuJoCo to sync forward kinematics before reading
                self.env.sim.forward()

                eef_data = {}

                # Iterate over all robots in the environment
                for robot_id, robot in enumerate(self.env.robots):
                    base_frame = f"robot{robot_id}_base"
                    logger.debug(f"Getting EEF for robot {robot_id} with base frame {base_frame}")

                    # robot.robot_model.eef_name is typically a dict mapping arm_id to body_name
                    # or a string for single-arm robots. We handle both for robustness.
                    eef_info = robot.robot_model.eef_name

                    for arm_id, eef_body in eef_info.items():
                        # Get pose in robot base frame using robosuite's public API
                        pose4 = robot.pose_in_base_from_name(eef_body)
                        
                        pos = np.asarray(pose4[:3, 3], dtype=np.float64).copy()
                        quat = R.from_matrix(np.asarray(pose4[:3, :3], dtype=np.float64)).as_quat()
                        
                        # Create pose in robot base frame
                        stamped_pose_base = StampedPose(
                            position=pos,
                            quaternion=quat,
                            frame_id=base_frame,
                            timestamp=time.time()
                        )
                        
                        # Transform to world frame
                        stamped_pose_world = self.transform_manager.transform_pose(
                            stamped_pose_base, 
                            constants.WORLD_FRAME
                        )
                        
                        logger.debug(f"Robot {robot_id} arm {arm_id} EEF pose in world frame: {stamped_pose_world}")
                        # Store in dictionary with key: robot_{robot_id}_{arm_id}
                        eef_data[f"robot_{robot_id}_{arm_id}"] = stamped_pose_world

                return {
                    "success": True,
                    "eef_coordinates": eef_data,
                    "timestamp": time.time(),
                    "frame": constants.WORLD_FRAME
                }

            except Exception as e:
                logger.error(f"Failed to get EEF coordinates: {e}")
                return {
                    "success": False,
                    "error": f"Failed to get EEF coordinates: {str(e)}"
                }

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
            return EEFResponse(status="error", message=str(e), success=False).model_dump()

        if req.command == "GET":
            response_data = self.get_eef_coordinates()
            logger.info(f"EEF GET request processed: {response_data}")
            if not response_data.get("success", False):
                return EEFResponse(
                    status="error", 
                    message=response_data.get("error", "Unknown error"),
                    success=False
                ).model_dump()
            
            return EEFResponse(**response_data).model_dump()
        else:
            return EEFResponse(status="error", message="Invalid command. Use 'GET'.", success=False).model_dump()