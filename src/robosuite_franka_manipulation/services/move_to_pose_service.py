#!/usr/bin/env python3
"""
Move To Pose Service.
Allows moving robot arms to specific poses via service calls.
"""

import zmq
import threading
import logging
from typing import Dict, Any, Optional, cast, Tuple
from robosuite_franka_manipulation.services.services_core import ServiceClient
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.controllers.arm import RobosuiteArmController
from robosuite_franka_manipulation.schema import (
    ResponseStatus, MoveToPoseRequest, CancelMoveRequest, MoveToPoseResponse,
    ArmStatusResponse, GetArmStatusRequest, FollowTrajectoryRequest
)

logger = logging.getLogger(__name__)

class MoveToPoseService:
    """
    ZMQ socket service for moving robot arms.
    """

    def __init__(self, 
                 host: str = constants.DEFAULT_HOST,
                 port: int = constants.MOVE_TO_POSE_SERVICE_PORT,
                 manager_host: str = constants.DEFAULT_HOST,
                 manager_port: int = constants.SERVICE_MANAGER_PORT):
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.service_client = ServiceClient(manager_host, manager_port)
        
        # Registry of controllers: {(robot_id, arm_id): controller}
        self.controllers: Dict[Tuple[int, str], RobosuiteArmController] = {}
        
        # Track active threads
        self.active_moves: Dict[Tuple[int, str], threading.Thread] = {}
        
        # Lock to protect active_moves and running state during threaded access
        self.state_lock = threading.RLock()

    def register_controller(self, robot_id: int, arm_id: str, controller: RobosuiteArmController) -> None:
        """Register a controller for a specific robot arm."""
        self.controllers[(robot_id, arm_id)] = controller
        logger.info(f"Registered controller for robot {robot_id}, arm {arm_id}")

    def start(self) -> None:
        """Start the Move To Pose service."""
        if self.running:
            logger.warning("Move To Pose service already running")
            return

        self.running = True
        
        # Register with Service Manager
        response = self.service_client.register(
            "move_to_pose_service",
            self.host,
            self.port,
            "control",
            "Move robot arm to pose service"
        )
        if response.get('status') != 'success':
            logger.warning("Failed to register Move To Pose service: %s", response.get('message'))
            self.running = False
            return

        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        logger.info(f"Move To Pose service started on tcp://{self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the service."""
        with self.state_lock:
            self.running = False
            
            # Cancel all active movements
            logger.info("Stopping Move To Pose service: cancelling all active moves")
            for key, controller in self.controllers.items():
                try:
                    controller.cancel_movement()
                except Exception as e:
                    logger.warning(f"Failed to cancel movement for {key}: {e}")
            
            # Copy items to list to iterate safely even if threads finish/modify dict (though lock prevents modification)
            active_threads = list(self.active_moves.items())

        # Wait for all movement threads to finish (outside lock to allow threads to exit)
        for key, thread in active_threads:
            if thread.is_alive():
                try:
                    thread.join(timeout=1.0)
                except Exception:
                    logger.warning(f"Failed to join movement thread for {key}")
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.socket:
            try:
                self.socket.close(linger=0)
            except Exception:
                logger.error("Failed to close ZMQ socket")
        if self.context:
            try:
                self.context.term()
            except Exception:
                logger.error("Failed to terminate ZMQ context")
        logger.info("Move To Pose service stopped")

    def _run_server(self) -> None:
        """Run the ZMQ server loop."""
        self.socket.bind(f"tcp://{self.host}:{self.port}")
        while self.running:
            try:
                if self.socket.poll(timeout=100):
                    message = cast(Dict[str, Any], self.socket.recv_json())
                    response = self._handle_request(message)
                    self.socket.send_json(response)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM or not self.running:
                    break
                logger.error(f"ZMQ error in Move To Pose service: {e}")
            except Exception as e:
                if not self.running:
                    break
                logger.error(f"Error in Move To Pose service: {e}")
                try:
                    self.socket.send_json({"status": "error", "message": str(e)})
                except Exception:
                    pass

    def _handle_request(self, request: Dict) -> Dict:
        """Handle a client request."""
        try:
            command = request.get("command")
            
            if command == "MOVE_TO_POSE":
                req = MoveToPoseRequest(**request)
                return self._handle_move(req)
            elif command == "FOLLOW_TRAJECTORY":
                req = FollowTrajectoryRequest(**request)
                return self._handle_follow_trajectory(req)
            elif command == "CANCEL_MOVE":
                req = CancelMoveRequest(**request)
                return self._handle_cancel(req)
            elif command == "GET_ARM_STATUS":
                req = GetArmStatusRequest(**request)
                return self._handle_get_status(req)
            else:
                return ResponseStatus(status="error", message=f"Unknown command: {command}").model_dump()
                
        except Exception as e:
            return ResponseStatus(status="error", message=str(e)).model_dump()

    def _handle_get_status(self, request: GetArmStatusRequest) -> Dict:
        """Handle GET_ARM_STATUS command."""
        robot_id = request.robot_id
        arm_id = request.arm
        key = (robot_id, arm_id)
        
        if key not in self.controllers:
             return ResponseStatus(status="error", message=f"No controller registered for robot {robot_id}, arm {arm_id}").model_dump()
        
        controller = self.controllers[key]
        status_code = -1 if controller.is_moving else controller.last_status
        
        return ArmStatusResponse(status_code=status_code).model_dump()

    def _handle_follow_trajectory(self, request: FollowTrajectoryRequest) -> Dict:
        """Handle FOLLOW_TRAJECTORY command."""
        try:
            robot_id = request.robot_id
            arm_id = request.arm
            gripper_action = request.gripper_action
            trajectory = request.trajectory

            key = (robot_id, arm_id)
            if key not in self.controllers:
                return ResponseStatus(status="error", message=f"No controller registered for robot {robot_id}, arm {arm_id}").model_dump()
            
            # Run in separate thread
            def traj_task():
                try:
                    logger.info(f"Starting trajectory following for robot {robot_id}, arm {arm_id}")
                    controller = self.controllers[key]
                    controller.follow_trajectory(
                        trajectory=trajectory,
                        gripper_action=gripper_action,
                    )
                    logger.info(f"Trajectory following complete for robot {robot_id}, arm {arm_id}")
                except Exception as e:
                    logger.error(f"Trajectory following failed for robot {robot_id}, arm {arm_id}: {e}")

            controller = self.controllers[key]

            with self.state_lock:
                if not self.running:
                    return ResponseStatus(status="error", message="Service is stopping").model_dump()

                # cleanup finished threads
                self.active_moves = {k: v for k, v in self.active_moves.items() if v.is_alive()}

                # Cancel existing move if any
                if key in self.active_moves and self.active_moves[key].is_alive():
                    logger.warning(f"Robot {robot_id} arm {arm_id} is already moving. Cancelling previous move.")
                    controller.cancel_movement()
                    self.state_lock.release()
                    try:
                        self.active_moves[key].join(timeout=5.0)
                    finally:
                        self.state_lock.acquire()
                    
                    if self.active_moves[key].is_alive():
                        return ResponseStatus(
                            status="error", 
                            message=f"Previous move failed to cancel for robot {robot_id}, arm {arm_id}."
                        ).model_dump()
                
                if not self.running:
                     return ResponseStatus(status="error", message="Service is stopping").model_dump()

                thread = threading.Thread(target=traj_task, daemon=True)
                self.active_moves[key] = thread
                thread.start()
            
            return MoveToPoseResponse(message="Trajectory following started").model_dump()

        except Exception as e:
            return ResponseStatus(status="error", message=f"Failed to process trajectory request: {e}").model_dump()

    def _handle_move(self, request: MoveToPoseRequest) -> Dict:
        """Handle MOVE_TO_POSE command."""
        try:
            robot_id = request.robot_id
            arm_id = request.arm
            gripper_action = request.gripper_action
            approach = request.approach
            target_pose = request.pose
            num_steps = request.num_steps

            if target_pose is None:
                return ResponseStatus(status="error", message="No target pose provided").model_dump()

            key = (robot_id, arm_id)
            if key not in self.controllers:
                return ResponseStatus(status="error", message=f"No controller registered for robot {robot_id}, arm {arm_id}").model_dump()
            
            # Run in separate thread to not block service
            def move_task():
                try:
                    logger.info(f"Starting move for robot {robot_id}, arm {arm_id}")
                    controller = self.controllers[key]
                    controller.move_to_pose(
                        target_pose=target_pose,
                        gripper_action=gripper_action,
                        approach=approach,
                        num_steps=num_steps
                    )
                    logger.info(f"Move complete for robot {robot_id}, arm {arm_id}")
                except Exception as e:
                    logger.error(f"Move failed for robot {robot_id}, arm {arm_id}: {e}")

            controller = self.controllers[key]

            with self.state_lock:
                if not self.running:
                    return ResponseStatus(status="error", message="Service is stopping").model_dump()

                # cleanup finished threads to prevent memory leak
                self.active_moves = {k: v for k, v in self.active_moves.items() if v.is_alive()}

                # Cancel existing move if any
                if key in self.active_moves and self.active_moves[key].is_alive():
                    logger.warning(f"Robot {robot_id} arm {arm_id} is already moving. Cancelling previous move.")
                    controller.cancel_movement()
                    # Wait longer for the thread to finish and verify it actually stopped
                    # We release the lock temporarily while joining to avoid blocking stop()
                    self.state_lock.release()
                    try:
                        self.active_moves[key].join(timeout=5.0)
                    finally:
                        self.state_lock.acquire()
                    
                    if self.active_moves[key].is_alive():
                        logger.error(f"Failed to stop existing move for robot {robot_id}, arm {arm_id}. Aborting new request.")
                        return ResponseStatus(
                            status="error", 
                            message=f"Previous move failed to cancel for robot {robot_id}, arm {arm_id}. Try again."
                        ).model_dump()
                
                # Double check running state after potential join delay
                if not self.running:
                     return ResponseStatus(status="error", message="Service is stopping").model_dump()

                thread = threading.Thread(target=move_task, daemon=True)
                self.active_moves[key] = thread
                thread.start()
            
            return MoveToPoseResponse(message="Move started").model_dump()

        except Exception as e:
            return ResponseStatus(status="error", message=f"Failed to process move request: {e}").model_dump()

    def _handle_cancel(self, request: CancelMoveRequest) -> Dict:
        """Handle CANCEL_MOVE command."""
        try:
            robot_id = request.robot_id
            arm_id = request.arm
            
            key = (robot_id, arm_id)
            if key not in self.controllers:
                return ResponseStatus(status="error", message=f"No controller registered for robot {robot_id}, arm {arm_id}").model_dump()
            
            controller = self.controllers[key]
            controller.cancel_movement()
            
            # Wait for thread to finish if tracked
            if key in self.active_moves and self.active_moves[key].is_alive():
                # We don't block too long here to keep service responsive
                self.active_moves[key].join(timeout=0.1)
                
            return MoveToPoseResponse(message="Cancellation requested").model_dump()
            
        except Exception as e:
            return ResponseStatus(status="error", message=f"Failed to cancel move: {e}").model_dump()
