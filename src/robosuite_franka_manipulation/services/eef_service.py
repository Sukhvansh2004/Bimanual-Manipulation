#!/usr/bin/env python3
"""
End-Effector Coordinates Service.
Provides current end-effector positions and orientations via TCP socket.
"""

import zmq
import threading
import time
import json
from typing import Dict, Any, Optional, Tuple
from robosuite_franka_manipulation.services.service_manager import ServiceClient

class EEFService:
    """
    ZMQ socket service for getting end-effector coordinates.
    Provides current positions and orientations of robot end-effectors.
    """

    def __init__(self, env: Any, host: str = 'localhost', port: int = 5559,
                 manager_host: str = 'localhost', manager_port: int = 5555):
        self.env = env
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.service_client = ServiceClient(manager_host, manager_port)

    def start(self) -> None:
        """Start the EEF coordinates service."""
        if self.running:
            print("EEF service already running")
            return

        self.running = True
        
        # Register with Service Manager
        registration_success = self.service_client.register(
            "eef_service",
            self.host,
            self.port,
            "eef",
            "End-effector coordinates service"
        )
        if not registration_success:
            print("Warning: Failed to register EEF service with Service Manager.")

        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"EEF service started on tcp://{self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the EEF coordinates service."""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        if self.thread:
            self.thread.join(timeout=1.0)
        print("EEF service stopped")

    def get_eef_coordinates(self) -> Dict[str, Any]:
        """
        Get current end-effector coordinates for all robots.

        Returns:
            Dict with positions and orientations in world frame
        """
        try:
            # Get current observation
            obs = self.env._get_observations()

            eef_data = {}

            # Extract EEF data for each robot
            for robot_id in [0, 1]:  # Two robots
                robot_key = f"robot{robot_id}"

                if f"{robot_key}_eef_pos" in obs:
                    pos = obs[f"{robot_key}_eef_pos"]
                    quat = obs[f"{robot_key}_eef_quat"]

                    # Convert to lists for JSON serialization
                    eef_data[f"robot_{robot_id}"] = {
                        "position": {
                            "x": float(pos[0]),
                            "y": float(pos[1]),
                            "z": float(pos[2])
                        },
                        "orientation": {
                            "w": float(quat[0]),
                            "x": float(quat[1]),
                            "y": float(quat[2]),
                            "z": float(quat[3])
                        },
                        "frame": "world"
                    }

            return {
                "success": True,
                "eef_coordinates": eef_data,
                "timestamp": time.time(),
                "frame": "world"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get EEF coordinates: {str(e)}"
            }

    def _run_server(self) -> None:
        """Run the ZMQ server loop."""
        self.socket.bind(f"tcp://{self.host}:{self.port}")
        while self.running:
            try:
                message = self.socket.recv_json()
                response = self._handle_request(message)
                self.socket.send_json(response)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break
                print(f"ZMQ error in EEF service: {e}")
            except Exception as e:
                print(f"Error in EEF service server: {e}")
                self.socket.send_json({"success": False, "error": str(e)})

    def _handle_request(self, request: Dict) -> Dict:
        """Handle a client request."""
        command = request.get("command")
        if command == "GET":
            return self.get_eef_coordinates()
        else:
            return {"success": False, "error": "Invalid command. Use 'GET'."}