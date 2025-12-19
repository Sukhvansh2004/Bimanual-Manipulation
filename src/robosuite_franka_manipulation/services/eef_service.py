#!/usr/bin/env python3
"""
End-Effector Coordinates Service.
Provides current end-effector positions and orientations via TCP socket.
"""

import socket
import threading
import time
import json
from typing import Dict, Any, Optional, Tuple


class EEFService:
    """
    TCP socket service for getting end-effector coordinates.
    Provides current positions and orientations of robot end-effectors.
    """

    def __init__(self, env: Any, host: str = 'localhost', port: int = 5559):
        self.env = env
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the EEF coordinates service."""
        if self.running:
            print("EEF service already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"EEF service started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the EEF coordinates service."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
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
        """Run the TCP server loop."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # Allow checking self.running

            print(f"EEF service listening on {self.host}:{self.port}")

            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"EEF service: Connection from {address}")
                    self._handle_client(client_socket)
                except socket.timeout:
                    continue
                except OSError:
                    break  # Socket was closed

        except Exception as e:
            print(f"EEF service error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()

    def _handle_client(self, client_socket: socket.socket) -> None:
        """Handle a client connection."""
        try:
            data = client_socket.recv(1024).decode('utf-8').strip()

            if not data:
                response = self.get_eef_coordinates()
            elif data.upper() == "GET":
                response = self.get_eef_coordinates()
            else:
                response = {
                    "success": False,
                    "error": "Invalid command. Use 'GET' or send empty message"
                }

            # Send JSON response
            client_socket.send(json.dumps(response).encode('utf-8'))

        except Exception as e:
            print(f"Error handling client: {e}")
            try:
                error_response = {"success": False, "error": str(e)}
                client_socket.send(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()


def main():
    """Run the EEF service standalone."""
    import argparse
    import robosuite as suite

    parser = argparse.ArgumentParser(description="End-Effector Coordinates Service")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5559, help="Port to bind to")
    parser.add_argument("--env", default="TwoArmPegInHole", help="Robosuite environment name")

    args = parser.parse_args()

    # Create environment
    env = suite.make(
        env_name=args.env,
        robots=['Panda', 'Panda'],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )

    service = EEFService(env, host=args.host, port=args.port)
    service.start()

    try:
        print("EEF service running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping EEF service...")
        service.stop()
    finally:
        env.close()


if __name__ == "__main__":
    main()