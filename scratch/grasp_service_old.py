#!/usr/bin/env python3
"""
Grasp Prediction Service using ZMQ.
Provides remote grasp prediction using GraspGen.
"""

import zmq
import json
import numpy as np
from typing import Dict, Any, Optional
import threading
import time


class GraspService:
    """
    ZMQ-based service for remote grasp prediction.
    Uses GraspGen to predict grasps for point clouds.
    """

    def __init__(self, host: str = 'localhost', port: int = 5558):
        self.host = host
        self.port = port
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Initialize GraspGen predictor
        try:
            from ..perception.grasp_predictor import GraspPredictor
            self.predictor = GraspPredictor()
            print("Grasp service: GraspGen predictor initialized")
        except Exception as e:
            print(f"Grasp service: Failed to initialize GraspGen: {e}")
            self.predictor = None

    def start(self) -> None:
        """Start the grasp prediction service."""
        if self.running:
            print("Grasp service already running")
            return

        if not self.predictor:
            print("Grasp service: Cannot start - GraspGen not available")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"Grasp service started on tcp://{self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the grasp prediction service."""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        if self.thread:
            self.thread.join(timeout=1.0)
        print("Grasp service stopped")

    def predict_grasps_remote(self, point_cloud: np.ndarray, object_pose: Dict,
                            num_grasps: int = 10) -> Dict[str, Any]:
        """
        Predict grasps for remote request.

        Args:
            point_cloud: Point cloud as numpy array
            object_pose: Object pose dict with 'pos' and 'quat'
            num_grasps: Number of grasps to predict

        Returns:
            Response dict with grasps or error
        """
        if not self.predictor:
            return {"error": "GraspGen not available"}

        try:
            grasps = self.predictor.predict_grasps(point_cloud, object_pose, num_grasps)

            # Convert numpy arrays to lists for JSON serialization
            serializable_grasps = []
            for grasp in grasps:
                serializable_grasp = {
                    "pos": grasp["pos"].tolist() if hasattr(grasp["pos"], "tolist") else grasp["pos"],
                    "quat": grasp["quat"].tolist() if hasattr(grasp["quat"], "tolist") else grasp["quat"],
                    "score": float(grasp["score"])
                }
                serializable_grasps.append(serializable_grasp)

            return {
                "success": True,
                "grasps": serializable_grasps,
                "num_grasps": len(serializable_grasps)
            }

        except Exception as e:
            return {"error": f"Grasp prediction failed: {str(e)}"}

    def _run_server(self) -> None:
        """Run the ZMQ server loop."""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://{self.host}:{self.port}")

            print(f"Grasp service listening on tcp://{self.host}:{self.port}")

            while self.running:
                try:
                    # Wait for request
                    message = self.socket.recv_json()

                    # Process request
                    if "point_cloud" not in message or "object_pose" not in message:
                        response = {"error": "Missing required fields: point_cloud, object_pose"}
                    else:
                        # Convert lists back to numpy arrays
                        point_cloud = np.array(message["point_cloud"])
                        object_pose = message["object_pose"]
                        num_grasps = message.get("num_grasps", 10)

                        response = self.predict_grasps_remote(point_cloud, object_pose, num_grasps)

                    # Send response
                    self.socket.send_json(response)

                except zmq.ZMQError:
                    break
                except Exception as e:
                    print(f"Grasp service error: {e}")
                    try:
                        self.socket.send_json({"error": str(e)})
                    except:
                        pass

        except Exception as e:
            print(f"Grasp service startup error: {e}")
        finally:
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()


def main():
    """Run the grasp service standalone."""
    import argparse

    parser = argparse.ArgumentParser(description="Grasp Prediction Service")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5558, help="Port to bind to")

    args = parser.parse_args()

    service = GraspService(host=args.host, port=args.port)
    service.start()

    try:
        print("Grasp service running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping grasp service...")
        service.stop()


if __name__ == "__main__":
    main()