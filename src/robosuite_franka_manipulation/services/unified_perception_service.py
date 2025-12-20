import zmq
import json
import numpy as np
import threading
import time
from typing import Dict, Any, List, Optional
import cv2

from robosuite_franka_manipulation.perception.object_detector import ObjectDetector
from robosuite_franka_manipulation.perception.grasp_predictor import GraspPredictor
from robosuite_franka_manipulation.services.service_manager import ServiceClient

class UnifiedPerceptionService:
    """
    A unified service that combines FastSAM segmentation, dynamic prompt handling,
    and GraspGen grasp prediction. It runs continuous perception and exposes
    APIs via a ZMQ REP socket for external control.
    """
    def __init__(self, env: Any, host: str = 'localhost', port: int = 5550, 
                 manager_host: str = 'localhost', manager_port: int = 5555,
                 debug: bool = True, visualize: bool = True):
        self.env = env
        self.host = host
        self.port = port
        self.debug = debug
        self.visualize = visualize
        self.current_prompt = "object" # Initial default prompt

        # Initialize perception components
        self.object_detector = ObjectDetector(self.env, default_prompt=self.current_prompt, debug=debug, visualize=visualize)
        self.grasp_predictor = GraspPredictor(debug=debug, visualize=visualize)
        
        # Threading for continuous perception and ZMQ server
        self.running = False
        self.perception_thread: Optional[threading.Thread] = None
        self.server_thread: Optional[threading.Thread] = None
        
        # Store latest perception results
        self.latest_detected_objects: List[Dict] = []
        self.latest_fastsam_frame: Optional[np.ndarray] = None # For 2D video feed

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        # Service client for registration
        self.service_client = ServiceClient(manager_host, manager_port)

    def start(self) -> None:
        """Starts the continuous perception loop and the ZMQ server."""
        if self.running:
            print("Unified Perception Service already running.")
            return

        self.running = True
        
        # Register with Service Manager
        registration_success = self.service_client.register("perception_service", self.host, self.port)
        if not registration_success:
            print("Warning: Failed to register with Service Manager.")

        self.perception_thread = threading.Thread(target=self._perception_loop, daemon=True)
        self.perception_thread.start()
        print("Continuous perception loop started.")

        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        print(f"Unified Perception Service API listening on tcp://{self.host}:{self.port}")

    def stop(self) -> None:
        """Stops the service threads."""
        self.running = False
        if self.perception_thread:
            self.perception_thread.join()
        if self.server_thread:
            self.server_thread.join()
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        cv2.destroyAllWindows() # Close any OpenCV windows
        print("Unified Perception Service stopped.")

    def _perception_loop(self) -> None:
        """
        Continuous loop for object detection and grasp prediction.
        Also handles 2D FastSAM visualization.
        """
        while self.running:
            try:
                # Get observations from environment
                obs = self.env._get_observations()
                rgb_image = obs.get(f'{self.object_detector.camera_name}_image')
                depth_image = obs.get(f'{self.object_detector.camera_name}_depth')

                if rgb_image is not None:
                    # Perform object detection and grasp prediction
                    detected_objects = self.object_detector.detect_objects(rgb_image=rgb_image, depth_image=depth_image)
                    self.latest_detected_objects = detected_objects
                
                time.sleep(0.05) # Control loop frequency
            except Exception as e:
                print(f"Error in perception loop: {e}")
                time.sleep(1) # Sleep to avoid rapid error logging

    def _run_server(self) -> None:
        """
        Runs the ZMQ server to handle external requests for prompts and grasps.
        """
        self.socket.bind(f"tcp://{self.host}:{self.port}")
        while self.running:
            try:
                message = self.socket.recv_json()
                response = self._handle_request(message)
                self.socket.send_json(response)
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM: # Context terminated
                    break
                print(f"ZMQ error in server loop: {e}")
            except Exception as e:
                print(f"Error in ZMQ server: {e}")
                self.socket.send_json({"status": "error", "message": str(e)}) # Send error back

    def _handle_request(self, request: Dict) -> Dict:
        """Handles incoming API requests."""
        command = request.get("command")
        
        if command == "SET_PROMPT":
            new_prompt = request.get("prompt")
            if new_prompt:
                self.current_prompt = new_prompt
                self.object_detector.default_prompt = new_prompt # Update detector's prompt directly
                return {"status": "success", "message": f"Prompt set to '{new_prompt}'"}
            else:
                return {"status": "error", "message": "Prompt not provided."}
        
        elif command == "GET_PROMPT":
            return {"status": "success", "prompt": self.current_prompt}
            
        elif command == "GET_LATEST_OBJECTS":
            # Convert numpy arrays to lists for JSON serialization
            serializable_objects = []
            for obj in self.latest_detected_objects:
                serializable_grasps = []
                for grasp in obj['grasps']:
                    serializable_grasps.append({
                        "pos": grasp["pos"].tolist(),
                        "quat": grasp["quat"].tolist(),
                        "score": float(grasp["score"])
                    })
                serializable_objects.append({
                    "name": obj["name"],
                    "pose": {
                        "pos": obj["pose"]["pos"].tolist(),
                        "quat": obj["pose"]["quat"].tolist()
                    },
                    # "pcd": obj["pcd"].tolist(), # Point cloud might be too large for JSON
                    "mask": obj["mask"].tolist(), # Send mask as list
                    "grasps": serializable_grasps,
                    "id": obj["id"]
                })
            return {"status": "success", "objects": serializable_objects}

        elif command == "GET_FASTSAM_FRAME":
            if self.latest_fastsam_frame is not None:
                # Encode image to base64 or send as raw bytes
                # For simplicity, returning a placeholder here.
                # Actual implementation would involve cv2.imencode or similar.
                return {"status": "success", "message": "Image data transfer not yet implemented."}
            else:
                return {"status": "error", "message": "No FastSAM frame available."}
        
        else:
            return {"status": "error", "message": f"Unknown command: {command}"}
