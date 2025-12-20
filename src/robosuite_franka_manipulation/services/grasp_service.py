#!/usr/bin/env python3
"""
Grasp Service - Grasp prediction service for segmented point clouds.

This service provides:
- Grasp prediction from 3D point clouds using GraspGen
- Batch processing of multiple objects
- Confidence scoring and grasp ranking
- World frame coordinate transformation
- ZMQ interface for distributed operation
"""

import zmq
import json
import numpy as np
import threading
import time
from typing import Dict, Any, List, Optional
import logging

from robosuite_franka_manipulation.perception.grasp_predictor import GraspPredictor
from robosuite_franka_manipulation.services.service_manager import ServiceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraspService:
    """
    Grasp Service for predicting grasps from 3D point clouds.

    This service takes segmented point clouds (typically from FastSAM service)
    and predicts 6-DOF grasps using the GraspGen model. It provides grasp
    ranking, confidence scoring, and coordinate transformations.
    """

    def __init__(self,
                 host: str = 'localhost',
                 port: int = 5552,
                 manager_host: str = 'localhost',
                 manager_port: int = 5555,
                 model_path: Optional[str] = None,
                 debug: bool = False,
                 visualize: bool = False):
        """
        Initialize Grasp Service.

        Args:
            host: Service host address
            port: Service port
            manager_host: Service manager host
            manager_port: Service manager port
            model_path: Path to GraspGen model config file
            debug: Enable debug logging
            visualize: Enable grasp visualization
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.visualize = visualize

        # Service state
        self.running = False

        # Initialize grasp predictor
        self.grasp_predictor = GraspPredictor(model_path=model_path, debug=debug, visualize=visualize)

        # Threading
        self.server_thread: Optional[threading.Thread] = None

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        # Service registration
        self.service_client = ServiceClient(manager_host, manager_port)

        logger.info(f"Grasp Service initialized on {host}:{port}")

    def _validate_pointcloud(self, pointcloud: np.ndarray) -> bool:
        """
        Validate point cloud data.

        Args:
            pointcloud: Nx3 point cloud array

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(pointcloud, np.ndarray):
            return False

        if pointcloud.size == 0:
            return False

        if pointcloud.shape[1] != 3:
            return False

        # Check for reasonable coordinate ranges
        if np.any(np.abs(pointcloud) > 10.0):  # 10 meter range
            return False

        return True

    def _transform_grasps_to_world_frame(self, grasps: List[Dict],
                                       object_pose: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Transform grasps from object frame to world frame.

        Args:
            grasps: List of grasp dictionaries with 'pos' and 'quat'
            object_pose: Object pose with 'pos' and 'quat' in world frame

        Returns:
            List of grasps transformed to world frame
        """
        try:
            object_pos = object_pose['pos']
            object_quat = object_pose['quat']

            # Convert quaternion to rotation matrix
            from scipy.spatial.transform import Rotation as R
            object_rot = R.from_quat(object_quat).as_matrix()

            transformed_grasps = []
            for grasp in grasps:
                # Transform position
                grasp_pos_local = np.array(grasp['pos'])
                grasp_pos_world = object_pos + object_rot @ grasp_pos_local

                # Transform orientation (compose rotations)
                grasp_quat_local = np.array(grasp['quat'])
                grasp_rot_local = R.from_quat(grasp_quat_local).as_matrix()
                grasp_rot_world = object_rot @ grasp_rot_local
                grasp_quat_world = R.from_matrix(grasp_rot_world).as_quat()

                transformed_grasp = grasp.copy()
                transformed_grasp['pos'] = grasp_pos_world.tolist()
                transformed_grasp['quat'] = grasp_quat_world.tolist()
                transformed_grasp['frame'] = 'world'

                transformed_grasps.append(transformed_grasp)

            return transformed_grasps

        except Exception as e:
            logger.error(f"Grasp transformation failed: {e}")
            return grasps  # Return original grasps if transformation fails

    def _predict_grasps_for_pointcloud(self, pointcloud: np.ndarray,
                                     object_pose: Dict[str, np.ndarray],
                                     num_grasps: int = 5) -> List[Dict]:
        """
        Predict grasps for a single point cloud.

        Args:
            pointcloud: Nx3 point cloud array
            object_pose: Object pose in world frame
            num_grasps: Number of grasps to predict

        Returns:
            List of grasp dictionaries
        """
        try:
            if not self._validate_pointcloud(pointcloud):
                logger.warning("Invalid point cloud provided")
                return []

            # Predict grasps using GraspGen
            grasps = self.grasp_predictor.predict_grasps(
                pointcloud=pointcloud,
                object_pose=object_pose,
                num_grasps=num_grasps
            )

            if not grasps:
                logger.info("No grasps predicted for point cloud")
                return []

            # Grasps are already in world frame from GraspPredictor
            # Add metadata
            for i, grasp in enumerate(grasps):
                grasp['id'] = i
                grasp['pointcloud_size'] = len(pointcloud)
                grasp['prediction_timestamp'] = time.time()

            if self.debug:
                logger.info(f"Predicted {len(grasps)} grasps for point cloud with {len(pointcloud)} points")

            return grasps

        except Exception as e:
            logger.error(f"Grasp prediction failed: {e}")
            return []

    def _server_loop(self) -> None:
        """ZMQ server loop for handling client requests."""
        logger.info(f"Starting ZMQ server on tcp://{self.host}:{self.port}")

        try:
            self.socket.bind(f"tcp://{self.host}:{self.port}")

            while self.running:
                try:
                    # Wait for request with timeout
                    if self.socket.poll(timeout=1000):  # 1 second timeout
                        message = self.socket.recv_json()
                        response = self._handle_request(message)
                        self.socket.send_json(response)

                except zmq.ZMQError as e:
                    if self.running:  # Only log if we're still supposed to be running
                        logger.error(f"ZMQ error: {e}")

        except Exception as e:
            logger.error(f"Server loop error: {e}")
        finally:
            self.socket.close()

        logger.info("Server loop stopped")

    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming ZMQ requests.

        Args:
            request: Request dictionary

        Returns:
            Response dictionary
        """
        try:
            command = request.get('command', '').upper()

            if command == 'PREDICT_GRASPS':
                # Extract parameters
                pointcloud_data = request.get('pointcloud', [])
                object_pose = request.get('object_pose', {})
                num_grasps = request.get('num_grasps', 5)

                if not pointcloud_data:
                    return {
                        'status': 'error',
                        'message': 'No pointcloud data provided'
                    }

                if not object_pose:
                    return {
                        'status': 'error',
                        'message': 'No object pose provided'
                    }

                # Convert data
                try:
                    pointcloud = np.array(pointcloud_data, dtype=np.float32)
                    object_pose_np = {
                        'pos': np.array(object_pose['pos']),
                        'quat': np.array(object_pose['quat'])
                    }
                except (KeyError, ValueError) as e:
                    return {
                        'status': 'error',
                        'message': f'Invalid data format: {e}'
                    }

                # Predict grasps
                grasps = self._predict_grasps_for_pointcloud(
                    pointcloud, object_pose_np, num_grasps
                )

                return {
                    'status': 'success',
                    'grasps': grasps,
                    'num_grasps_predicted': len(grasps),
                    'pointcloud_size': len(pointcloud)
                }

            elif command == 'PREDICT_BATCH_GRASPS':
                # Handle batch prediction for multiple objects
                objects_data = request.get('objects', [])
                num_grasps = request.get('num_grasps', 5)

                if not objects_data:
                    return {
                        'status': 'error',
                        'message': 'No objects data provided'
                    }

                batch_results = []
                total_grasps = 0

                for obj_data in objects_data:
                    pointcloud_data = obj_data.get('pointcloud', [])
                    object_pose = obj_data.get('pose', {})

                    if not pointcloud_data or not object_pose:
                        batch_results.append({
                            'status': 'error',
                            'message': 'Missing pointcloud or pose data'
                        })
                        continue

                    try:
                        pointcloud = np.array(pointcloud_data, dtype=np.float32)
                        object_pose_np = {
                            'pos': np.array(object_pose['pos']),
                            'quat': np.array(object_pose['quat'])
                        }

                        grasps = self._predict_grasps_for_pointcloud(
                            pointcloud, object_pose_np, num_grasps
                        )

                        batch_results.append({
                            'status': 'success',
                            'grasps': grasps,
                            'pointcloud_size': len(pointcloud)
                        })

                        total_grasps += len(grasps)

                    except Exception as e:
                        batch_results.append({
                            'status': 'error',
                            'message': str(e)
                        })

                return {
                    'status': 'success',
                    'batch_results': batch_results,
                    'total_objects': len(objects_data),
                    'total_grasps': total_grasps
                }

            elif command == 'GET_STATUS':
                return {
                    'status': 'success',
                    'service': 'GraspService',
                    'running': self.running,
                    'grasp_predictor_loaded': self.grasp_predictor is not None
                }

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

    def start(self) -> None:
        """Start the Grasp service."""
        if self.running:
            logger.warning("Service already running")
            return

        logger.info("Starting Grasp Service...")
        self.running = True

        # Register with service manager
        try:
            self.service_client.register(
                name="grasp_service",
                host=self.host,
                port=self.port,
                service_type="grasp_prediction",
                description="GraspGen grasp prediction from point clouds"
            )
            logger.info("Registered with service manager")
        except Exception as e:
            logger.warning(f"Failed to register with service manager: {e}")

        # Start server thread
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()

        logger.info("Grasp Service started successfully")

    def stop(self) -> None:
        """Stop the Grasp service."""
        if not self.running:
            return

        logger.info("Stopping Grasp Service...")
        self.running = False

        # Wait for server thread to finish
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)

        # Close ZMQ
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

        logger.info("Grasp Service stopped")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()