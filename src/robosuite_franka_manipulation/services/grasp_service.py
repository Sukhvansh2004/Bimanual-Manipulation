#!/usr/bin/env python3
"""
Grasp Service - Grasp prediction service for segmented point clouds.
"""

import zmq
import numpy as np
import threading
import time
from typing import Dict, Any, List, Optional, cast
import logging

from robosuite.environments.base import MujocoEnv
from robosuite.wrappers import Wrapper
from typing import Union

from robosuite_franka_manipulation.perception.grasp_predictor import GraspPredictor
from robosuite_franka_manipulation.services.service_manager import ServiceClient
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms import StampedPose, Transform, TransformManager
from robosuite_franka_manipulation.utils.serialization import convert_grasps_to_json
from robosuite_franka_manipulation.utils.pointcloud import validate_point_cloud

# Configure logging
logging.basicConfig(level=constants.LOG_LEVEL)
logger = logging.getLogger(__name__)


class GraspService:
    """
    Grasp Service for predicting grasps from 3D point clouds.

    This service takes segmented point clouds (typically from FastSAM service)
    and predicts 6-DOF grasps using the GraspGen model. It provides grasp
    ranking, confidence scoring, and coordinate transformations.
    """

    def __init__(self,
                 host: str = constants.DEFAULT_HOST,
                 port: int = constants.GRASP_SERVICE_PORT,
                 manager_host: str = constants.DEFAULT_HOST,
                 manager_port: int = constants.SERVICE_MANAGER_PORT,
                 env: Optional[Union[MujocoEnv, Wrapper]] = None,
                 env_lock: Optional[threading.Lock] = None,
                 tf_manager: Optional[TransformManager] = None,
                 debug: bool = False,
                 visualize: bool = False):
        """
        Initialize Grasp Service.

        Args:
            host: Service host address
            port: Service port
            manager_host: Service manager host
            manager_port: Service manager port
            debug: Enable debug logging
            visualize: Enable grasp visualization
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.visualize = visualize

        # Optional shared TF manager (preferred). If not provided, create one when env is available.
        self.tf_manager: Optional[TransformManager] = tf_manager
        if self.tf_manager is None and env is not None:
            try:
                self.tf_manager = TransformManager(env=env, auto_discover=True, env_lock=env_lock)
            except Exception as e:
                logger.warning("Failed to initialize TransformManager for grasp visualization: %s", e)

        # Service state
        self.running = False

        # Initialize grasp predictors for different gripper types
        self.grasp_predictors: Dict[str, GraspPredictor] = {}
        gripper_types = constants.GRIPPER_TYPES
        
        for gripper_type in gripper_types:
            try:
                self.grasp_predictors[gripper_type] = GraspPredictor(
                    gripper_type=gripper_type, debug=debug, visualize=visualize, transform_manager=self.tf_manager
                )
                logger.info(f"Loaded GraspPredictor for {gripper_type}")
            except Exception as e:
                logger.warning(f"Failed to load GraspPredictor for {gripper_type}: {e}")

        # Threading
        self.server_thread: Optional[threading.Thread] = None

        # ZMQ setup for service
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        # Service registration
        self.service_client = ServiceClient(manager_host, manager_port)

        logger.info(f"Grasp Service initialized on {host}:{port}")

        if self.debug:
            logger.setLevel(logging.DEBUG)

    def _validate_pointcloud(self, pointcloud: np.ndarray) -> bool:
        """Validate point cloud data using utility function."""
        return validate_point_cloud(pointcloud, max_range=constants.MAX_POINTCLOUD_RANGE)

    def _predict_grasps_for_pointcloud(self, pointcloud: np.ndarray,
                                     object_pose: StampedPose,
                                     num_grasps: int = constants.DEFAULT_NUM_GRASPS,
                                     gripper_type: str = 'finger') -> List[Dict]:
        """
        Predict grasps for a single point cloud.

        Args:
            pointcloud: Nx3 point cloud array
            object_pose: Object pose in world frame
            num_grasps: Number of grasps to predict
            gripper_type: Type of gripper ('finger' or 'suction')

        Returns:
            List of grasp dictionaries
        """
        # Select predictor based on gripper type
        if gripper_type == 'suction':
            predictor = self.grasp_predictors[constants.GRIPPER_TYPE_MAPPING['suction']]
        elif gripper_type == 'finger':
            predictor = self.grasp_predictors[constants.GRIPPER_TYPE_MAPPING['finger']]
        else:
            logger.error(f"Unknown gripper type: {gripper_type}")
            return []
        
        try:
            if not self._validate_pointcloud(pointcloud):
                logger.warning("Invalid point cloud provided")
                return []

            # Predict grasps using GraspGen
            grasps = predictor.predict_grasps(
                object_pcd=pointcloud,
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

            logger.debug(f"Predicted {len(grasps)} grasps for point cloud with {len(pointcloud)} points")

            return grasps

        except Exception as e:
            logger.error(f"Grasp prediction failed: {e}")
            return []

    def _query_fastsam_service(self) -> Optional[Dict[str, Any]]:
        """
        Query the FastSAM service for the latest detections via service manager.
        
        Returns:
            FastSAM response dictionary or None if failed
        """
        try:
            # Query FastSAM service through service manager
            response = self.service_client.call('fastsam_service', {'command': 'GET_LATEST_DETECTION'})
            
            if response.get('status') == 'success':
                return response
            else:
                logger.warning(f"FastSAM service error: {response.get('message', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to query FastSAM service: {e}")
            return None

    def _server_loop(self) -> None:
        """ZMQ server loop for handling client requests."""
        logger.info(f"Starting ZMQ server on tcp://{self.host}:{self.port}")

        try:
            self.socket.bind(f"tcp://{self.host}:{self.port}")

            while self.running:
                try:
                    # Wait for request with timeout
                    if self.socket.poll(timeout=constants.ZMQ_POLL_TIMEOUT):
                        message = cast(Dict[str, Any], self.socket.recv_json())
                        response = self._handle_request(message)
                        self.socket.send_json(response)

                except zmq.ZMQError as e:
                    if self.running:  # Only log if we're still supposed to be running
                        logger.error(f"ZMQ error: {e}")

        except Exception as e:
            logger.error(f"Server loop error: {e}")
        finally:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass

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

            if command == 'GET_GRASPS':
                # Extract parameters - gripper_type should be simplified ('finger' or 'suction')
                simplified_type = request.get('gripper_type', 'finger')
                num_grasps = request.get('num_grasps', constants.DEFAULT_NUM_GRASPS)
                
                # Validate simplified gripper type
                if simplified_type not in ['finger', 'suction']:
                    return {
                        'status': 'error',
                        'message': f'Invalid gripper type: {simplified_type}. Use "finger" or "suction"'
                    }
                
                # Map simplified type to full gripper type for predictor lookup
                if simplified_type == 'suction':
                    full_gripper_type = constants.GRIPPER_TYPE_MAPPING['suction']
                else:  # finger
                    full_gripper_type = constants.GRIPPER_TYPE_MAPPING['finger']
                
                # Check if gripper type predictor is loaded
                if full_gripper_type not in self.grasp_predictors:
                    return {
                        'status': 'error',
                        'message': f'Grasp predictor for {simplified_type} gripper not loaded'
                    }
                
                # Check if pointcloud and pose are provided
                pointcloud_data = request.get('pointcloud')
                object_pose = request.get('object_pose')
                
                object_id: Optional[str] = None

                # If not provided, query FastSAM service
                if pointcloud_data is None or object_pose is None:
                    logger.info("No pointcloud/pose provided, querying FastSAM service")
                    fastsam_data = self._query_fastsam_service()
                    if fastsam_data and 'detections' in fastsam_data and len(fastsam_data['detections']) > 0:
                        # Use the first detection
                        detection = fastsam_data['detections'][0]
                        object_id = str(detection.get('id', detection.get('name', '0')))
                        pointcloud_data = detection.get('pcd', [])
                        object_pose = detection.get('pose', {})
                        logger.info(f"Using detection from FastSAM: {len(pointcloud_data)} points")
                    else:
                        return {
                            'status': 'error',
                            'message': 'No pointcloud/pose provided and FastSAM service returned no detections'
                        }

                if not pointcloud_data:
                    return {
                        'status': 'error',
                        'message': 'No pointcloud data available'
                    }

                if not object_pose:
                    return {
                        'status': 'error',
                        'message': 'No object pose available'
                    }

                # Convert data
                try:
                    pointcloud = np.array(pointcloud_data, dtype=np.float32)
                    object_pose_stamped = StampedPose(
                        position=np.array(object_pose['pos']),
                        quaternion=np.array(object_pose['quat']),
                        frame_id='world'
                    )
                except (KeyError, ValueError) as e:
                    return {
                        'status': 'error',
                        'message': f'Invalid data format: {e}'
                    }

                # Ensure object frames are registered in the shared TF manager.
                # This fixes missing `object_local` and keeps TF visualization consistent.
                if self.tf_manager is not None:
                    try:
                        # Register/update object_<id> frame if we know an id
                        if object_id is not None:
                            self.tf_manager.update_object_frame(object_id=object_id, pose=object_pose_stamped)

                        # Register/update the canonical object-local frame used by GraspGen
                        self.tf_manager.add_static_transform(
                            Transform(
                                translation=object_pose_stamped.position,
                                rotation=object_pose_stamped.quaternion,
                                parent_frame=object_pose_stamped.frame_id,
                                child_frame="object_local",
                                is_static=False,
                            )
                        )
                    except Exception as e:
                        logger.debug("Failed to register object frames in TF manager: %s", e)

                # Predict grasps using the appropriate predictor
                grasps = self._predict_grasps_for_pointcloud(
                    pointcloud, object_pose_stamped, num_grasps, simplified_type
                )

                # Publish grasp poses as TF frames for MeshCat / TF debugging.
                # These frames are in world coordinates (consistent with returned grasp poses).
                if self.tf_manager is not None:
                    try:
                        for i, grasp in enumerate(grasps):
                            pos = grasp.get("pos")
                            quat = grasp.get("quat")
                            if pos is None or quat is None:
                                continue
                            self.tf_manager.add_static_transform(
                                Transform(
                                    translation=np.asarray(pos, dtype=np.float64),
                                    rotation=np.asarray(quat, dtype=np.float64),
                                    parent_frame="world",
                                    child_frame=f"grasp_{i}",
                                    is_static=False,
                                )
                            )
                    except Exception as e:
                        logger.debug("Failed to publish grasp frames to TF manager: %s", e)

                # Convert grasps to JSON-serializable format using utility
                json_grasps = convert_grasps_to_json(grasps)

                return {
                    'status': 'success',
                    'grasps': json_grasps,
                    'num_grasps_predicted': len(grasps),
                    'pointcloud_size': len(pointcloud),
                    'gripper_type': simplified_type
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
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
        if self.context:
            try:
                self.context.term()
            except Exception:
                pass

        logger.info("Grasp Service stopped")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop()