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

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment

from robosuite_franka_manipulation.perception.grasp_predictor import GraspPredictor
from robosuite_franka_manipulation.services.service_manager import ServiceClient
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms import StampedPose, Transform, TransformManager
from robosuite_franka_manipulation.utils.pointcloud import validate_point_cloud
from robosuite_franka_manipulation.schema import GraspRequest, GraspResponse, GraspSchema, ResponseStatus

# Configure logging
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
                 env: Optional[DualArmEnvironment] = None,
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
        gripper_types = constants.GRIPPER_TYPE_MAPPING.values()
        
        for gripper_type in gripper_types:
            try:
                self.grasp_predictors[gripper_type] = GraspPredictor(
                    gripper_type=gripper_type, debug=debug, visualize=visualize
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

    def _validate_point_cloud(self, pcd: np.ndarray) -> bool:
        """Validate point cloud data using utility function."""
        return validate_point_cloud(pcd, max_range=constants.MAX_POINTCLOUD_RANGE)

    def _predict_grasps_for_pointcloud(self, pcd: np.ndarray,
                                     object_pose: StampedPose,
                                     num_grasps: int = constants.DEFAULT_NUM_GRASPS,
                                     gripper_type: str = constants.GRIPPER_TYPES[0]) -> List[GraspSchema]:
        """
        Predict grasps for a single point cloud.

        Args:
            pcd: Nx3 point cloud array
            object_pose: Object pose in world frame
            num_grasps: Number of grasps to predict
            gripper_type: Type of gripper ('finger' or 'suction')

        Returns:
            List of grasp dictionaries
        """
        # Select predictor based on gripper type
        if gripper_type in constants.GRIPPER_TYPE_MAPPING:
            full_gripper_type = constants.GRIPPER_TYPE_MAPPING[gripper_type]
            predictor = self.grasp_predictors.get(full_gripper_type)
        else:
            logger.error(f"Unknown gripper type: {gripper_type}")
            return []
        
        try:
            if not self._validate_point_cloud(pcd):
                logger.warning("Invalid point cloud provided")
                return []

            # Predict grasps using GraspGen
            grasps = predictor.predict_grasps(
                object_pcd=pcd,
                object_pose=object_pose,
                num_grasps=num_grasps
            )

            if not grasps:
                logger.info("No grasps predicted for point cloud")
                return []
            
            logger.debug(f"Predicted {len(grasps)} grasps for point cloud with {len(pcd)} points")

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

    def _get_grasps(self, req: GraspRequest) -> GraspResponse:
        simplified_type = req.gripper_type
        num_grasps = req.num_grasps
        
        # Validate simplified gripper type
        if simplified_type not in constants.GRIPPER_TYPE_MAPPING:
            return ResponseStatus(
                status='error', 
                message=f'Invalid gripper type: {simplified_type}. Use amongst {constants.GRIPPER_TYPES}'
            ).model_dump()
        
        # Map simplified type to full gripper type for predictor lookup
        full_gripper_type = constants.GRIPPER_TYPE_MAPPING[simplified_type]
        
        # Check if gripper type predictor is loaded
        if full_gripper_type not in self.grasp_predictors:
            return ResponseStatus(
                status='error',
                message=f'Grasp predictor for {simplified_type} gripper not loaded'
            ).model_dump()
        
        # Check if pcd and pose are provided
        pcd_data = req.pcd
        object_pose_data = req.object_pose
        
        object_id: Optional[str] = None

        # If not provided, query FastSAM service
        if pcd_data is None or object_pose_data is None:
            logger.info("No pcd/pose provided, querying FastSAM service")
            fastsam_data = self._query_fastsam_service()
            if fastsam_data and 'detections' in fastsam_data and len(fastsam_data['detections']) > 0:
                # Use the first detection
                detection = fastsam_data['detections'][0]
                object_id = str(detection.get('id', detection.get('name', '0')))
                pcd_data = detection.get('pcd', [])
                object_pose_data = detection.get('pose', {})
                logger.info(f"Using detection from FastSAM: {len(pcd_data)} points")
            else:
                return ResponseStatus(
                    status='error',
                    message='No pcd/pose provided and FastSAM service returned no detections'
                ).model_dump()

        if not pcd_data:
            return ResponseStatus(status='error', message='No pcd data available').model_dump()

        if not object_pose_data:
            return ResponseStatus(status='error', message='No object pose available').model_dump()

        # Convert data
        try:
            pcd = np.array(pcd_data, dtype=np.float32)
            
            # Handle StampedPose input (dict or StampedPose object)
            if isinstance(object_pose_data, dict):
                object_pose_stamped = StampedPose.from_dict(object_pose_data, frame_id=constants.WORLD_FRAME)
            elif isinstance(object_pose_data, StampedPose):
                object_pose_stamped = object_pose_data
            else:
                return ResponseStatus(status='error', message='Invalid pose format').model_dump()
                
        except (KeyError, ValueError, AttributeError) as e:
            return ResponseStatus(
                status='error',
                message=f'Invalid data format: {e}'
            ).model_dump()

        # Ensure object frames are registered in the shared TF manager.
        if self.tf_manager is not None:
            try:
                if object_id is not None:
                    self.tf_manager.update_object_frame(object_id=object_id, pose=object_pose_stamped)

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
            pcd, object_pose_stamped, num_grasps, simplified_type
        )

        # Publish grasp poses as TF frames for MeshCat / TF debugging.
        if self.tf_manager is not None:
            try:
                for i, grasp in enumerate(grasps):
                    # grasp is GraspSchema which inherits StampedPose
                    pos = grasp.position
                    quat = grasp.quaternion
                    if pos is None or quat is None:
                        continue
                    self.tf_manager.add_static_transform(
                        Transform(
                            translation=np.asarray(pos, dtype=np.float64),
                            rotation=np.asarray(quat, dtype=np.float64),
                            parent_frame=constants.WORLD_FRAME,
                            child_frame=f"grasp_{i}",
                            is_static=False,
                        )
                    )
            except Exception as e:
                logger.debug("Failed to publish grasp frames to TF manager: %s", e)

        # grasps is already List[GraspSchema]
        return GraspResponse(
            status='success',
            grasps=grasps,
            num_grasps_predicted=len(grasps),
            pcd_size=len(pcd),
            gripper_type=simplified_type
        ).model_dump()
        
    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming ZMQ requests.

        Args:
            request: Request dictionary

        Returns:
            Response dictionary
        """
        try:
            req = GraspRequest(**request)
        except Exception as e:
            return ResponseStatus(status="error", message=str(e)).model_dump()

        command = req.command.upper()

        if command == 'GET_GRASPS':
            return self._get_grasps(req)
        else:
            return ResponseStatus(
                status='error', 
                message=f'Unknown command: {command}'
            ).model_dump()

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