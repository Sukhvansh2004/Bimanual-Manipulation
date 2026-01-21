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
from robosuite_franka_manipulation.services.services_core import ServiceClient
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms import StampedPose, Transform, TransformManager
from robosuite_franka_manipulation.utils.pointcloud import validate_point_cloud
from robosuite_franka_manipulation.utils.env_utils import resolve_mesh_path
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
        """
        Unified handler for grasp prediction requests.
        Handles both point cloud input and mesh/object name input.
        """
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
        
        # Check available inputs
        pcd_data = req.pcd
        object_pose_stamped = req.object_pose
        object_name = req.object_name
        
        object_id: Optional[str] = object_name
        mesh_path: Optional[str] = None
        
        # Scenario 1: Object Name provided (try to find mesh)
        if object_name:
            mesh_path = resolve_mesh_path(object_name)
            if not mesh_path:
                logger.error(f"Could not resolve mesh for object {object_name}.")
                return ResponseStatus(
                    status='error',
                    message=f'Could not resolve mesh for object {object_name}'
                ).model_dump()

        # Scenario 2: No inputs, query FastSAM
        elif pcd_data is None or object_pose_stamped is None:
            logger.info("No pcd/pose/mesh provided, querying FastSAM service for pcd and object pose.")
            fastsam_data = self._query_fastsam_service()
            if fastsam_data and 'detections' in fastsam_data and len(fastsam_data['detections']) > 0:
                # Use the first detection
                detection = fastsam_data['detections'][0]
                object_id = str(detection.get('id', detection.get('name', '0')))
                pcd_data = detection.get('pcd', [])
                
                # Parse pose from detection
                pose_dict = detection.get('pose', {})
                try:
                    object_pose_stamped = StampedPose(**pose_dict)
                except Exception as e:
                    logger.error(f"Failed to parse pose from FastSAM detection: {e}")
                
                logger.info(f"Using detection from FastSAM: {len(pcd_data)} points")
            else:
                logger.error("FastSAM service returned no detections")
                return ResponseStatus(
                    status='error',
                    message='No pcd/pose provided and FastSAM service returned no detections'
                ).model_dump()
            
        # Scenario 3: PCD and Pose provided directly - use as is
        else:
            logger.info("Using provided point cloud and object pose for grasp prediction. Ensure the pcd is in world frame for attaining valid grasps.") 

        if not object_pose_stamped:
            return ResponseStatus(status='error', message='No object pose available').model_dump()

        # Enforce World Frame
        if object_pose_stamped.frame_id != constants.WORLD_FRAME:
            if self.tf_manager:
                try:
                    # Transform pose to world frame
                    object_pose_stamped = self.tf_manager.transform_pose(
                        object_pose_stamped, 
                        constants.WORLD_FRAME
                    )
                except Exception as e:
                    return ResponseStatus(
                        status='error', 
                        message=f'Failed to transform pose to world frame: {e}'
                    ).model_dump()
            else:
                return ResponseStatus(
                    status='error', 
                    message=f'Object pose is in {object_pose_stamped.frame_id}, but needs to be in {constants.WORLD_FRAME} and no TransformManager is available.'
                ).model_dump()

        # Parse Point Cloud (if available and needed)
        pcd_np = None
        if pcd_data:
            try:
                pcd_np = np.array(pcd_data, dtype=np.float32)
                if not self._validate_point_cloud(pcd_np):
                     logger.warning("Invalid point cloud provided")
                     pcd_np = None # Fallback or fail?
            except Exception as e:
                logger.warning(f"Error parsing point cloud: {e}")

        if pcd_np is None and mesh_path is None:
             return ResponseStatus(status='error', message='No valid point cloud or mesh available').model_dump()

        # Update TF Manager
        if self.tf_manager is not None:
            try:
                if object_id:
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

        # Predict Grasps
        try:
            predictor = self.grasp_predictors[full_gripper_type]
            grasps = predictor.predict_grasps(
                object_pose=object_pose_stamped,
                num_grasps=num_grasps,
                object_pcd=pcd_np,
                mesh_file=mesh_path
            )

            # Publish grasp poses as TF frames
            if self.tf_manager is not None:
                try:
                    for i, grasp in enumerate(grasps):
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

            return GraspResponse(
                status='success',
                grasps=grasps,
                num_grasps_predicted=len(grasps),
                pcd_size=len(pcd_np) if pcd_np is not None else 0,
                gripper_type=simplified_type
            ).model_dump()

        except Exception as e:
            logger.error(f"Grasp prediction failed: {e}")
            return ResponseStatus(status='error', message=str(e)).model_dump()
        
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
        response = self.service_client.register(
            name="grasp_service",
            host=self.host,
            port=self.port,
            service_type="grasp_prediction",
            description="GraspGen grasp prediction from point clouds"
        )
        
        if response.get('status') != 'success':
            logger.warning("Failed to register with service manager: %s", response.get('message'))
            self.running = False
            return
        
        logger.info("Registered with service manager")

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
