#!/usr/bin/env python3
"""
Grasp Service - Grasp prediction service for segmented point clouds.
"""

import zmq
import numpy as np
import threading
from typing import Dict, Any, Optional, cast, List, Tuple
import logging
import trimesh
import trimesh.transformations as tra

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment

from robosuite_franka_manipulation.perception.grasp_predictor import GraspPredictor
from robosuite_franka_manipulation.services.services_core import ServiceClient
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms import StampedPose, Transform, TransformManager
from robosuite_franka_manipulation.utils.pointcloud import validate_point_cloud
from robosuite_franka_manipulation.utils.env_utils import resolve_mesh_path
from robosuite_franka_manipulation.schema import GraspRequest, GraspResponse, ResponseStatus, GraspSchema

from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps

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
                 env_lock: Optional[threading.RLock] = None,
                 tf_manager: Optional[TransformManager] = None,
                 debug: bool = False,
                 visualize: bool = False):
        """
        Initialize Grasp Service.
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.visualize = visualize

        # Optional shared TF manager
        self.tf_manager: Optional[TransformManager] = tf_manager
        if self.tf_manager is None and env is not None:
            try:
                self.tf_manager = TransformManager(env=env, auto_discover=True, env_lock=env_lock)
            except Exception as e:
                logger.warning("Failed to initialize TransformManager for grasp visualization: %s", e)

        # Service state
        self.running = False

        # Initialize grasp predictors
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

        # ZMQ setup
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
        """Query the FastSAM service for the latest detections."""
        try:
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
                    if self.socket.poll(timeout=constants.ZMQ_POLL_TIMEOUT):
                        message = cast(Dict[str, Any], self.socket.recv_json())
                        response = self._handle_request(message)
                        self.socket.send_json(response)
                except zmq.ZMQError as e:
                    if self.running:
                        logger.error(f"ZMQ error: {e}")
        except Exception as e:
            logger.error(f"Server loop error: {e}")
        finally:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass
        logger.info("Server loop stopped")

    def _resolve_input_data(self, req: GraspRequest) -> Tuple[Optional[np.ndarray], Optional[StampedPose], Optional[str], Optional[str]]:
        """
        Resolve input data from request or FastSAM service.
        Returns: (pcd_np, object_pose_stamped, object_id, mesh_path)
        """
        pcd_data = req.pcd
        object_pose_stamped = req.object_pose
        object_name = req.object_name
        
        object_id: Optional[str] = object_name
        mesh_path: Optional[str] = None
        
        if object_name:
            mesh_path = resolve_mesh_path(object_name)
            if not mesh_path:
                logger.error(f"Could not resolve mesh for object {object_name}.")
                # Return partial data, caller handles missing mesh_path if strictly needed
        
        elif pcd_data is None or object_pose_stamped is None:
            logger.info("No pcd/pose/mesh provided, querying FastSAM service.")
            fastsam_data = self._query_fastsam_service()
            if fastsam_data and 'detections' in fastsam_data and len(fastsam_data['detections']) > 0:
                detection = fastsam_data['detections'][0]
                object_id = str(detection.get('id', detection.get('name', '0')))
                pcd_data = detection.get('pcd', [])
                pose_dict = detection.get('pose', {})
                try:
                    object_pose_stamped = StampedPose(**pose_dict)
                except Exception as e:
                    logger.error(f"Failed to parse pose from FastSAM detection: {e}")
                logger.info(f"Using detection from FastSAM: {len(pcd_data)} points")
        
        pcd_np = None
        if pcd_data:
            try:
                pcd_np = np.array(pcd_data, dtype=np.float32)
                if not self._validate_point_cloud(pcd_np):
                    logger.warning("Invalid point cloud provided")
                    # Should we invalidate pcd_np? keeping for now but logging
            except Exception as e:
                logger.warning(f"Error parsing point cloud: {e}")
                
        return pcd_np, object_pose_stamped, object_id, mesh_path

    def _transform_pose_to_world(self, pose: StampedPose) -> Optional[StampedPose]:
        """Transform pose to world frame if needed."""
        if pose.frame_id == constants.WORLD_FRAME:
            return pose
        
        if self.tf_manager:
            try:
                return self.tf_manager.transform_pose(pose, constants.WORLD_FRAME)
            except Exception as e:
                logger.error(f"Failed to transform pose: {e}")
                return None
        return None

    def _update_tf_manager(self, object_id: Optional[str], pose: StampedPose):
        """Update object frame in TF manager."""
        if self.tf_manager:
            try:
                if object_id:
                    self.tf_manager.update_object_frame(object_id=object_id, pose=pose)
                
                self.tf_manager.add_static_transform(
                    Transform(
                        translation=pose.position,
                        rotation=pose.quaternion,
                        parent_frame=pose.frame_id,
                        child_frame="object_local",
                        is_static=False,
                    )
                )
            except Exception as e:
                logger.debug("Failed to update TF: %s", e)

    def _visualize_grasps(self, grasps: List[GraspSchema]):
        """Publish grasp poses to TF manager for visualization."""
        if not self.tf_manager:
            return

        try:
            # We need to track indices per gripper type to generate unique names
            # assuming grasps come in a list, we can just use global index or type-based
            type_counters: Dict[str, int] = {}
            
            for grasp in grasps:
                g_type = grasp.gripper_type or "unknown"
                if g_type not in type_counters:
                    type_counters[g_type] = 0
                idx = type_counters[g_type]
                type_counters[g_type] += 1
                
                self.tf_manager.add_static_transform(
                    Transform(
                        translation=np.asarray(grasp.position, dtype=np.float64),
                        rotation=np.asarray(grasp.quaternion, dtype=np.float64),
                        parent_frame=constants.WORLD_FRAME,
                        child_frame=f"grasp_{g_type}_{idx}",
                        is_static=False,
                    )
                )
        except Exception as e:
            logger.debug("Failed to publish grasp frames: %s", e)

    def _collect_grasp_candidates(self,
                                  gripper_types: List[str],
                                  object_pose: StampedPose,
                                  pcd_np: Optional[np.ndarray],
                                  mesh_path: Optional[str],
                                  num_grasps: int) -> List[GraspSchema]:
        """Collect raw grasp predictions from all specified gripper types."""
        candidates = []
        for gripper_type_simple in gripper_types:
            full_type = constants.GRIPPER_TYPE_MAPPING.get(gripper_type_simple)
            # Fallback if mapping fails or simple type is already full type
            if not full_type:
                 full_type = gripper_type_simple

            if full_type not in self.grasp_predictors:
                logger.warning(f"Predictor not loaded for {gripper_type_simple}")
                continue
            
            predictor = self.grasp_predictors[full_type]
            logger.info(f"Predicting grasps for {gripper_type_simple}...")
            
            grasps = predictor.predict_grasps(
                object_pose=object_pose,
                num_grasps=num_grasps,
                object_pcd=pcd_np,
                mesh_file=mesh_path
            )
            if grasps:
                candidates.extend(grasps)
        return candidates

    def _is_grasp_colliding(self, 
                            grasp: GraspSchema, 
                            occupied_pcd_points: List[np.ndarray]) -> bool:
        """
        Check if a single grasp collides with the currently occupied space.
        Returns True if colliding (invalid), False otherwise.
        """
        if not occupied_pcd_points:
            return False # No collision if space is empty

        g_type = grasp.gripper_type
        if not g_type or g_type not in self.grasp_predictors:
            return True # Treat unknown type as invalid/colliding
        
        predictor = self.grasp_predictors[g_type]
        
        # Prepare pose matrix
        t = Transform(
            translation=grasp.position, 
            rotation=grasp.quaternion,
            parent_frame=grasp.frame_id,
            child_frame="candidate"
        )
        grasp_pose_mat = t.to_matrix()
        
        try:
            scene_occupied = np.concatenate(occupied_pcd_points, axis=0)
            gripper_collision_mesh = predictor.get_gripper_collision_mesh()
            
            # Check collision
            valid_mask = filter_colliding_grasps(
                scene_pc=scene_occupied,
                grasp_poses=np.array([grasp_pose_mat]),
                gripper_collision_mesh=gripper_collision_mesh,
                collision_threshold=constants.GRIPPER_COLLISION_THRESHOLD,
                num_collision_samples=constants.GRIPPER_COLLISION_MESH_SAMPLES
            )
            # valid_mask[0] is True if VALID (no collision).
            return not valid_mask[0] 
        except Exception as e:
            logger.warning(f"Collision check failed for grasp {grasp.id}: {e}")
            return True # Fail safe

    def _update_occupied_space(self, 
                               grasp: GraspSchema, 
                               occupied_pcd_points: List[np.ndarray]):
        """Sample points from the grasp's gripper mesh and add to occupied space."""
        g_type = grasp.gripper_type
        if not g_type or g_type not in self.grasp_predictors:
            return

        predictor = self.grasp_predictors[g_type]
        
        t = Transform(
            translation=grasp.position, 
            rotation=grasp.quaternion,
            parent_frame=grasp.frame_id,
            child_frame="candidate"
        )
        grasp_pose_mat = t.to_matrix()

        try:
            gripper_collision_mesh = predictor.get_gripper_collision_mesh()
            points, _ = trimesh.sample.sample_surface(gripper_collision_mesh, constants.GRIPPER_COLLISION_MESH_SAMPLES)
            points_transformed = tra.transform_points(points, grasp_pose_mat)
            occupied_pcd_points.append(points_transformed)
        except Exception as e:
            logger.warning(f"Failed to sample occupied points for grasp {grasp.id}: {e}")

    def _predict_and_filter_grasps(self, 
                                   gripper_types: List[str], 
                                   object_pose: StampedPose, 
                                   pcd_np: Optional[np.ndarray], 
                                   mesh_path: Optional[str], 
                                   num_grasps: int) -> List[GraspSchema]:
        """
        Core logic: Predict grasps for all gripper types, sort by score,
        and filter collisions against higher-scoring grasps (both inter- and intra-type).
        """
        # 1. Collect all predictions
        all_candidates = self._collect_grasp_candidates(
            gripper_types, object_pose, pcd_np, mesh_path, num_grasps
        )

        # 2. Sort by score (descending)
        all_candidates.sort(key=lambda g: g.score, reverse=True)
        
        # 3. Filter collisions sequentially
        final_grasps: List[GraspSchema] = []
        occupied_pcd_points: List[np.ndarray] = []
        
        logger.info(f"Filtering {len(all_candidates)} total candidates for collisions...")

        for grasp in all_candidates:
            # Check explicit type validity here or rely on helpers
            if not grasp.gripper_type or grasp.gripper_type not in self.grasp_predictors:
                logger.warning(f"Unknown gripper type {grasp.gripper_type}, skipping.")
                continue

            # Check collision
            if self._is_grasp_colliding(grasp, occupied_pcd_points):
                continue
            
            # Valid -> Add and Update
            final_grasps.append(grasp)
            self._update_occupied_space(grasp, occupied_pcd_points)

        logger.info(f"Final valid grasps: {len(final_grasps)}/{len(all_candidates)}")
        return final_grasps

    def _get_grasps(self, req: GraspRequest) -> GraspResponse:
        """Unified handler for grasp prediction requests."""
        # 1. Resolve Input Data
        pcd_np, object_pose_stamped, object_id, mesh_path = self._resolve_input_data(req)
        
        if pcd_np is None and mesh_path is None:
            return ResponseStatus(status='error', message='No valid point cloud or mesh available').model_dump()
        
        if not object_pose_stamped:
            return ResponseStatus(status='error', message='No object pose available').model_dump()
            
        # 2. Transform Pose to World
        object_pose_world = self._transform_pose_to_world(object_pose_stamped)
        if not object_pose_world:
             return ResponseStatus(status='error', message='Failed to transform pose to world frame').model_dump()
             
        # 3. Update TF Manager
        self._update_tf_manager(object_id, object_pose_world)
        
        # 4. Predict and Filter
        logger.debug(f"Predicting grasps with gripper types: {req.gripper_type}")
        all_grasps = self._predict_and_filter_grasps(
            gripper_types=req.gripper_type,
            object_pose=object_pose_world,
            pcd_np=pcd_np,
            mesh_path=mesh_path,
            num_grasps=req.num_grasps
        )
        
        # 6. Visualize
        if self.visualize:
            self._visualize_grasps(all_grasps)
        
        return GraspResponse(
            status='success',
            grasps=all_grasps,
            num_grasps_predicted=len(all_grasps),
            pcd_size=len(pcd_np) if pcd_np is not None else 0,
            gripper_type=req.gripper_type
        ).model_dump()

    def _handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming ZMQ requests."""
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
        
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        logger.info("Grasp Service started successfully")

    def stop(self) -> None:
        """Stop the Grasp service."""
        if not self.running:
            return

        logger.info("Stopping Grasp Service...")
        self.running = False

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=2.0)

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