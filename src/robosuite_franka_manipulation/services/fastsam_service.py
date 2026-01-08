#!/usr/bin/env python3
"""
FastSAM Service - Continuous object detection service.
"""

import zmq
import numpy as np
import threading
import time
from typing import Dict, Any, List, Optional, Tuple, cast, Union
import logging

from robosuite.utils.camera_utils import get_real_depth_map

from scipy.spatial.transform import Rotation as R
from robosuite.environments.base import MujocoEnv
from robosuite.wrappers import Wrapper

from robosuite_franka_manipulation.perception.object_detector import ObjectDetector
from robosuite_franka_manipulation.services.service_manager import ServiceClient
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.utils.serialization import convert_detections_to_json
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose

# Configure logging
logging.basicConfig(level=constants.LOG_LEVEL)
logger = logging.getLogger(__name__)


class FastSAMService:
    """
    FastSAM Service for continuous object detection with camera visualization.

    This service continuously processes camera feeds from robosuite environment,
    runs FastSAM segmentation, and provides visualization with detection overlays.
    Supports dynamic prompt updates and provides segmented point clouds for grasp planning.
    """

    # Optional shared environment lock, set per instance
    env_lock: Optional[threading.Lock] = None

    def __init__(self,
                 env: Union[MujocoEnv, Wrapper],
                 host: str = constants.DEFAULT_HOST,
                 port: int = constants.FASTSAM_SERVICE_PORT,
                 manager_host: str = constants.DEFAULT_HOST,
                 manager_port: int = constants.SERVICE_MANAGER_PORT,
                 camera_name: str = constants.DEFAULT_CAMERA_NAME,
                 visualize: bool = True,
                 debug: bool = False,
                 env_lock: Optional[threading.Lock] = None,
                 tf_manager: Optional[TransformManager] = None):
        """
        Initialize FastSAM Service.

        Args:
            env: Robosuite environment instance
            host: Service host address
            port: Service port
            manager_host: Service manager host
            manager_port: Service manager port
            camera_name: Name of camera to use for detection
            visualize: Whether to show OpenCV visualization windows
            debug: Enable debug logging
            env_lock: Optional threading lock for environment access
        """
        self.env = env
        self.host = host
        self.port = port
        self.camera_name = camera_name
        self.visualize = visualize
        self.debug = debug
        self.env_lock = env_lock
        self.tf_manager = tf_manager

        # Service state
        self.running = False
        self.current_prompt = constants.DEFAULT_DETECTION_PROMPT

        # Camera and detection state
        self.camera_intrinsics = None
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_detections = []
        self.latest_segmented_pointclouds = []

        # Get camera intrinsics
        if self.camera_name in self.env.camera_names:
            logger.info(f"Using camera '{self.camera_name}' for FastSAM detection")
            camera_id = self.env.camera_names.index(self.camera_name)
            self.camera_intrinsics = self.env.sim.model.cam_intrinsic[camera_id]
            self.camera_width = self.env.camera_widths[camera_id]
            self.camera_height = self.env.camera_heights[camera_id]
        else:
            raise(ValueError(f"Camera '{self.camera_name}' not found in environment"))

        logger.info(f"Camera intrinsics: {self.camera_intrinsics}")
        logger.info(f"Camera resolution: {self.camera_width}x{self.camera_height}")

        # Initialize perception components
        self.object_detector = ObjectDetector(
            camera_intrinsics=self.camera_intrinsics,
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            prompt=self.current_prompt,
            debug=debug,
            visualize=visualize
        )

        # Threading
        self.detection_thread: Optional[threading.Thread] = None
        self.server_thread: Optional[threading.Thread] = None

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        # Service registration
        self.service_client = ServiceClient(manager_host, manager_port)

        # Elevate logger level if debug enabled
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        logger.info(f"FastSAM Service initialized on {host}:{port}")

    def _normalize_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """Convert depth to 2D array. Raises error if shape is invalid."""
        if depth_image is None:
            raise ValueError("Depth image is None")
        
        depth = np.asarray(depth_image)
        if depth.ndim == 1:
            expected = self.camera_height * self.camera_width
            if depth.size != expected:
                raise ValueError(f"Flat depth buffer size {depth.size} != expected {expected}")
            depth = depth.reshape(self.camera_height, self.camera_width)
        
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        
        if depth.ndim != 2:
            raise ValueError(f"Depth has invalid shape {depth.shape}, expected 2D")
        
        return depth

    def _get_camera_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract RGB and depth images from robosuite environment.

        Returns:
            Tuple of (rgb_image, depth_image) or (None, None) if unavailable
        """
        # Acquire lock if available to synchronize env access
        if self.env_lock:
            self.env_lock.acquire()
        
        try:
            obs = self.env._get_observations()

            # Get RGB image
            rgb_key = f"{self.camera_name}_image"
            rgb_image = obs.get(rgb_key)

            # Get depth image and normalize shape
            depth_key = f"{self.camera_name}_depth"
            depth_image = self._normalize_depth(obs.get(depth_key))

            # Robosuite returns OpenGL-style depth buffers (typically near ~1.0).
            # Convert to real depth in meters for correct 3D geometry.
            try:
                depth_image = get_real_depth_map(self.env.sim, depth_image)
            except Exception as e:
                logger.debug("Depth conversion via get_real_depth_map failed: %s", e)

            if rgb_image is not None:
                # Convert from (H, W, C) to (H, W, C) if needed
                if rgb_image.shape[-1] == 3:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                else:
                    rgb_image = np.transpose(rgb_image, (1, 2, 0))
                    rgb_image = (rgb_image * 255).astype(np.uint8)

            return rgb_image, depth_image
        finally:
            if self.env_lock:
                self.env_lock.release()

    def _process_detections(self, rgb_image: np.ndarray, depth_image: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process RGB image through FastSAM and extract detections with point clouds.

        Args:
            rgb_image: RGB image from camera
            depth_image: Depth image from camera (required)

        Returns:
            List of detection dictionaries with masks, point clouds, etc.
        """
        if depth_image is None:
            raise ValueError("Depth image is required for object detection")
        
        # Normalize depth to 2D
        depth_image = self._normalize_depth(depth_image)
        logger.debug(f"Depth shape after normalization: {depth_image.shape}, ndim={depth_image.ndim}")

        # Get camera pose (guarded by env_lock to avoid concurrent MuJoCo access)
        if self.env_lock:
            with self.env_lock:
                cam_pos = self.env.sim.data.get_camera_xpos(self.camera_name)
                cam_rot_mat = self.env.sim.data.get_camera_xmat(self.camera_name).reshape(3, 3)
        else:
            cam_pos = self.env.sim.data.get_camera_xpos(self.camera_name)
            cam_rot_mat = self.env.sim.data.get_camera_xmat(self.camera_name).reshape(3, 3)
        cam_rot = R.from_matrix(cam_rot_mat)

        # Run object detection
        detections = self.object_detector.detect_objects(rgb_image, depth_image, cam_pos, cam_rot)

        # Enhance detections with point clouds
        result: List[Dict[str, Any]] = []
        for detection in detections:
            det_dict: Dict[str, Any] = dict(detection)  # Convert TypedDict to regular dict
            if 'mask' in det_dict:
                # Extract point cloud from depth and mask
                image_shape: Tuple[int, int] = (rgb_image.shape[0], rgb_image.shape[1])
                pointcloud = self._extract_pointcloud_from_depth(
                    depth_image, det_dict['mask'], image_shape
                )
                det_dict['pointcloud'] = pointcloud
            else:
                det_dict['pointcloud'] = np.array([])
            result.append(det_dict)

        return result

    def _get_intrinsic_params(self) -> Tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy) from stored intrinsics. Raises error if intrinsics invalid."""
        if self.camera_intrinsics is None:
            raise ValueError("Camera intrinsics not available")

        I = np.asarray(self.camera_intrinsics)

        # Handle 1D array: [fx, fy, cx, cy]
        if I.ndim == 1:
            if I.shape[0] != 4:
                raise ValueError(f"1D intrinsics array has invalid length {I.shape[0]}, expected 4 [fx, fy, cx, cy]")
            fx, fy, cx, cy = float(I[0]), float(I[1]), float(I[2]), float(I[3])
        # Handle 2D matrix: standard 3x3 or 4x4
        elif I.ndim == 2:
            if I.shape[0] < 2 or I.shape[1] < 3:
                raise ValueError(f"Invalid intrinsics matrix shape {I.shape}, expected at least (2, 3)")
            fx, fy, cx, cy = float(I[0, 0]), float(I[1, 1]), float(I[0, 2]), float(I[1, 2])
        else:
            raise ValueError(f"Invalid intrinsics shape {I.shape}, expected 1D or 2D array")

        logger.debug(f"Using intrinsics fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        return fx, fy, cx, cy

    def _extract_pointcloud_from_depth(self, depth_image: np.ndarray,
                                     mask: np.ndarray,
                                     image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract 3D point cloud from depth image and segmentation mask.

        Args:
            depth_image: Depth image from camera (must be 2D)
            mask: Binary segmentation mask
            image_shape: Shape of RGB image (H, W)

        Returns:
            Nx3 array of 3D points in camera frame
        """
        # Ensure depth is a 2D array
        depth = np.asarray(depth_image)
        if depth.ndim == 1:
            expected = self.camera_height * self.camera_width
            if depth.size != expected:
                raise ValueError(f"Flat depth buffer size {depth.size} != expected {expected}")
            depth = depth.reshape(self.camera_height, self.camera_width)
        
        if depth.ndim != 2:
            raise ValueError(f"Depth has invalid shape {depth.shape}, expected 2D")

        logger.debug(f"Pointcloud extraction depth shape={depth.shape}, mask shape={mask.shape}")

        # Get camera intrinsics from environment
        fx, fy, cx, cy = self._get_intrinsic_params()

        # Find masked pixels
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) == 0:
            return np.array([])

        # Get depth values for masked pixels
        y_coords, x_coords = mask_indices
        depth_values = depth[y_coords, x_coords]

        # Filter out invalid depth values
        valid_mask = (depth_values > 0) & (depth_values < constants.MAX_POINTCLOUD_RANGE)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        depth_values = depth_values[valid_mask]

        if len(depth_values) == 0:
            return np.array([])

        # Convert to 3D points (camera coordinates)
        z = depth_values
        x = (x_coords - cx) * z / fx
        y = (y_coords - cy) * z / fy

        # Stack into Nx3 array
        points_3d = np.column_stack([x, y, z])

        return points_3d.astype(np.float32)

    def _continuous_detection_loop(self) -> None:
        """Main detection loop that continuously processes camera feed."""
        logger.info("Starting continuous detection loop")

        while self.running:
            # Get camera data
            rgb_image, depth_image = self._get_camera_data()

            if rgb_image is not None:
                # Store latest images
                self.latest_rgb_image = rgb_image
                self.latest_depth_image = depth_image

                # Process detections
                detections = self._process_detections(rgb_image, depth_image)
                self.latest_detections = detections

                # Register/update object frames in the shared TF manager (if provided)
                if self.tf_manager is not None:
                    for det in detections:
                        try:
                            object_id = str(det.get('id', det.get('name', '0')))
                            pose = det.get('pose')
                            if pose is None:
                                continue
                            if isinstance(pose, StampedPose):
                                pose_sp = pose
                            else:
                                pose_sp = StampedPose(
                                    position=np.array(pose['pos']),
                                    quaternion=np.array(pose['quat']),
                                    frame_id='world',
                                )
                            self.tf_manager.update_object_frame(object_id=object_id, pose=pose_sp)
                        except Exception:
                            continue

                # Extract point clouds for all detections
                self.latest_segmented_pointclouds = [
                    detection.get('pointcloud', np.array([]))
                    for detection in detections
                ]

                logger.debug(f"Processed {len(detections)} detections")

            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)

        logger.info("Detection loop stopped")

    def _server_loop(self) -> None:
        """ZMQ server loop for handling client requests."""
        logger.info(f"Starting ZMQ server on tcp://{self.host}:{self.port}")

        self.socket.bind(f"tcp://{self.host}:{self.port}")

        while self.running:
            # Wait for request with timeout
            if self.socket.poll(timeout=constants.ZMQ_POLL_TIMEOUT):
                message = cast(Dict[str, Any], self.socket.recv_json())
                response = self._handle_request(message)
                self.socket.send_json(response)

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
        command = request.get('command', '').upper()

        if command == 'SET_PROMPT':
            new_prompt = request.get('prompt', 'object')
            self.current_prompt = new_prompt
            self.object_detector.set_prompt(new_prompt)
            return {
                'status': 'success',
                'message': f'Prompt updated to: {new_prompt}'
            }

        elif command == 'GET_PROMPT':
            return {
                'status': 'success',
                'prompt': self.current_prompt
            }

        elif command == 'GET_LATEST_DETECTION':
            # Convert detections to JSON-serializable format using utility
            json_detections = convert_detections_to_json(self.latest_detections)
            return {
                'status': 'success',
                'detections': json_detections,
                'timestamp': time.time()
            }

        else:
            return {
                'status': 'error',
                'message': f'Unknown command: {command}'
            }

    def start(self) -> None:
        """Start the FastSAM service."""
        if self.running:
            logger.warning("Service already running")
            return

        logger.info("Starting FastSAM Service...")
        self.running = True

        # Register with service manager
        self.service_client.register(
            name="fastsam_service",
            host=self.host,
            port=self.port,
            service_type="detection",
            description="FastSAM object detection with camera feed"
        )
        logger.info("Registered with service manager")

        # Start threads
        self.detection_thread = threading.Thread(target=self._continuous_detection_loop, daemon=True)
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)

        self.detection_thread.start()
        self.server_thread.start()

        logger.info("FastSAM Service started successfully")

    def stop(self) -> None:
        """Stop the FastSAM service."""
        if not self.running:
            return

        logger.info("Stopping FastSAM Service...")
        self.running = False

        # Wait for threads to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)

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

        # Close any OpenCV visualization windows owned by the detector
        try:
            if hasattr(self, "object_detector") and self.object_detector is not None:
                self.object_detector._close_visualization_windows()
        except Exception:
            pass

        logger.info("FastSAM Service stopped")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop()
