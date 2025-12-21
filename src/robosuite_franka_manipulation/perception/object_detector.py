import numpy as np
from typing import List, Dict, Optional
import sys
import cv2
import logging
from PIL import Image

from robosuite_franka_manipulation import constants

from ultralytics import FastSAM
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Class to detect objects in the scene using FastSAM segmentation.
    Pipeline: FastSAM detection -> Point cloud from depth -> Local grasp prediction
    """
    def __init__(self, camera_intrinsics: np.ndarray, camera_width: int = None, camera_height: int = None,
                 prompt: str = None, debug: bool = True, visualize: bool = False) -> None:
        # Set defaults if not provided
        if prompt is None:
            prompt = constants.DEFAULT_DETECTION_PROMPT
        self.camera_intrinsics = camera_intrinsics
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.prompt = prompt
        self.debug = debug
        self.visualize = visualize

        # Add GraspGen to path for FastSAM
        GRASPGEN_PATH = constants.get_project_root() / constants.GRASPGEN_DEPS_DIR
        if str(GRASPGEN_PATH) not in sys.path:
            sys.path.append(str(GRASPGEN_PATH))

        # Elevate logger level if debug enabled
        if self.debug:
            logger.setLevel(logging.DEBUG)

        # Load FastSAM model from models folder
        model_path = constants.get_project_root() / constants.FASTSAM_MODEL_PATH
        self.model = FastSAM(str(model_path))  # Load model

    def set_prompt(self, prompt: str) -> None:
        """Set the detection prompt."""
        self.prompt = prompt

    def _get_intrinsic_params(self) -> tuple:
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
        
        if self.debug:
            logger.debug(f"Using intrinsics fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        return fx, fy, cx, cy

    def _ensure_depth_2d(self, depth_image: np.ndarray) -> np.ndarray:
        """Convert depth array to 2D. Raises error if malformed."""
        if depth_image is None:
            raise ValueError("Depth image is None")
        
        depth = depth_image
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        
        if depth.ndim != 2:
            raise ValueError(f"Depth image has invalid shape {depth.shape}, expected 2D")
        
        return depth

    def detect_objects(self, rgb_image: np.ndarray, depth_image: Optional[np.ndarray],
                      camera_pos: np.ndarray, camera_rot: R) -> List[Dict]:
        """
        Detect objects in the scene using FastSAM on camera images.

        Args:
            rgb_image: RGB image from camera
            depth_image: Depth image from camera (required for point clouds)
            camera_pos: Camera position in world frame
            camera_rot: Camera rotation (scipy Rotation object)

        Returns:
            List of detected objects with poses, point clouds, and masks
        """
        if depth_image is None:
            raise ValueError("Depth image is required for object detection and point cloud extraction")
        
        objects = []

        if not self.model:
            logger.error("FastSAM model not available")
            return objects

        if rgb_image is None:
            logger.error("No RGB image available")
            return objects

        # Normalize depth to 2D
        depth_image = self._ensure_depth_2d(depth_image)

        # Run FastSAM with text prompt
        if self.debug:
            logger.debug(f"Running FastSAM with prompt: '{self.prompt}'")
        
        # Convert numpy array to PIL Image for FastSAM
        pil_image = Image.fromarray(rgb_image)
        results = self.model(
            pil_image,
            device=constants.DEVICE,
            retina_masks=True,
            imgsz=1024,
            conf=constants.FASTSAM_CONF_THRESHOLD,
            iou=constants.FASTSAM_IOU_THRESHOLD,
            texts=self.prompt,
            verbose=False,
        )

        if not results or len(results) == 0:
            logger.debug("No FastSAM results")
            return objects

        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            logger.debug("No masks detected")
            return objects

        masks = result.masks.data.cpu().numpy()
        
        if self.debug:
            logger.debug(f"FastSAM detected {len(masks)} objects with prompt '{self.prompt}'")
        
        if self.visualize:
            self._visualize_fastsam_results(result, rgb_image)

        # Process each detected object
        for i, mask in enumerate(masks):
            # Create point cloud from mask and depth in camera frame
            pcd_camera_frame = self._mask_to_point_cloud(mask, depth_image)
            
            if pcd_camera_frame.size == 0:
                continue

            # Transform point cloud to world frame
            pcd_world_frame = camera_rot.apply(pcd_camera_frame) + camera_pos
            
            # Estimate pose from point cloud in world frame
            pose = self._estimate_pose_from_point_cloud(pcd_world_frame)

            objects.append({
                'name': f'object_{i}',
                'pose': pose,
                'pcd': pcd_world_frame,
                'pcd_local': self._transform_to_object_frame(pcd_world_frame, pose),
                'mask': mask,
                'id': i
            })

        return objects

    def _mask_to_point_cloud(self, mask: np.ndarray, depth: Optional[np.ndarray]) -> np.ndarray:
        """Convert mask to point cloud using depth and camera intrinsics. Requires valid depth."""
        if depth is None:
            raise ValueError("Depth image required for point cloud extraction")
        
        # Ensure depth is 2D; reshape flat buffers when dimensions are known
        depth = np.asarray(depth)
        if depth.ndim == 1:
            expected = self.camera_height * self.camera_width
            if depth.size != expected:
                raise ValueError(f"Flat depth buffer size {depth.size} != expected {expected}")
            depth = depth.reshape(self.camera_height, self.camera_width)

        depth = self._ensure_depth_2d(depth)
        
        if self.debug:
            logger.debug(f"mask_to_point_cloud depth shape={depth.shape}, ndim={depth.ndim}")

        fx, fy, cx, cy = self._get_intrinsic_params()

        # Vectorized point cloud extraction (much faster than nested loops)
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) == 0:
            return np.array([])

        y_coords, x_coords = mask_indices
        depth_values = depth[y_coords, x_coords]

        # Filter valid depth values
        valid_mask = (depth_values > 0) & (depth_values < constants.MAX_POINTCLOUD_RANGE)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        depth_values = depth_values[valid_mask]

        if len(depth_values) == 0:
            return np.array([])

        # Deproject to 3D
        z = depth_values
        x = (x_coords - cx) * z / fx
        y = (y_coords - cy) * z / fy

        return np.column_stack([x, y, z]).astype(np.float32)

    def _estimate_pose_from_point_cloud(self, pcd: np.ndarray) -> Dict:
        """Estimate object pose from point cloud using PCA for orientation."""
        if len(pcd) == 0:
            return {'pos': np.array([0, 0, 0]), 'quat': np.array([0, 0, 0, 1])}

        # Centroid
        centroid = np.mean(pcd, axis=0)

        # Center the point cloud
        pcd_centered = pcd - centroid

        # Compute covariance matrix
        cov_matrix = np.cov(pcd_centered.T)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues (largest first)
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_indices]

        # Create rotation matrix from eigenvectors
        # X axis: direction of maximum variance
        # Y axis: direction of second maximum variance  
        # Z axis: up direction (minimum variance, assuming objects rest on surfaces)
        rot_matrix = eigenvectors.copy()
        
        # Ensure right-handed coordinate system
        if np.linalg.det(rot_matrix) < 0:
            rot_matrix[:, 2] = -rot_matrix[:, 2]

        # Convert to quaternion
        quat = R.from_matrix(rot_matrix).as_quat()

        return {
            'pos': centroid,
            'quat': quat
        }

    def _transform_to_object_frame(self, pcd_world: np.ndarray, object_pose: Dict) -> np.ndarray:
        """
        Transform point cloud from world frame to object-local frame.
        
        Args:
            pcd_world: Point cloud in world frame
            object_pose: Object pose in world frame
            
        Returns:
            Point cloud in object-local frame
        """
        object_pos = np.array(object_pose['pos'])
        object_quat = np.array(object_pose['quat'])
        
        # Convert quaternion to rotation matrix
        object_rot = R.from_quat(object_quat).as_matrix()
        
        # Transform: translate then rotate
        pcd_centered = pcd_world - object_pos
        pcd_local = (object_rot.T @ pcd_centered.T).T  # Inverse rotation
        
        return pcd_local

    def _visualize_fastsam_results(self, result, rgb_image: np.ndarray) -> None:
        """
        Visualize FastSAM detection results using Ultralytics built-in plot function.
        Shows both raw camera image and annotated segmentation results.

        Args:
            result: FastSAM result object from model inference
            rgb_image: Original RGB image from camera
        """
        if not self.visualize or result is None:
            return
        
        # Show raw camera image
        cv2.imshow(f"Camera View: {self.prompt}", rgb_image)
        
        # Show annotated segmentation results
        annotated_image = result.plot()
        cv2.imshow(f"FastSAM Segmentation: {self.prompt}", annotated_image)
        
        cv2.waitKey(1)  # Refresh windows
