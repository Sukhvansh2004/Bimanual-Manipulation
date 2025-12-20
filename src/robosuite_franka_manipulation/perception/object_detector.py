import numpy as np
from typing import Any, List, Dict, Optional
import sys
import pathlib
import cv2
from PIL import Image

from robosuite_franka_manipulation.constants import (
    FASTSAM_MODEL_PATH, GRASPGEN_DEPS_DIR, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT, DEFAULT_DETECTION_PROMPT,
    get_project_root
)

from ultralytics import FastSAM
from scipy.spatial.transform import Rotation as R

class ObjectDetector:
    """
    Class to detect objects in the scene using FastSAM segmentation.
    Pipeline: FastSAM detection -> Point cloud from depth -> Local grasp prediction
    """
    def __init__(self, camera_intrinsics: np.ndarray, camera_width: int = DEFAULT_CAMERA_WIDTH, camera_height: int = DEFAULT_CAMERA_HEIGHT,
                 prompt: str = DEFAULT_DETECTION_PROMPT, debug: bool = True, visualize: bool = False) -> None:
        self.camera_intrinsics = camera_intrinsics
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.prompt = prompt
        self.debug = debug
        self.visualize = visualize

        # Add GraspGen to path for FastSAM
        GRASPGEN_PATH = get_project_root() / GRASPGEN_DEPS_DIR
        if str(GRASPGEN_PATH) not in sys.path:
            sys.path.append(str(GRASPGEN_PATH))

        # Load FastSAM model from models folder
        model_path = get_project_root() / FASTSAM_MODEL_PATH
        self.model = FastSAM(str(model_path))  # Load model

    def set_prompt(self, prompt: str) -> None:
        """Set the detection prompt."""
        self.prompt = prompt

    def detect_objects(self, rgb_image: np.ndarray, depth_image: Optional[np.ndarray],
                      camera_pos: np.ndarray, camera_rot: R) -> List[Dict]:
        """
        Detect objects in the scene using FastSAM on camera images.

        Args:
            rgb_image: RGB image from camera
            depth_image: Depth image from camera (optional)
            camera_pos: Camera position in world frame
            camera_rot: Camera rotation (scipy Rotation object)

        Returns:
            List of detected objects with poses, point clouds, and masks
        """
        objects = []

        if not self.model:
            print("FastSAM model not available")
            return objects

        if rgb_image is None:
            print("No RGB image available")
            return objects

        # Note: depth image may not be available in all environments
        if depth_image is None:
            print("No depth image available - using fallback point cloud generation")
        
        # Use provided camera intrinsics
        intrinsics = self.camera_intrinsics

        # Get text prompt
        text_prompt = self.prompt

        # Run FastSAM with text prompt
        print(f"Running FastSAM with prompt: '{text_prompt}'")
        
        # Convert numpy array to PIL Image for FastSAM
        pil_image = Image.fromarray(rgb_image)
        results = self.model(pil_image, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        if not results or len(results) == 0:
            print("No FastSAM results")
            return objects

        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            print("No masks detected")
            return objects

        masks = result.masks.data.cpu().numpy()
        
        if self.debug:
            print(f"FastSAM detected {len(masks)} objects with prompt '{text_prompt}'")
        
        if self.visualize:
            self._visualize_fastsam_results(rgb_image, list(masks), text_prompt)

        # Process each detected object
        for i, mask in enumerate(masks):
            # Create point cloud from mask and depth in camera frame
            pcd_camera_frame = self._mask_to_point_cloud(mask, depth_image, rgb_image, intrinsics)
            
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

    def _mask_to_point_cloud(self, mask: np.ndarray, depth: Optional[np.ndarray],
                           rgb: Optional[np.ndarray], intrinsics: np.ndarray) -> np.ndarray:
        """Convert mask to point cloud using depth and camera intrinsics."""
        if depth is None:
            # Fallback to simple normalized coordinates
            points = []
            h, w = mask.shape
            for y in range(h):
                for x in range(w):
                    if mask[y, x] > 0:
                        points.append([x / w, y / h, 0.5])  # Default depth
            return np.array(points)

        fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]

        points = []
        h, w = mask.shape

        for y in range(h):
            for x in range(w):
                if mask[y, x] > 0:
                    z = depth[y, x]
                    if z > 0:  # Valid depth
                        # Deproject to 3D
                        X = (x - cx) * z / fx
                        Y = (y - cy) * z / fy
                        Z = z
                        points.append([X, Y, Z])

        return np.array(points)

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

    def _visualize_fastsam_results(self, rgb_image: np.ndarray, masks: List[np.ndarray], prompt: str) -> Optional[np.ndarray]:
        """
        Visualize FastSAM detection results using OpenCV for a video feed.

        Args:
            rgb_image: Original RGB image.
            masks: Detected segmentation masks.
            prompt: The prompt used for detection.

        Returns:
            The overlaid image (segmented mask on RGB) or None if visualization is disabled.
        """
        if not self.visualize:
            return None

        overlaid_image = rgb_image.copy()
        
        # Color map for different objects
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)] # BGR colors
        
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            
            # Create a colored mask overlay
            mask_rgb = np.zeros_like(overlaid_image)
            mask_rgb[mask > 0] = color
            
            # Blend with original image
            alpha = 0.5
            cv2.addWeighted(mask_rgb, alpha, overlaid_image, 1 - alpha, 0, overlaid_image)
            
            # Add label
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) > 0 and len(x_coords) > 0:
                centroid_y = int(np.mean(y_coords))
                centroid_x = int(np.mean(x_coords))
                
                label = f'Obj {i}'
                cv2.putText(overlaid_image, label, (centroid_x, centroid_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        window_name = f"FastSAM Segmentation: {prompt}"
        cv2.imshow(window_name, overlaid_image)
        cv2.waitKey(1) # Refresh window

        return overlaid_image