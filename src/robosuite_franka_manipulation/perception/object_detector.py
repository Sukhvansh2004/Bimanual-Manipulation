import numpy as np
from typing import Any, List, Dict, Optional
import sys
import os
import pathlib
import cv2

# Add GraspGen to path for FastSAM
package_path = pathlib.Path(__file__).parent.parent.parent.parent  # Up to repo root
GRASPGEN_PATH = package_path / 'deps' / 'GraspGen'
if str(GRASPGEN_PATH) not in sys.path:
    sys.path.append(str(GRASPGEN_PATH))

from ultralytics import FastSAM
from robosuite_franka_manipulation.perception.grasp_predictor import GraspPredictor
from scipy.spatial.transform import Rotation as R

class ObjectDetector:
    """
    Class to detect objects in the scene using FastSAM segmentation.
    Pipeline: FastSAM detection -> Point cloud from depth -> Local grasp prediction
    """
    def __init__(self, env: Any, default_prompt: str = "object", debug: bool = True, visualize: bool = False) -> None:
        self.env = env
        self.default_prompt = default_prompt
        self.debug = debug
        self.visualize = visualize

        # Get camera info for agentview
        self.camera_name = "agentview"
        if self.camera_name in self.env.camera_names:
            self.camera_id = self.env.camera_names.index(self.camera_name)
            self.camera_intrinsics = self.env.sim.model.cam_intrinsic[self.camera_id]
            self.camera_width = self.env.camera_widths[self.env.camera_names.index(self.camera_name)]
            self.camera_height = self.env.camera_heights[self.env.camera_names.index(self.camera_name)]
        else:
            print(f"Warning: Camera '{self.camera_name}' not found in environment. Perception might fail.")
            self.camera_id = None
            self.camera_intrinsics = None
            self.camera_width = 640  # Default if not found
            self.camera_height = 480 # Default if not found

        # Load FastSAM model from models folder
        package_path = pathlib.Path(__file__).parent.parent.parent.parent
        model_path = package_path / 'models' / 'FastSAM-x.pt'
        self.model = FastSAM(str(model_path))  # Load model

        # Initialize local grasp predictor
        self.grasp_predictor = GraspPredictor(debug=debug, visualize=visualize)



    def detect_objects(self, rgb_image: Optional[np.ndarray] = None,
                      depth_image: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Detect objects in the scene using FastSAM on camera images.
        This is called after teleoperating the arm to desired configuration.

        Args:
            rgb_image: RGB image from camera (if None, gets from env)
            depth_image: Depth image from camera (if None, gets from env)

        Returns:
            List of detected objects with poses, point clouds, and masks
        """
        objects = []

        if not self.model:
            print("FastSAM model not available")
            return objects

        # Get camera observations if not provided
        if rgb_image is None or depth_image is None:
            obs = self.env._get_observations()
            if rgb_image is None and f'{self.camera_name}_image' in obs:
                rgb_image = obs[f'{self.camera_name}_image']
            if depth_image is None and f'{self.camera_name}_depth' in obs:
                depth_image = obs[f'{self.camera_name}_depth']

        if rgb_image is None:
            print("No RGB image available from specified camera")
            return objects

        # Note: depth image may not be available in all environments
        if depth_image is None:
            print("No depth image available - using fallback point cloud generation")
        
        # Check if intrinsics are available
        if self.camera_intrinsics is None:
            print("Camera intrinsics not available - cannot create accurate point cloud")
            # Fallback to default if no intrinsics are found
            intrinsics = np.array([[615.0, 0, 310.0],[0, 615.0, 236.0],[0, 0, 1]])
        else:
            intrinsics = self.camera_intrinsics

        # Get text prompt
        text_prompt = self.default_prompt

        # Run FastSAM with text prompt
        print(f"Running FastSAM with prompt: '{text_prompt}'")
        
        # Convert numpy array to PIL Image for FastSAM
        from PIL import Image
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

        # Get camera pose in world frame
        cam_pos = self.env.sim.data.get_camera_xpos(self.camera_name)
        cam_rot_mat = self.env.sim.data.get_camera_xmat(self.camera_name).reshape(3, 3)
        cam_rot = R.from_matrix(cam_rot_mat)

        # Process each detected object
        for i, mask in enumerate(masks):
            # Create point cloud from mask and depth in camera frame
            pcd_camera_frame = self._mask_to_point_cloud(mask, depth_image, rgb_image, intrinsics)
            
            if pcd_camera_frame.size == 0:
                continue

            # Transform point cloud to world frame
            pcd_world_frame = cam_rot.apply(pcd_camera_frame) + cam_pos
            
            # Estimate pose from point cloud in world frame
            pose = self._estimate_pose_from_point_cloud(pcd_world_frame)

            # Get grasp predictions for this object
            grasps = self._predict_grasps(pcd_world_frame, pose)

            objects.append({
                'name': f'object_{i}',
                'pose': pose,
                'pcd': pcd_world_frame,
                'mask': mask,
                'grasps': grasps,
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
        """Estimate object pose from point cloud (centroid and orientation)."""
        if len(pcd) == 0:
            return {'pos': np.array([0, 0, 0]), 'quat': np.array([0, 0, 0, 1])}

        # Simple centroid
        centroid = np.mean(pcd, axis=0)

        # For orientation, could implement PCA or other methods
        # For now, use identity orientation
        quat = np.array([0, 0, 0, 1])

        return {
            'pos': centroid,
            'quat': quat
        }

    def _predict_grasps(self, pcd: np.ndarray, pose: Dict) -> List[Dict]:
        """Predict grasps using local GraspPredictor."""
        if self.grasp_predictor and self.grasp_predictor.sampler:
            try:
                return self.grasp_predictor.predict_grasps(pcd, pose, num_grasps=10)
            except Exception as e:
                print(f"Error in local grasp prediction: {e}")
                return []
        else:
            print("Grasp predictor not available")
            return []

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