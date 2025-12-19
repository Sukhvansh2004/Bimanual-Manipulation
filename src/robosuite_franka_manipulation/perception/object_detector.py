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
from .grasp_predictor import GraspPredictor

class ObjectDetector:
    """
    Class to detect objects in the scene using FastSAM segmentation.
    Pipeline: FastSAM detection -> Point cloud from depth -> Local grasp prediction
    """
    def __init__(self, env: Any, default_prompt: str = "object", debug: bool = False, visualize: bool = False) -> None:
        self.env = env
        self.default_prompt = default_prompt
        self.debug = debug
        self.visualize = visualize

        # Load FastSAM model from models folder
        package_path = pathlib.Path(__file__).parent.parent.parent.parent
        model_path = package_path / 'models' / 'FastSAM-x.pt'
        self.model = FastSAM(str(model_path))  # Load model

        # Initialize local grasp predictor
        self.grasp_predictor = GraspPredictor(debug=debug, visualize=visualize)

    def get_current_prompt(self) -> str:
        """Get the current text prompt (returns default prompt)."""
        return self.default_prompt

    def detect_objects(self, rgb_image: Optional[np.ndarray] = None,
                      depth_image: Optional[np.ndarray] = None,
                      camera_intrinsics: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Detect objects in the scene using FastSAM on camera images.
        This is called after teleoperating the arm to desired configuration.

        Args:
            rgb_image: RGB image from camera (if None, gets from env)
            depth_image: Depth image from camera (if None, gets from env)
            camera_intrinsics: Camera intrinsics matrix (if None, uses defaults)

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
            if rgb_image is None and 'agentview_image' in obs:
                rgb_image = obs['agentview_image']
            if depth_image is None and 'agentview_depth' in obs:
                depth_image = obs['agentview_depth']

        if rgb_image is None:
            print("No RGB image available")
            return objects

        # Note: depth image may not be available in all environments
        if depth_image is None:
            print("No depth image available - using fallback point cloud generation")

        # Get text prompt
        text_prompt = self.get_current_prompt()

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
            self._visualize_fastsam_results(rgb_image, masks, text_prompt)

        # Process each detected object
        for i, mask in enumerate(masks):
            # Create point cloud from mask and depth
            pcd = self._mask_to_point_cloud(mask, depth_image, rgb_image, camera_intrinsics)

            # Estimate pose from point cloud
            pose = self._estimate_pose_from_point_cloud(pcd)

            # Get grasp predictions for this object
            grasps = self._predict_grasps(pcd, pose)

            objects.append({
                'name': f'object_{i}',
                'pose': pose,
                'pcd': pcd,
                'mask': mask,
                'grasps': grasps,
                'id': i
            })

        return objects

    def _mask_to_point_cloud(self, mask: np.ndarray, depth: Optional[np.ndarray],
                           rgb: Optional[np.ndarray], intrinsics: Optional[np.ndarray]) -> np.ndarray:
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

        # Use camera intrinsics for proper 3D reconstruction
        if intrinsics is None:
            # Default camera intrinsics (example values)
            intrinsics = np.array([
                [615.0, 0, 310.0],
                [0, 615.0, 236.0],
                [0, 0, 1]
            ])

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

    def _visualize_fastsam_results(self, rgb_image: np.ndarray, masks: np.ndarray, prompt: str):
        """Visualize FastSAM detection results."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original image
            ax1.imshow(rgb_image)
            ax1.set_title(f"Original Image\nPrompt: '{prompt}'")
            ax1.axis('off')
            
            # Overlay masks on image
            ax2.imshow(rgb_image)
            ax2.set_title(f"FastSAM Segmentation\nDetected {len(masks)} objects")
            ax2.axis('off')
            
            # Color map for different objects
            colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
            
            for i, mask in enumerate(masks):
                # Create colored overlay
                color = colors[i]
                colored_mask = np.zeros_like(rgb_image, dtype=np.uint8)
                colored_mask[mask > 0] = (color[:3] * 255).astype(np.uint8)
                
                # Blend with original image
                alpha = 0.5
                overlay = cv2.addWeighted(rgb_image, 1-alpha, colored_mask, alpha, 0)
                ax2.imshow(overlay)
                
                # Add label
                # Find mask centroid for label placement
                y_coords, x_coords = np.where(mask > 0)
                if len(y_coords) > 0 and len(x_coords) > 0:
                    centroid_y = int(np.mean(y_coords))
                    centroid_x = int(np.mean(x_coords))
                    ax2.text(centroid_x, centroid_y, f'Obj {i}', 
                            color='white', fontsize=12, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Error in FastSAM visualization: {e}")