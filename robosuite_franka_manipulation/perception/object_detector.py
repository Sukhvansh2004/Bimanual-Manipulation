import numpy as np
from typing import Any, List, Dict
import sys
import os
import socket

# Add GraspGen to path for FastSAM
GRASPGEN_PATH = '/home/sukhvansh/DualArmFranka/GraspGen'
if GRASPGEN_PATH not in sys.path:
    sys.path.append(GRASPGEN_PATH)

try:
    from ultralytics import FastSAM
except ImportError:
    print("FastSAM not available, using mock detector")
    FastSAM = None

class ObjectDetector:
    """
    Class to detect objects in the scene using FastSAM segmentation.
    """
    def __init__(self, env: Any, prompt_host: str = 'localhost', prompt_port: int = 5557) -> None:
        self.env = env
        self.prompt_host = prompt_host
        self.prompt_port = prompt_port
        if FastSAM:
            self.model = FastSAM('FastSAM-x.pt')  # Load model
        else:
            self.model = None

    def get_current_prompt(self) -> str:
        """Get the current text prompt from the prompt service."""
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((self.prompt_host, self.prompt_port))
            client.send(b"GET")
            prompt = client.recv(1024).decode('utf-8')
            client.close()
            return prompt
        except:
            return "object"  # Default fallback

    def detect_objects(self) -> List[Dict]:
        """
        Detect objects in the scene using FastSAM on camera images.

        Returns:
            List of detected objects with poses, point clouds, and masks
        """
        objects = []
        
        if not self.model:
            # Fallback to sim poses
            return self._detect_from_sim()
        
        # Get camera observations
        obs = self.env._get_observations()
        
        # Assuming camera images are available
        if 'image' in obs:
            rgb_image = obs['image']  # Assuming shape (H, W, 3)
            depth_image = obs.get('depth', None)  # Assuming depth available
            
            # Run FastSAM with text prompt
            text_prompt = self.get_current_prompt()
            results = self.model(rgb_image, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
            
            for result in results:
                masks = result.masks.data.cpu().numpy()
                for i, mask in enumerate(masks):
                    # Create point cloud from mask and depth
                    pcd = self._mask_to_point_cloud(mask, depth_image, rgb_image)
                    
                    # Estimate pose (centroid for now)
                    pose = self._estimate_pose_from_mask(mask, depth_image)
                    
                    objects.append({
                        'name': f'object_{i}',
                        'pose': pose,
                        'pcd': pcd,
                        'mask': mask,
                        'id': i
                    })
        
        return objects

    def _mask_to_point_cloud(self, mask: np.ndarray, depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Convert mask to point cloud using depth."""
        # Simplified point cloud generation
        # In real implementation, use camera intrinsics and proper deprojection
        points = []
        h, w = mask.shape
        for y in range(h):
            for x in range(w):
                if mask[y, x] > 0:
                    z = depth[y, x] if depth is not None else 1.0
                    # Normalize coordinates
                    points.append([x / w, y / h, z])
        return np.array(points)

    def _estimate_pose_from_mask(self, mask: np.ndarray, depth: np.ndarray) -> Dict:
        """Estimate object pose from mask."""
        # Simplified pose estimation
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0:
            return {'pos': np.array([0, 0, 0]), 'quat': np.array([0, 0, 0, 1])}
        
        center_x = np.mean(x_indices) / mask.shape[1]  # Normalized
        center_y = np.mean(y_indices) / mask.shape[0]
        center_z = np.mean(depth[y_indices, x_indices]) if depth is not None else 0.5
        
        return {
            'pos': np.array([center_x, center_y, center_z]),
            'quat': np.array([0, 0, 0, 1])  # Identity quaternion
        }

    def _detect_from_sim(self) -> List[Dict]:
        """Fallback detection using sim object poses."""
        objects = []
        # In robosuite TwoArmHandover, the object is named 'object'
        try:
            object_pos = self.env.sim.data.get_body_xpos('object')
            object_quat = self.env.sim.data.get_body_xquat('object')
            # Mock point cloud
            pcd = self._get_object_pcd()
            objects.append({
                "name": "object",
                "pose": {"pos": object_pos, "quat": object_quat},
                "pcd": pcd
            })
        except:
            print("Object not found in environment")
        return objects

    def _get_object_pcd(self) -> np.ndarray:
        """
        Get point cloud of the object.
        For simplicity, return a mock point cloud.
        """
        # Mock point cloud: a cube
        points = []
        for x in np.linspace(-0.05, 0.05, 10):
            for y in np.linspace(-0.05, 0.05, 10):
                for z in np.linspace(-0.05, 0.05, 10):
                    points.append([x, y, z])
        return np.array(points)