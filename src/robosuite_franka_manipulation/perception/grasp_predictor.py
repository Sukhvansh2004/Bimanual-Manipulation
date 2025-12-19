import numpy as np
import sys
import os
from typing import Optional, List, Dict, Any
import pathlib
import traceback
# Add GraspGen to path
package_path = pathlib.Path(__file__).parent.parent.parent.parent  # Up to repo root
GRASPGEN_PATH = package_path / 'deps' / 'GraspGen'
if str(GRASPGEN_PATH) not in sys.path:
    sys.path.append(str(GRASPGEN_PATH))

try:
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.utils.point_cloud_utils import (
        depth_and_segmentation_to_point_clouds, 
        point_cloud_outlier_removal, 
        filter_colliding_grasps
    )
    from grasp_gen.robot import get_gripper_info
except ImportError:
    traceback.print_exc()
    print("GraspGen not available, using mock predictor")
    GraspGenSampler = None

class GraspPredictor:
    """
    Class to predict grasps for objects using GraspGen.
    """
    def __init__(self, model_path: Optional[str] = None) -> None:
        if GraspGenSampler:
            # Load GraspGen config from GraspGenModels
            if model_path is None:
                package_path = pathlib.Path(__file__).parent.parent.parent.parent
                model_path = package_path / 'deps' / 'GraspGenModels' / 'checkpoints' / 'graspgen_franka_panda.yml'
                model_path = str(model_path)
            cfg = load_grasp_cfg(model_path)
            self.sampler = GraspGenSampler(cfg)
        else:
            self.sampler = None

    def predict_grasps(self, object_pcd: np.ndarray, object_pose: Dict, num_grasps: int = 10) -> List[Dict]:
        """
        Predict grasp poses for an object.

        Args:
            object_pcd: Point cloud of the object
            object_pose: Pose of the object in world frame
            num_grasps: Number of grasps to predict

        Returns:
            List of grasp poses (position, quaternion)
        """
        if self.sampler and len(object_pcd) > 0:
            try:
                # Convert to torch tensor if needed
                import torch
                if isinstance(object_pcd, np.ndarray):
                    object_pcd = torch.from_numpy(object_pcd).float()
                
                # Move to same device as model
                device = next(self.sampler.model.parameters()).device
                object_pcd = object_pcd.to(device)
                
                # Use GraspGen to predict grasps
                # The sample method returns (grasps, grasp_conf, None)
                # grasps: (N, 4, 4) transformation matrices
                # grasp_conf: (N,) confidence scores
                grasps, grasp_conf, _ = self.sampler.sample(
                    object_pcd, 
                    threshold=-1.0,  # Use all grasps
                    num_grasps=num_grasps,
                    remove_outliers=False  # Don't remove outliers to avoid empty results
                )
                
                # Convert to list of dicts
                grasp_list = []
                if len(grasps) > 0:
                    for i in range(len(grasps)):
                        # Extract position from translation part of matrix
                        pos = grasps[i][:3, 3].cpu().numpy()
                        
                        # Convert rotation matrix to quaternion
                        rot_matrix = grasps[i][:3, :3].cpu().numpy()
                        quat = self._rotation_matrix_to_quaternion(rot_matrix)
                        
                        grasp = {
                            "pos": pos,
                            "quat": quat,
                            "score": grasp_conf[i].cpu().numpy()
                        }
                        grasp_list.append(grasp)
                
                return grasp_list
            except Exception as e:
                print(f"Error in GraspGen prediction: {e}")
                return self._mock_grasps(object_pose, num_grasps)
        else:
            # Mock grasps if no sampler or empty point cloud
            return self._mock_grasps(object_pose, num_grasps)

    def _transform_grasp_to_world(self, grasp, object_pose):
        """
        Transform grasp from object frame to world frame.
        """
        # Assuming grasp is a pose relative to object
        # Apply transformation
        # For simplicity, return object_pose + small offset
        pos_offset = np.array([0, 0, 0.05])  # Above object
        quat = np.array([0, 0, 0, 1])  # Identity
        return {"pos": object_pose["pos"] + pos_offset, "quat": quat}

    def _rotation_matrix_to_quaternion(self, R):
        """
        Convert rotation matrix to quaternion.
        """
        import scipy.spatial.transform as st
        r = st.Rotation.from_matrix(R)
        return r.as_quat()

    def _mock_grasps(self, object_pose, num_grasps):
        """
        Return mock grasp poses.
        """
        grasps = []
        for i in range(num_grasps):
            angle = i * 2 * np.pi / num_grasps
            pos = object_pose["pos"] + np.array([0.05 * np.cos(angle), 0.05 * np.sin(angle), 0.05])
            quat = np.array([0, 0, 0, 1])  # Identity
            grasps.append({"pos": pos, "quat": quat})
        return grasps