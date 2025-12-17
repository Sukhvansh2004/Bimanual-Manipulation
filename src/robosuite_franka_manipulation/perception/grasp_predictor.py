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
        if self.sampler:
            # Use GraspGen to predict grasps
            # Assuming object_pcd is already processed
            grasps = self.sampler.sample_grasps(object_pcd, num_grasps=num_grasps)
            # Transform to world frame
            world_grasps = []
            for grasp in grasps:
                # Apply object pose transformation
                world_grasp = self._transform_grasp_to_world(grasp, object_pose)
                world_grasps.append(world_grasp)
            return world_grasps
        else:
            # Mock grasps
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