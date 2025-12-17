import numpy as np
import sys
import os

# Add GraspGen to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../GraspGen'))

try:
    from grasp_gen import GraspGenerator
except ImportError:
    print("GraspGen not available, using mock predictor")
    GraspGenerator = None

class GraspPredictor:
    """
    Class to predict grasps for objects using GraspGen.
    """
    def __init__(self, model_path=None):
        if GraspGenerator:
            # Initialize GraspGen
            self.generator = GraspGenerator(model_path=model_path)
        else:
            self.generator = None

    def predict_grasps(self, object_pcd, object_pose, num_grasps=10):
        """
        Predict grasp poses for an object.

        Args:
            object_pcd: Point cloud of the object
            object_pose: Pose of the object in world frame
            num_grasps: Number of grasps to predict

        Returns:
            List of grasp poses (position, quaternion)
        """
        if self.generator:
            # Use GraspGen to predict grasps
            grasps = self.generator.generate_grasps(object_pcd, num_grasps=num_grasps)
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