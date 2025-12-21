import numpy as np
import logging
from typing import Optional, List, Dict
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation import constants

logger = logging.getLogger(__name__)
from robosuite_franka_manipulation.constants import get_graspgen_config_path
from robosuite_franka_manipulation.transforms import StampedPose, Transform
from robosuite_franka_manipulation.utils import downsample_point_cloud

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

class GraspPredictor:
    """
    Class to predict grasps for objects using GraspGen.
    """
    def __init__(self, gripper_type: str = None, model_path: Optional[str] = None, debug: bool = False, visualize: bool = False) -> None:
        if gripper_type is None:
            gripper_type = constants.DEFAULT_GRIPPER_TYPE
        self.debug = debug
        self.visualize = visualize
        self.gripper_type = gripper_type
        
        # Load GraspGen config from GraspGenModels based on gripper type
        if model_path is None:
            model_path = str(get_graspgen_config_path(gripper_type))
        cfg = load_grasp_cfg(model_path)
        self.sampler = GraspGenSampler(cfg)

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
        if self.debug:
            logger.debug(f"GraspPredictor: Predicting {num_grasps} grasps for point cloud with {len(object_pcd)} points")
            logger.debug(f"Object pose: pos={object_pose['pos']}, quat={object_pose['quat']}")
        
        # Validate inputs
        if self.sampler is None:
            raise RuntimeError("GraspGen sampler not initialized")
        if len(object_pcd) == 0:
            raise ValueError("Empty point cloud provided")
        
        # Preprocess point cloud to match GraspGen expectations
        # GraspGen expects point clouds in object-local frame, centered at origin
        object_pcd_processed = self._preprocess_point_cloud(object_pcd, object_pose)
        
        # Convert to torch tensor if needed
        if isinstance(object_pcd_processed, np.ndarray):
            object_pcd_processed = torch.from_numpy(object_pcd_processed).float()
        
        # Move to same device as model
        device = next(self.sampler.model.parameters()).device
        object_pcd_processed = object_pcd_processed.to(device)
        
        # Use GraspGen to predict grasps
        # The run_inference method returns (grasps, grasp_conf)
        # grasps: (N, 4, 4) transformation matrices in object-local frame
        # grasp_conf: (N,) confidence scores
        grasps, grasp_conf = GraspGenSampler.run_inference(
            object_pcd_processed,
            self.sampler,
            grasp_threshold=-1.0,  # Use all grasps
            num_grasps=num_grasps,
            topk_num_grasps=num_grasps,  # Return top k grasps
            min_grasps=1,  # Minimum grasps to return
            max_tries=3,  # Maximum inference attempts
            remove_outliers=False  # We already did outlier removal in preprocessing
        )
        
        # Convert to list of dicts
        grasp_list = []
        if len(grasps) > 0:
            # Create object pose as StampedPose for transform
            object_stamped = StampedPose(
                position=object_pose['pos'],
                quaternion=object_pose['quat'],
                frame_id="world"
            )
            
            for i in range(len(grasps)):
                # Extract position from translation part of matrix
                pos = grasps[i][:3, 3].cpu().numpy()
                
                # Convert rotation matrix to quaternion using scipy directly
                rot_matrix = grasps[i][:3, :3].cpu().numpy()
                quat = R.from_matrix(rot_matrix).as_quat()  # Returns (x, y, z, w)
                
                # Create grasp as StampedPose in object-local frame
                grasp_local = StampedPose(
                    position=pos,
                    quaternion=quat,
                    frame_id="object_local"
                )
                
                # Transform to world frame using object pose
                world_grasp = self._transform_grasp_to_world(grasp_local, object_stamped)
                
                grasp_list.append({
                    "pos": world_grasp.position,
                    "quat": world_grasp.quaternion,
                    "score": float(grasp_conf[i].cpu().numpy()),
                    "frame_id": "world"
                })
        
        if self.debug:
            logger.debug(f"GraspGen generated {len(grasp_list)} grasps")
            if grasp_list:
                logger.debug(f"Best grasp score: {max(g['score'] for g in grasp_list):.4f}")
        
        if self.visualize and grasp_list:
            # Pass the original object point cloud (ensure it's numpy)
            pcd_for_vis = object_pcd.cpu().numpy() if isinstance(object_pcd, torch.Tensor) else object_pcd
            self._visualize_grasps(pcd_for_vis, grasp_list, object_pose)
        
        return grasp_list

    def _transform_grasp_to_world(self, grasp_local: StampedPose, object_pose: StampedPose) -> StampedPose:
        """
        Transform grasp from object-local frame to world frame using Transform class.

        Args:
            grasp_local: Grasp pose in object's local frame
            object_pose: Object's pose in world frame

        Returns:
            Grasp pose in world frame
        """
        # Create transform from object-local frame to world frame
        # This transform represents where the object is in the world
        object_to_world = Transform(
            translation=object_pose.position,
            rotation=object_pose.quaternion,
            parent_frame="world",
            child_frame="object_local"
        )
        
        # Apply transform to get grasp in world frame
        return object_to_world.apply_to_pose(grasp_local)

    def _visualize_grasps(self, pcd: np.ndarray, grasps: List[Dict], object_pose: Dict):
        """
        Visualize the object point cloud and predicted grasps using Open3D.
        Each grasp is represented as a coordinate frame.
        """
        if not self.visualize:
            return

        logger.info("Visualizing grasps with Open3D...")

        # Create an Open3D point cloud from the NumPy array
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])  # Ensure only 3D points
        
        # Color the point cloud (e.g., light gray)
        o3d_pcd.paint_uniform_color([0.7, 0.7, 0.7])

        geometries = [o3d_pcd]

        # Add object centroid as a sphere (optional)
        if len(pcd) > 0:
            centroid = np.mean(pcd[:, :3], axis=0)
            centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            centroid_sphere.translate(centroid)
            centroid_sphere.paint_uniform_color([1.0, 0, 0]) # Red
            geometries.append(centroid_sphere)

        # Add coordinate frame for each grasp
        for i, grasp in enumerate(grasps):
            pos = grasp['pos']
            quat = grasp['quat']
            
            # Create a small coordinate frame (axes) for the grasp
            grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            
            # Create transformation matrix from position and quaternion
            rot_matrix = R.from_quat(quat).as_matrix()
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rot_matrix
            transform_matrix[:3, 3] = pos
            
            # Apply transformation to the grasp frame
            grasp_frame.transform(transform_matrix)
            geometries.append(grasp_frame)

            # Optionally add a text label for the grasp (Open3D doesn't have native text rendering)
            # This part is more complex and usually requires custom shaders or external libraries.
            # For a basic RVIZ-like visualization, coordinate frames are usually sufficient.

        # Create a visualizer window
        o3d.visualization.draw_geometries(geometries, window_name="Grasp Visualization",
                                          width=800, height=600,
                                          front=[0.5, -0.8, -0.2],    # Example camera position
                                          lookat=[0.0, 0.0, 0.0],     # Look at origin
                                          up=[0.0, 0.0, 1.0],         # Z-axis up
                                          zoom=0.8)

        logger.info("Open3D grasp visualization closed.")

    def _preprocess_point_cloud(self, pcd: np.ndarray, object_pose: Dict) -> np.ndarray:
        """
        Preprocess point cloud for GraspGen.
        Transform from world frame to object-local frame.
        
        Args:
            pcd: Point cloud in world frame
            object_pose: Object pose in world frame
            
        Returns:
            Point cloud in object-local frame
        """
        # Create transform from world to object-local frame (inverse of object pose)
        world_to_object = Transform(
            translation=object_pose['pos'],
            rotation=object_pose['quat'],
            parent_frame="world",
            child_frame="object_local"
        ).inverse()  # We want world -> object_local, so invert object -> world
        
        # Apply transform to each point
        # For efficiency, do this with matrix operations
        transform_matrix = world_to_object.to_matrix()
        
        # Convert to homogeneous coordinates
        ones = np.ones((len(pcd), 1))
        pcd_homogeneous = np.hstack([pcd, ones])
        
        # Apply transform
        pcd_local = (transform_matrix @ pcd_homogeneous.T).T[:, :3]
        
        # Remove outliers
        pcd_filtered = self._remove_outliers(pcd_local)
        
        # Downsample to ~2048 points (GraspGen standard) using utility function
        pcd_downsampled = downsample_point_cloud(pcd_filtered, target_points=2048, method='random')
        
        return pcd_downsampled
    
    def _remove_outliers(self, pcd: np.ndarray) -> np.ndarray:
        """Remove outlier points using GraspGen's KNN-based method."""
        if len(pcd) < 10:
            return pcd
        
        # Use GraspGen's point cloud outlier removal with adjusted threshold
        # The default threshold of 0.014 might be too aggressive for some point clouds
        pcd_tensor = torch.from_numpy(pcd).float()
        filtered_pcd, _ = point_cloud_outlier_removal(pcd_tensor, threshold=0.05, K=20)
        
        # Ensure we keep at least some points
        if len(filtered_pcd) < 10:
            # If too many points were removed, use a more lenient threshold
            filtered_pcd, _ = point_cloud_outlier_removal(pcd_tensor, threshold=0.1, K=20)
        
        return filtered_pcd.cpu().numpy()