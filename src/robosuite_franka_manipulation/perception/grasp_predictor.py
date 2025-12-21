import numpy as np
from typing import Optional, List, Dict
import torch
import pathlib
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation import constants

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
            package_path = pathlib.Path(__file__).parent.parent.parent.parent
            config_name = f'graspgen_{gripper_type}.yml'
            model_path = package_path / 'deps' / 'GraspGenModels' / 'checkpoints' / config_name
            model_path = str(model_path)
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
            print(f"GraspPredictor: Predicting {num_grasps} grasps for point cloud with {len(object_pcd)} points")
            print(f"Object pose: pos={object_pose['pos']}, quat={object_pose['quat']}")
        
        # If visualize is true, we will visualize both the point cloud and the grasps in _visualize_grasps
        # So _visualize_point_cloud is no longer needed separately
        if self.sampler and len(object_pcd) > 0:
            try:
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
                    threshold=-1.0,  # Use all grasps
                    num_grasps=num_grasps,
                    topk_num_grasps=num_grasps,  # Return top k grasps
                    min_grasps=1,  # Minimum grasps to return
                    max_tries=3,  # Maximum inference attempts
                    remove_outliers=False  # We already did outlier removal in preprocessing
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
                        
                        # Since GraspGen returns grasps in object-local frame,
                        # we need to transform them to world frame using object_pose
                        grasp_local = {
                            "pos": pos,
                            "quat": quat,
                            "score": grasp_conf[i].cpu().numpy()
                        }
                        world_grasp = self._transform_grasp_to_world(grasp_local, object_pose)
                        grasp_list.append(world_grasp)
                
                if self.debug:
                    print(f"GraspGen generated {len(grasp_list)} grasps")
                    if grasp_list:
                        print(f"Best grasp score: {max(g['score'] for g in grasp_list):.4f}")
                
                if self.visualize and grasp_list:
                    # Pass the original object point cloud (ensure it's numpy)
                    pcd_for_vis = object_pcd.cpu().numpy() if isinstance(object_pcd, torch.Tensor) else object_pcd
                    self._visualize_grasps(pcd_for_vis, grasp_list, object_pose)
                
                return grasp_list
            except Exception as e:
                print(f"Error in GraspGen prediction: {e}")
                return self._mock_grasps(object_pose, num_grasps)
        else:
            # Mock grasps if no sampler or empty point cloud
            return self._mock_grasps(object_pose, num_grasps)

    def _transform_grasp_to_world(self, grasp: Dict, object_pose: Dict) -> Dict:
        """
        Transform grasp from object frame (relative to object_pose) to world frame.

        Args:
            grasp (Dict): Grasp pose in object's local frame with 'pos' and 'quat'.
            object_pose (Dict): Object's pose in world frame with 'pos' and 'quat'.

        Returns:
            Dict: Grasp pose in world frame with 'pos' and 'quat'.
        """
        # Convert object pose to a scipy Rotation object
        object_rot = R.from_quat(object_pose['quat'])
        
        # Convert grasp pose (relative to object) to a scipy Rotation object
        grasp_rot_rel = R.from_quat(grasp['quat'])
        
        # Combine rotations
        world_rot = object_rot * grasp_rot_rel
        world_quat = world_rot.as_quat()
        
        # Combine positions
        # Rotate the relative grasp position by the object's orientation
        rotated_grasp_pos_rel = object_rot.apply(grasp['pos'])
        world_pos = object_pose['pos'] + rotated_grasp_pos_rel
        
        return {"pos": world_pos, "quat": world_quat, "score": grasp.get("score", 0.0)}

    def _rotation_matrix_to_quaternion(self, R_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion.
        
        Note: GraspGen's matrix_to_quaternion returns (w, x, y, z) format,
        but scipy expects (x, y, z, w). We use scipy directly for consistency
        with the rest of the codebase.
        
        Args:
            R_matrix: 3x3 rotation matrix
            
        Returns:
            Quaternion in (x, y, z, w) format (scipy convention)
        """
        # Use scipy for consistent quaternion format throughout the codebase
        return R.from_matrix(R_matrix).as_quat()

    def _visualize_grasps(self, pcd: np.ndarray, grasps: List[Dict], object_pose: Dict):
        """
        Visualize the object point cloud and predicted grasps using Open3D.
        Each grasp is represented as a coordinate frame.
        """
        if not self.visualize:
            return

        print("Visualizing grasps with Open3D...")

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

        print("Open3D grasp visualization closed.")

    def _preprocess_point_cloud(self, pcd: np.ndarray, object_pose: Dict) -> np.ndarray:
        """
        Preprocess point cloud for GraspGen.
        Transform from world frame to object-local frame and center at origin.
        
        Args:
            pcd: Point cloud in world frame
            object_pose: Object pose in world frame
            
        Returns:
            Point cloud in object-local frame, centered at origin
        """
        # Transform point cloud from world to object frame
        object_pos = np.array(object_pose['pos'])
        object_quat = np.array(object_pose['quat'])
        
        # Convert quaternion to rotation matrix
        object_rot = R.from_quat(object_quat).as_matrix()
        
        # Transform points: first translate, then rotate
        pcd_centered = pcd - object_pos
        pcd_local = (object_rot.T @ pcd_centered.T).T  # Inverse rotation
        
        # Remove outliers
        pcd_filtered = self._remove_outliers(pcd_local)
        
        # Downsample to ~2048 points (GraspGen standard)
        pcd_downsampled = self._downsample_point_cloud(pcd_filtered, target_points=2048)
        
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
    
    def _downsample_point_cloud(self, pcd: np.ndarray, target_points: int = 2048) -> np.ndarray:
        """Downsample point cloud to target number of points."""
        if len(pcd) <= target_points:
            return pcd
        
        # Random downsampling
        indices = np.random.choice(len(pcd), target_points, replace=False)
        return pcd[indices]
    
    def _mock_grasps(self, object_pose: Dict, num_grasps: int) -> List[Dict]:
        """
        Generate mock grasps for testing when GraspGen is not available.
        
        Args:
            object_pose: Object pose in world frame
            num_grasps: Number of mock grasps to generate
            
        Returns:
            List of mock grasp dictionaries
        """
        grasps = []
        object_pos = np.array(object_pose['pos'])
        
        for i in range(num_grasps):
            # Generate random position near the object
            offset = np.random.uniform(-0.05, 0.05, 3)
            pos = object_pos + offset
            
            # Generate random orientation
            quat = np.random.uniform(-1, 1, 4)
            quat = quat / np.linalg.norm(quat)  # Normalize
            
            grasps.append({
                "pos": pos,
                "quat": quat,
                "score": 0.5 + np.random.uniform(-0.2, 0.2)  # Random score around 0.5
            })
        
        return grasps