import numpy as np
import sys
from typing import Optional, List, Dict
import pathlib
import open3d as o3d
from scipy.spatial.transform import Rotation as R
# Add GraspGen to path
package_path = pathlib.Path(__file__).parent.parent.parent.parent  # Up to repo root
GRASPGEN_PATH = package_path / 'deps' / 'GraspGen'
if str(GRASPGEN_PATH) not in sys.path:
    sys.path.append(str(GRASPGEN_PATH))

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

class GraspPredictor:
    """
    Class to predict grasps for objects using GraspGen.
    """
    def __init__(self, model_path: Optional[str] = None, debug: bool = False, visualize: bool = False) -> None:
        self.debug = debug
        self.visualize = visualize
        # Load GraspGen config from GraspGenModels
        if model_path is None:
            package_path = pathlib.Path(__file__).parent.parent.parent.parent
            model_path = package_path / 'deps' / 'GraspGenModels' / 'checkpoints' / 'graspgen_franka_panda.yml'
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
                        
                        grasp_item = {
                            "pos": pos,
                            "quat": quat,
                            "score": grasp_conf[i].cpu().numpy()
                        }
                        # Transform grasp from object frame to world frame
                        world_grasp = self._transform_grasp_to_world(grasp_item, object_pose)
                        grasp_list.append(world_grasp)
                
                if self.debug:
                    print(f"GraspGen generated {len(grasp_list)} grasps")
                    if grasp_list:
                        print(f"Best grasp score: {max(g['score'] for g in grasp_list):.4f}")
                
                if self.visualize and grasp_list:
                    # Pass the original object point cloud (CPU numpy version)
                    self._visualize_grasps(object_pcd.cpu().numpy(), grasp_list, object_pose)
                
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
        """
        r = R.from_matrix(R_matrix)
        return r.as_quat()

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