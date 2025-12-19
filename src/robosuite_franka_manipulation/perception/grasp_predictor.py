import numpy as np
import sys
from typing import Optional, List, Dict
import pathlib
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
        
        if self.visualize:
            self._visualize_point_cloud(object_pcd, object_pose)
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
                
                if self.debug:
                    print(f"GraspGen generated {len(grasp_list)} grasps")
                    if grasp_list:
                        print(f"Best grasp score: {max(g['score'] for g in grasp_list):.4f}")
                
                if self.visualize and grasp_list:
                    self._visualize_grasps(object_pcd, grasp_list, object_pose)
                
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

    def _visualize_point_cloud(self, pcd: np.ndarray, pose: Dict):
        """Visualize the input point cloud."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='blue', alpha=0.6, s=1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            ax.set_title(f'Object Point Cloud\n{len(pcd)} points, Pose: {pose["pos"]}')
            
            # Set equal aspect ratio
            max_range = np.ptp(pcd, axis=0).max() / 2.0
            mid_x = pcd[:, 0].mean()
            mid_y = pcd[:, 1].mean()
            mid_z = pcd[:, 2].mean()
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for point cloud visualization")
        except Exception as e:
            print(f"Error in point cloud visualization: {e}")

    def _visualize_grasps(self, pcd: np.ndarray, grasps: List[Dict], object_pose: Dict):
        """Visualize the generated grasps."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot point cloud
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='blue', alpha=0.3, s=1, label='Object')
            
            # Plot grasps
            colors = plt.cm.viridis(np.linspace(0, 1, len(grasps)))
            for i, grasp in enumerate(grasps):
                pos = grasp['pos']
                score = grasp['score']
                
                # Color by score
                color = colors[i]
                size = 50 + score * 50  # Size based on score
                
                ax.scatter(pos[0], pos[1], pos[2], c=[color], s=size, alpha=0.8, 
                          label=f'Grasp {i}' if i < 5 else "")
                
                # Draw a small arrow or line to show orientation (simplified)
                # This would need proper quaternion to direction vector conversion
                ax.text(pos[0], pos[1], pos[2], '.2f', fontsize=8)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Generated Grasps\n{len(grasps)} grasps for object at {object_pose["pos"]}')
            
            # Set equal aspect ratio
            all_points = np.vstack([pcd, np.array([g['pos'] for g in grasps])])
            max_range = np.ptp(all_points, axis=0).max() / 2.0
            mid_x = all_points[:, 0].mean()
            mid_y = all_points[:, 1].mean()
            mid_z = all_points[:, 2].mean()
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.legend()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for grasp visualization")
        except Exception as e:
            print(f"Error in grasp visualization: {e}")

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