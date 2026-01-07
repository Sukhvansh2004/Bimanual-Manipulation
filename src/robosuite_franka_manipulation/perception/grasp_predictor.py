import numpy as np
import logging
from typing import List, Dict
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation import constants

from robosuite_franka_manipulation.constants import get_graspgen_config_path
from robosuite_franka_manipulation.transforms import StampedPose, Transform
from robosuite_franka_manipulation.utils import downsample_point_cloud

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraspPredictor:
    """
    Class to predict grasps for objects using GraspGen.
    """
    def __init__(
        self,
        gripper_type: str = constants.DEFAULT_GRIPPER_TYPE,
        model_path: str = "",
        debug: bool = False,
        visualize: bool = False
    ) -> None:
        self.debug = debug
        self.visualize = visualize
        self.gripper_type = gripper_type
        
        # Load GraspGen config from GraspGenModels based on gripper type
        if not model_path:
            model_path = str(get_graspgen_config_path(gripper_type))
        cfg = load_grasp_cfg(model_path)
        self.sampler = GraspGenSampler(cfg)

        if self.debug:
            logger.setLevel(logging.DEBUG)

    def predict_grasps(self, object_pcd: np.ndarray, object_pose: StampedPose, num_grasps: int = 10) -> List[Dict]:
        """
        Predict grasp poses for an object.

        Args:
            object_pcd: Point cloud of the object
            object_pose: Pose of the object in world frame
            num_grasps: Number of grasps to predict

        Returns:
            List of grasp poses (position, quaternion)
        """
        logger.debug(f"GraspPredictor: Predicting {num_grasps} grasps for point cloud with {len(object_pcd)} points")
        logger.debug(f"Object pose: pos={object_pose.position}, quat={object_pose.quaternion}")
        
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
            topk_num_grasps=num_grasps,  # Return top k grasps
            min_grasps=1,  # Minimum grasps to return
            max_tries=3,  # Maximum inference attempts
            remove_outliers=False  # We already did outlier removal in preprocessing
        )
        
        # Convert to list of dicts
        grasp_list = []
        if len(grasps) > 0:
            # object_pose is already a StampedPose, use it directly
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
                world_grasp = self._transform_grasp_to_world(grasp_local, object_pose)
                
                grasp_list.append({
                    "pos": world_grasp.position,
                    "quat": world_grasp.quaternion,
                    "score": float(grasp_conf[i].cpu().numpy()),
                    "frame_id": "world"
                })
        
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

    def _visualize_grasps(self, pcd: np.ndarray, grasps: List[Dict], object_pose: StampedPose) -> None:
        """
        Visualize the object point cloud and predicted grasps.

        Uses Plotly when available (works well on headless machines) and falls back
        to an Open3D window otherwise.
        """
        if not self.visualize:
            return

        # Prefer Plotly for robust headless visualization. Open3D's Plotly wrapper can
        # render point clouds confusingly (often looking like a sphere). Here we
        # explicitly plot point markers + grasp axes.
        pcd_xyz = np.asarray(pcd)
        if pcd_xyz.ndim != 2 or pcd_xyz.shape[1] < 3:
            logger.warning("Cannot visualize point cloud: expected Nx3+, got shape %s", getattr(pcd_xyz, "shape", None))
            return
        pcd_xyz = pcd_xyz[:, :3]

        try:
            import plotly.graph_objects as go
        except Exception as e:
            logger.warning("Plotly unavailable (%s). Falling back to Open3D window.", e)
            try:
                o3d_pcd = o3d.geometry.PointCloud()
                o3d_pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
                o3d_pcd.paint_uniform_color([0.7, 0.7, 0.7])

                geometries = [o3d_pcd]
                for grasp in grasps:
                    pos = np.asarray(grasp["pos"], dtype=np.float64)
                    quat = np.asarray(grasp["quat"], dtype=np.float64)
                    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                    transform_matrix = np.eye(4)
                    transform_matrix[:3, :3] = R.from_quat(quat).as_matrix()
                    transform_matrix[:3, 3] = pos
                    grasp_frame.transform(transform_matrix)
                    geometries.append(grasp_frame)

                o3d.visualization.draw_geometries(geometries, window_name="Grasp Visualization")
            except Exception as e2:
                logger.warning("Open3D visualization unavailable: %s. Skipping visualization.", e2)
            return

        logger.info("Visualizing grasps with Plotly...")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=pcd_xyz[:, 0],
                y=pcd_xyz[:, 1],
                z=pcd_xyz[:, 2],
                mode="markers",
                name="object_pcd",
                marker=dict(size=2, color="rgba(160,160,160,0.7)"),
            )
        )

        axis_len = 0.05
        for i, grasp in enumerate(grasps):
            pos = np.asarray(grasp["pos"], dtype=np.float64)
            quat = np.asarray(grasp["quat"], dtype=np.float64)
            rot = R.from_quat(quat)

            x_axis = rot.apply([1.0, 0.0, 0.0])
            y_axis = rot.apply([0.0, 1.0, 0.0])
            z_axis = rot.apply([0.0, 0.0, 1.0])

            def _add_axis(vec, color, axis_name: str):
                end = pos + axis_len * np.asarray(vec)
                fig.add_trace(
                    go.Scatter3d(
                        x=[pos[0], end[0]],
                        y=[pos[1], end[1]],
                        z=[pos[2], end[2]],
                        mode="lines",
                        name=f"grasp_{i}_{axis_name}",
                        line=dict(color=color, width=6),
                        showlegend=False,
                    )
                )

            _add_axis(x_axis, "rgb(220,60,60)", "x")
            _add_axis(y_axis, "rgb(60,220,60)", "y")
            _add_axis(z_axis, "rgb(60,60,220)", "z")

            score = grasp.get("score", None)
            label = f"{i}" if score is None else f"{i} ({float(score):.3f})"
            fig.add_trace(
                go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode="markers+text",
                    text=[label],
                    textposition="top center",
                    marker=dict(size=3, color="rgba(40,40,40,0.9)"),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title="Grasp Visualization",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="z",
                aspectmode="data",
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0),
        )

        try:
            fig.show()
            logger.info("Plotly grasp visualization closed.")
        except Exception as e:
            logger.warning("Plotly visualization failed: %s. Skipping visualization.", e)

    def _preprocess_point_cloud(self, pcd: np.ndarray, object_pose: StampedPose) -> np.ndarray:
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
            translation=object_pose.position,
            rotation=object_pose.quaternion,
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