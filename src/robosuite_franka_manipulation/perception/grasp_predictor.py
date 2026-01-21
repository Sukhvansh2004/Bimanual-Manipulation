import numpy as np
import logging
from typing import List, Dict, Optional
import torch
from scipy.spatial.transform import Rotation as R
import meshcat
import meshcat.geometry as g
import trimesh
import open3d as o3d
import trimesh.transformations as tra

from robosuite_franka_manipulation import constants

from robosuite_franka_manipulation.constants import get_graspgen_config_path
from robosuite_franka_manipulation.transforms import StampedPose, Transform
from robosuite_franka_manipulation.utils import downsample_point_cloud, transform_point_cloud
from robosuite_franka_manipulation.schema import GraspSchema

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

# Configure logging
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
        visualize: bool = False,
    ) -> None:
        self.debug = debug
        self.visualize = visualize
        self.gripper_type = gripper_type

        # Lazy-created MeshCat visualizer (shared across calls)
        self._meshcat_vis: Optional[meshcat.Visualizer] = None
        
        # Load GraspGen config from GraspGenModels based on gripper type
        if not model_path:
            model_path = str(get_graspgen_config_path(gripper_type))
        cfg = load_grasp_cfg(model_path)
        self.sampler = GraspGenSampler(cfg)

        if self.debug:
            logger.setLevel(logging.DEBUG)

    def predict_grasps(self, object_pcd: np.ndarray, object_pose: StampedPose, num_grasps: int = 10) -> List[GraspSchema]:
        """
        Predict grasp poses for an object.

        Args:
            object_pcd: Point cloud of the object
            object_pose: Pose of the object in world frame
            num_grasps: Number of grasps to predict

        Returns:
            List of grasp poses as GraspSchema
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
        
        # Convert to list of GraspSchema
        grasp_list: List[GraspSchema] = []
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
                
                # Apply gripper-specific offset transform
                world_grasp = self._apply_gripper_offset(world_grasp)
                
                grasp_schema = GraspSchema(
                    pos=world_grasp.position.tolist(),
                    quat=world_grasp.quaternion.tolist(),
                    score=float(grasp_conf[i].cpu().numpy()),
                    id=i,
                    frame_id=constants.WORLD_FRAME,
                    pcd_size=len(object_pcd)
                )
                grasp_list.append(grasp_schema)
        
        logger.debug(f"GraspGen generated {len(grasp_list)} grasps")
        if grasp_list:
            logger.debug(f"Best grasp score: {max(g.score for g in grasp_list):.4f}")
        
        if self.visualize and grasp_list:
            # Pass the original object point cloud (ensure it's numpy)
            pcd_for_vis = object_pcd.cpu().numpy() if isinstance(object_pcd, torch.Tensor) else object_pcd
            self._visualize_grasps(pcd_for_vis, grasp_list, object_pose)
        
        return grasp_list

    def predict_grasps_from_mesh(self, mesh_file: str, object_pose: StampedPose, num_grasps: int = 10, mesh_scale: float = 1.0) -> List[GraspSchema]:
        """
        Predict grasp poses for an object using a mesh file.
        
        Args:
            mesh_file: Path to the mesh file (.obj, .stl, etc.)
            object_pose: Pose of the object in world frame
            num_grasps: Number of grasps to predict
            mesh_scale: Scale factor to apply to the mesh
            
        Returns:
            List of grasp poses as GraspSchema
        """
        logger.debug(f"GraspPredictor: Predicting {num_grasps} grasps for mesh: {mesh_file}")
        
        # Load mesh
        try:
            if mesh_file.endswith("ply"):
                pcd_o3d = o3d.io.read_point_cloud(mesh_file)
                pcd_local = np.array(pcd_o3d.points).astype(np.float32)
                # Simple downsampling if needed
                if len(pcd_local) > 2000:
                    indices = np.random.choice(len(pcd_local), 2000, replace=False)
                    pcd_local = pcd_local[indices]
            else:
                mesh = trimesh.load(mesh_file)
                if isinstance(mesh, trimesh.Scene):
                     # Handle scenes by concatenating all geometries
                     # or just taking the first one. For YCB, it's usually a single mesh.
                     if len(mesh.geometry) == 0:
                         raise ValueError("Loaded mesh scene has no geometry")
                     # If it is a scene, we dump it to a single mesh
                     mesh = mesh.dump(concatenate=True)
                
                mesh.apply_scale(mesh_scale)
                # Sample points from surface
                pcd_local, _ = trimesh.sample.sample_surface(mesh, 2048) # Sample 2048 points
                pcd_local = np.array(pcd_local).astype(np.float32)
                
        except Exception as e:
            logger.error(f"Failed to load mesh {mesh_file}: {e}")
            return []

        # Center point cloud (this is what GraspGen expects as input)
        centroid = pcd_local.mean(axis=0)
        T_subtract_pc_mean = tra.translation_matrix(-centroid)
        pcd_centered = tra.transform_points(pcd_local, T_subtract_pc_mean)
        
        # Convert to torch tensor
        pcd_tensor = torch.from_numpy(pcd_centered).float()
        
        # Move to device
        device = next(self.sampler.model.parameters()).device
        pcd_tensor = pcd_tensor.to(device)
        
        # Run Inference
        grasps, grasp_conf = GraspGenSampler.run_inference(
            pcd_tensor,
            self.sampler,
            grasp_threshold=-1.0,
            topk_num_grasps=num_grasps,
            min_grasps=1,
            max_tries=3,
            remove_outliers=False 
        )
        
        grasp_list: List[GraspSchema] = []
        if len(grasps) > 0:
            # Transform grasps back to object-local frame (undo centering)
            # grasp_in_centered -> grasp_in_local -> grasp_in_world
            
            # Inverse of T_subtract_pc_mean is just adding the centroid
            T_add_pc_mean = tra.inverse_matrix(T_subtract_pc_mean)
            
            grasps_np = grasps.cpu().numpy()
            
            for i in range(len(grasps)):
                # 1. Transform from centered frame to object-local frame
                # G_local = T_add_pc_mean @ G_centered
                grasp_centered_mat = grasps_np[i] # 4x4
                grasp_local_mat = T_add_pc_mean @ grasp_centered_mat
                
                # Extract pos/quat from local grasp
                pos_local = grasp_local_mat[:3, 3]
                rot_local = grasp_local_mat[:3, :3]
                quat_local = R.from_matrix(rot_local).as_quat()
                
                grasp_local = StampedPose(
                    position=pos_local,
                    quaternion=quat_local,
                    frame_id="object_local"
                )
                
                # 2. Transform from object-local to world using object_pose
                world_grasp = self._transform_grasp_to_world(grasp_local, object_pose)
                
                # 3. Apply gripper offset
                world_grasp = self._apply_gripper_offset(world_grasp)
                
                grasp_schema = GraspSchema(
                    pos=world_grasp.position.tolist(),
                    quat=world_grasp.quaternion.tolist(),
                    score=float(grasp_conf[i].cpu().numpy()),
                    id=i,
                    frame_id=constants.WORLD_FRAME,
                    pcd_size=len(pcd_local)
                )
                grasp_list.append(grasp_schema)
                
        logger.debug(f"GraspGen generated {len(grasp_list)} grasps from mesh")
        
        if self.visualize and grasp_list:
            # For visualization, we want to show the points in World Frame
            # pcd_local is in object frame.
            # Transform pcd_local to world
            object_to_world = Transform(
                translation=object_pose.position,
                rotation=object_pose.quaternion,
                parent_frame=constants.WORLD_FRAME,
                child_frame="object_local"
            )
            transform_matrix = object_to_world.to_matrix()
            pcd_world = transform_point_cloud(pcd_local, transform_matrix)
            
            self._visualize_grasps(pcd_world, grasp_list, object_pose)

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
            parent_frame=constants.WORLD_FRAME,
            child_frame="object_local"
        )
        
        # Apply transform to get grasp in world frame
        return object_to_world.apply_to_pose(grasp_local)
    
    def _apply_gripper_offset(self, grasp_pose: StampedPose) -> StampedPose:
        """
        Apply gripper-specific offset transform to a grasp pose.
        
        Different grippers require different orientations relative to the predicted grasp.
        For example, finger grippers need a 90 deg Z rotation to align the fingers properly.
        
        Args:
            grasp_pose: Grasp pose in world frame
            
        Returns:
            Grasp pose with gripper offset applied
        """
        # Get offset for current gripper type
        if self.gripper_type not in constants.GRIPPER_GRASP_OFFSETS:
            logger.warning(f"No grasp offset defined for gripper type '{self.gripper_type}', using identity")
            return grasp_pose
        
        offset = constants.GRIPPER_GRASP_OFFSETS[self.gripper_type]
        offset_pos = offset['position']
        offset_quat = offset['quaternion']
        
        # Check if offset is identity (skip computation)
        if np.allclose(offset_quat, [0, 0, 0, 1]) and np.allclose(offset_pos, [0, 0, 0]):
            return grasp_pose
        
        # Apply rotation offset: new_quat = grasp_quat * offset_quat
        # This rotates in the local grasp frame
        grasp_rot = R.from_quat(grasp_pose.quaternion)
        offset_rot = R.from_quat(offset_quat)
        new_rot = grasp_rot * offset_rot
        new_quat = new_rot.as_quat()
        
        # Position offset is already in local grasp frame, add directly
        # (Currently all offsets have zero position, so this is just grasp_pose.position)
        new_pos = grasp_pose.position + offset_pos
        
        return StampedPose(
            position=new_pos,
            quaternion=new_quat,
            frame_id=grasp_pose.frame_id
        )

    def _visualize_grasps(self, pcd: np.ndarray, grasps: List[Dict], object_pose: StampedPose) -> None:
        """
        Visualize the object point cloud and predicted grasps.

        MeshCat-only visualization: publishes into the MeshCat scene tree so it can be
        toggled alongside TF frames.
        """
        if not self.visualize:
            return

        pcd_xyz = np.asarray(pcd)
        if pcd_xyz.ndim != 2 or pcd_xyz.shape[1] < 3:
            logger.warning(
                "Cannot visualize point cloud: expected Nx3+, got shape %s",
                getattr(pcd_xyz, "shape", None),
            )
            return
        pcd_xyz = pcd_xyz[:, :3].astype(np.float32, copy=False)

        # Create or reuse the MeshCat visualizer. Do NOT clear the whole scene;
        # TF streaming and other visuals may be present.
        if self._meshcat_vis is None:
            self._meshcat_vis = meshcat.Visualizer(zmq_url=str(constants.MESHCAT_ZMQ_URL))

        vis = self._meshcat_vis
        root = vis["perception/grasp_predictor"]

        # Clear only our subtree so updates don't accumulate.
        try:
            root.delete()
        except Exception:
            pass

        # Downsample for performance.
        max_points = 8000
        if pcd_xyz.shape[0] > max_points:
            idx = np.random.choice(pcd_xyz.shape[0], size=max_points, replace=False)
            pcd_vis = pcd_xyz[idx]
        else:
            pcd_vis = pcd_xyz

        # MeshCat PointCloud expects 3xN.
        pts = pcd_vis.T
        colors = np.tile(np.array([[0.7], [0.7], [0.7]], dtype=np.float32), (1, pts.shape[1]))
        # g.PointCloud returns a MeshCat Object, so pass it directly.
        root["object_pcd"].set_object(
            g.PointCloud(
                position=pts,
                color=colors,
                size=0.002,
            )
        )

        # NOTE: We intentionally do NOT draw grasp axes / frames here.
        # Grasp frames are already published into the shared TransformManager by
        # GraspService (e.g., grasp_0, grasp_1, ...), and are visualized via the
        # MeshCat TF streamer. Keeping this view pointcloud-only avoids duplicates.

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
            parent_frame=constants.WORLD_FRAME,
            child_frame="object_local"
        ).inverse()  # We want world -> object_local, so invert object -> world
        
        transform_matrix = world_to_object.to_matrix()
        pcd_local = transform_point_cloud(pcd, transform_matrix)
        
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