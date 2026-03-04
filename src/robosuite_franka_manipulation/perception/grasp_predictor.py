import numpy as np
import logging
from typing import List,Optional
import torch
from scipy.spatial.transform import Rotation as R
import meshcat
import meshcat.geometry as g
import trimesh.transformations as tra
import trimesh

from robosuite_franka_manipulation import constants

from robosuite_franka_manipulation.constants import get_graspgen_config_path
from robosuite_franka_manipulation.transforms import StampedPose, Transform
from robosuite_franka_manipulation.utils import downsample_point_cloud, transform_point_cloud
from robosuite_franka_manipulation.utils.mesh_utils import load_mesh_and_sample
from robosuite_franka_manipulation.schema import GraspSchema

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal
from grasp_gen.robot import get_gripper_info

# Configure logging
logger = logging.getLogger(__name__)

class GraspPredictor:
    """
    Class to predict grasps for objects using GraspGen.
    """
    def __init__(
        self,
        gripper_type: str,
        model_path: str = "",
        debug: bool = False,
        visualize: bool = False,
        use_dpp: bool = True,
    ) -> None:
        self.debug = debug
        self.visualize = visualize
        self.gripper_type = gripper_type
        self.use_dpp = use_dpp

        # Lazy-created MeshCat visualizer (shared across calls)
        self._meshcat_vis: Optional[meshcat.Visualizer] = None
        
        # Load GraspGen config from GraspGenModels based on gripper type
        if not model_path:
            model_path = str(get_graspgen_config_path(gripper_type))
        cfg = load_grasp_cfg(model_path)
        self.sampler = GraspGenSampler(cfg)

        if self.debug:
            logger.setLevel(logging.DEBUG)

    def get_gripper_collision_mesh(self) -> trimesh.Trimesh:
        """
        Get the collision mesh for the loaded gripper type.
        """
        try:
            if self.gripper_type in ('suction', constants.GRIPPER_TYPE_MAPPING['suction']):
                mesh_path = constants.COBOT_PUMP_MESH_PATH
                if mesh_path.exists():
                    mesh = trimesh.load(str(mesh_path))
                    # Shift origin to TCP
                    # If TCP is at offset relative to base, we translate the mesh by -offset to put TCP at (0,0,0)
                    T = tra.translation_matrix(-constants.SUCTION_TCP_OFFSET)
                    mesh.apply_transform(T)
                    return mesh
                else:
                    logger.warning(f"Custom suction mesh not found at {mesh_path}, falling back to default info.")
                    info = get_gripper_info(self.gripper_type)
                    return info.collision_mesh
            else:
                info = get_gripper_info(self.gripper_type)
                return info.collision_mesh
        except Exception as e:
            logger.error(f"Failed to load gripper collision mesh for {self.gripper_type}: {e}")
            raise

    def predict_grasps(self, 
                       object_pose: StampedPose, 
                       num_grasps: int, 
                       object_pcd: Optional[np.ndarray] = None, 
                       mesh_file: Optional[str] = None,
                       mesh_scale: float = constants.MESH_SCALE) -> List[GraspSchema]:
        """
        Predict grasp poses for an object using either a point cloud or a mesh file.
        
        Args:
            object_pose: Pose of the object in world frame
            num_grasps: Number of grasps to predict
            object_pcd: Point cloud of the object (optional)
            mesh_file: Path to the mesh file (optional)
            mesh_scale: Scale factor for the mesh
            
        Returns:
            List of grasp poses as GraspSchema
        """
        if object_pcd is None and mesh_file is None:
             raise ValueError("Either object_pcd or mesh_file must be provided")

        logger.debug(f"GraspPredictor: Predicting {num_grasps} grasps")
        logger.debug(f"Object pose: pos={object_pose.position}, quat={object_pose.quaternion}")
        
        if self.sampler is None:
            raise RuntimeError("GraspGen sampler not initialized")
            
        # Prepare Point Cloud
        pcd_data = None
        
        if mesh_file:
            logger.debug(f"Loading mesh from {mesh_file}")
            pcd_local = load_mesh_and_sample(mesh_file, num_points=constants.GRASPGEN_POINTCLOUD_SIZE, scale=mesh_scale)
            if pcd_local is None:
                logger.error(f"Failed to load mesh from {mesh_file}")
                return []
            
            # Mesh points are in object frame, but not necessarily centered at (0,0,0)
            # GraspGen expects centered input.
            centroid = pcd_local.mean(axis=0)
            T_subtract_pc_mean = tra.translation_matrix(-centroid)
            pcd_centered = tra.transform_points(pcd_local, T_subtract_pc_mean)
            
            pcd_data = pcd_centered
            
        else:
             # Use provided point cloud (assumed to be in WORLD frame)
             if len(object_pcd) == 0:
                 raise ValueError("Empty point cloud provided")
             
             # Preprocess: World -> Object Local -> Centered/Downsampled
             # self._preprocess_point_cloud does World -> Object Local and downsampling
             # It assumes origin of object frame is the center
             pcd_data = self._preprocess_point_cloud(object_pcd, object_pose)

        # Run Prediction Core
        return self._predict_grasps_core(pcd_data, object_pose, num_grasps)

    def _predict_grasps_core(self, 
                             pcd_processed: np.ndarray, 
                             object_pose: StampedPose, 
                             num_grasps: int) -> List[GraspSchema]:
        """
        Core prediction logic.
        
        Args:
            pcd_processed: Preprocessed point cloud (tensor-ready)
            object_pose: Object world pose
            num_grasps: Number of grasps
            is_centered_input: True if pcd_processed is already centered (mean subtracted)
            original_pcd: Original/Local point cloud for visualization size
        """
        # Convert to torch tensor
        pcd_tensor = torch.from_numpy(pcd_processed).float()
        
        # Move to device
        device = next(self.sampler.model.parameters()).device
        pcd_tensor = pcd_tensor.to(device)
        
        if num_grasps <= constants.MIN_GRASPS: 
            logger.warning(f"Requested number of grasps {num_grasps} is very low, setting to minimum {constants.MIN_GRASPS}.")
            num_grasps = constants.MIN_GRASPS
            
        # Run Inference
        grasps, grasp_conf = GraspGenSampler.run_inference(
            pcd_tensor,
            self.sampler,
            grasp_threshold=constants.GRASPGEN_PREDICTION_THRESHOLD,
            num_grasps=num_grasps*5,
            min_grasps=num_grasps,
            remove_outliers=False,  # We already removed outliers in preprocessing, and GraspGen's outlier removal can be too aggressive
        )
        
        grasp_list: List[GraspSchema] = []
        if len(grasps) > 0:
            grasps_np = grasps.cpu().numpy()
            
            for i in range(len(grasps)):
                grasp_mat = grasps_np[i]
                
                # Extract pos/quat (now in object-local frame)
                pos = grasp_mat[:3, 3]
                rot_mat = grasp_mat[:3, :3]
                quat = R.from_matrix(rot_mat).as_quat()
                
                grasp_local = StampedPose(
                    position=pos,
                    quaternion=quat,
                    frame_id="object_local"
                )
                
                # Transform to world frame
                world_grasp = self._transform_grasp_to_world(grasp_local, object_pose)
                
                grasp_list.append(GraspSchema(
                    pos=world_grasp.position.tolist(),
                    quat=world_grasp.quaternion.tolist(),
                    score=float(grasp_conf[i].cpu().numpy()),
                    id=i,
                    frame_id=constants.WORLD_FRAME,
                    pcd_size=len(pcd_processed),
                    gripper_type=self.gripper_type
                ))

        logger.debug(f"GraspGen generated {len(grasp_list)} grasps from pointcloud of size {len(pcd_processed)}")
        
        if self.use_dpp:
            grasp_list = maximize_entropy_grasps(grasp_list, num_grasps)
        else:
            # Select top-N grasps by score (no diversity maximization)
            grasp_list = sorted(grasp_list, key=lambda g: g.score, reverse=True)[:num_grasps]

        if self.visualize and grasp_list:
             # Visualization logic
            object_to_world = Transform(
                translation=object_pose.position,
                rotation=object_pose.quaternion,
                parent_frame=constants.WORLD_FRAME,
                child_frame="object_local"
            )

            pcd_for_vis = transform_point_cloud(pcd_processed, object_to_world.to_matrix()) 
            self._visualize_pcd(pcd_for_vis)

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
    
    def apply_gripper_offset(self, grasp_pose: StampedPose) -> StampedPose:
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
        
        # Create transform from world to predicted grasp frame
        world_to_predicted = Transform(
            translation=grasp_pose.position,
            rotation=grasp_pose.quaternion,
            parent_frame=grasp_pose.frame_id,
            child_frame="predicted_grasp"
        )
        
        # Create transform from predicted grasp frame to true grasp frame (offset)
        predicted_to_true = Transform(
            translation=offset_pos,
            rotation=offset_quat,
            parent_frame="predicted_grasp",
            child_frame="true_grasp"
        )
        
        # Compose: world -> predicted -> true
        world_to_true = world_to_predicted.compose(predicted_to_true)
        
        return StampedPose(
            position=world_to_true.translation,
            quaternion=world_to_true.rotation,
            frame_id=grasp_pose.frame_id
        )

    def visualize_gripper(self, grasp_pose: StampedPose, name_suffix: str = "") -> None:
        """
        Visualize the gripper collision mesh at the given pose.
        
        Args:
            grasp_pose: Pose to place the gripper mesh
            name_suffix: Unique suffix for the meshcat path
        """
        if not self.visualize:
            return

        if self._meshcat_vis is None:
            self._meshcat_vis = meshcat.Visualizer(zmq_url=str(constants.MESHCAT_ZMQ_URL))
            
        vis = self._meshcat_vis
        # Use a separate path for gripper visualizations
        path = f"perception/predicted_grippers/{self.gripper_type}/{name_suffix}"
        
        try:
            mesh = self.get_gripper_collision_mesh()
            # Convert trimesh to meshcat geometry
            # We can use the utility or convert manually. 
            # Simple manual conversion:
            vertices = mesh.vertices
            faces = mesh.faces
            geom = g.TriangularMeshGeometry(vertices=vertices, faces=faces)
            
            # Simple material (transparent red?)
            material = g.MeshPhongMaterial(color=0xFF0000, opacity=0.5, transparent=True)
            
            vis[path].set_object(geom, material)
            
            # Set transform
            T = Transform(
                translation=grasp_pose.position,
                rotation=grasp_pose.quaternion,
                parent_frame=grasp_pose.frame_id,
                child_frame=f"gripper_{name_suffix}_mesh"
            ).to_matrix()
            vis[path].set_transform(T)
            
        except Exception as e:
            logger.warning(f"Failed to visualize gripper mesh: {e}")

    def clear_visualizations(self) -> None:
        """Clear visualized gripper meshes from MeshCat."""
        if self._meshcat_vis is None:
            return
        
        try:
            self._meshcat_vis["perception/predicted_grippers"].delete()
        except Exception:
            pass

    def _visualize_pcd(self, pcd: np.ndarray) -> None:
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
        
        # Downsample to ~constants.GRASPGEN_POINTCLOUD points (GraspGen standard) using utility function
        pcd_downsampled = downsample_point_cloud(pcd_filtered, target_points=constants.GRASPGEN_POINTCLOUD_SIZE, method='random')
        
        return pcd_downsampled
    
    def _remove_outliers(self, pcd: np.ndarray) -> np.ndarray:
        """Remove outlier points using GraspGen's KNN-based method."""
        if len(pcd) < 10:
            return pcd
        
        pcd_tensor = torch.from_numpy(pcd).float()
        filtered_pcd, _ = point_cloud_outlier_removal(pcd_tensor, threshold=0.1, K=20)
        
        # Ensure we keep at least some points
        if len(filtered_pcd) < 10:
            filtered_pcd, _ = point_cloud_outlier_removal(pcd_tensor, threshold=0.2, K=20)
        
        return filtered_pcd.cpu().numpy()
    

def maximize_entropy_grasps(grasps: List[GraspSchema], num_grasps: int, break_at_neg: bool = False) -> List[GraspSchema]:
    """
    Select a subset of grasps that maximizes entropy (diversity and quality).
    Uses a greedy Determinantal Point Process (DPP) approach.
    
    The entropy of a subset S in a DPP defined by L-ensemble matrix L is log det(L_S).
    We greedily select grasps to maximize this determinant.
    
    Args:
        grasps: List of GraspSchema objects.
        num_grasps: Number of grasps to select.
        
    Returns:
        A list of selected GraspSchema objects.
    """
    if not grasps or num_grasps <= 0:
        return []
    
    if len(grasps) <= num_grasps:
        # Sort by score anyway if we return all of them
        return sorted(grasps, key=lambda g: g.score, reverse=True)

    # 1. Prepare features for each grasp
    # We use position and orientation as features
    num_total = len(grasps)
    positions = np.array([g.pos for g in grasps])
    quaternions = np.array([g.quat for g in grasps])
    # Ensure scores are positive for DPP L-ensemble
    scores = np.array([max(g.score, 1e-4) for g in grasps])
    
    # 2. Compute similarity kernel matrix K
    # K_ij = exp(-0.5 * d(g_i, g_j)^2)
    
    # Pairwise squared Euclidean distance for positions
    # (dist_pos^2)_ij = ||p_i - p_j||^2
    diff_pos = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_pos_sq = np.sum(diff_pos**2, axis=-1)
    
    # Pairwise distance for orientations
    # d_rot = 1 - |<q_i, q_j>|
    # dot_products = |<q_i, q_j>|
    dot_products = np.abs(np.einsum('id,jd->ij', quaternions, quaternions))
    # Clip for numerical stability
    dot_products = np.clip(dot_products, 0.0, 1.0)
    dist_rot = 1.0 - dot_products
    
    # Combined distance squared
    # sigma_p and sigma_r control the diversity scale
    sigma_p = constants.ENTROPY_POSITION_SCALE
    sigma_r = constants.ENTROPY_ORIENTATION_SCALE
    
    # d^2 = (dist_pos / sigma_p)^2 + (dist_rot / sigma_r)^2
    d_sq = dist_pos_sq / (sigma_p**2) + (dist_rot**2) / (sigma_r**2)
    K = np.exp(-0.5 * d_sq)
    
    # 3. Construct L-ensemble matrix L
    # L_ij = score_i * score_j * K_ij
    L = scores[:, np.newaxis] * scores[np.newaxis, :] * K
    
    # 4. Greedy selection to maximize det(L_S)
    selected_indices = []
    remaining_indices = list(range(num_total))
    
    # Start with the grasp with the highest score
    first_idx = int(np.argmax(scores))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively add elements that maximize the conditional log-determinant (Schur complement)
    for it in range(num_grasps - 1):
        best_idx = -1
        
        S = selected_indices
        L_SS = L[np.ix_(S, S)]
        
        try:
            # Add a small epsilon for numerical stability
            L_SS_inv = np.linalg.inv(L_SS + np.eye(len(S)) * 1e-6)
        except np.linalg.LinAlgError:
            L_SS_inv = np.linalg.pinv(L_SS)
            
        # This loop can be vectorized if num_total is very large
        L_ii_diag = np.diag(L)
        # We only need L_iS for i in remaining_indices
        L_rem_S = L[remaining_indices, :][:, S]
        
        # gains = L_ii - L_iS @ L_SS_inv @ L_Si
        # L_iS is (len(rem), len(S))
        # L_SS_inv is (len(S), len(S))
        # Result of L_rem_S @ L_SS_inv @ L_rem_S.T is (len(rem), len(rem))
        # We only want the diagonal
        intermediate = L_rem_S @ L_SS_inv
        gains = L_ii_diag[remaining_indices] - np.einsum('ij,ij->i', intermediate, L_rem_S)
        
        best_idx_in_rem = np.argmax(gains)
        best_idx = remaining_indices[best_idx_in_rem]
        
        if gains[best_idx_in_rem] < 0 and break_at_neg:
            logger.info(f"Breaking early at {it} grasps as more entropy can't be maximized more")
            break
            
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
    return [grasps[i] for i in selected_indices]
