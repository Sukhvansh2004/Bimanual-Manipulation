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

    def get_gripper_collision_mesh(self) -> trimesh.Trimesh:
        """
        Get the collision mesh for the loaded gripper type.
        """
        # The gripper_type passed to init (e.g. 'franka_panda') matches the name expected by get_gripper_info
        try:
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
        
        # Run Inference
        grasps, grasp_conf = GraspGenSampler.run_inference(
            pcd_tensor,
            self.sampler,
            grasp_threshold=constants.GRASPGEN_PREDICTION_THRESHOLD,
            num_grasps=num_grasps
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
        
        # Use GraspGen's point cloud outlier removal with adjusted threshold
        # The default threshold of 0.014 might be too aggressive for some point clouds
        pcd_tensor = torch.from_numpy(pcd).float()
        filtered_pcd, _ = point_cloud_outlier_removal(pcd_tensor, threshold=0.05, K=20)
        
        # Ensure we keep at least some points
        if len(filtered_pcd) < 10:
            # If too many points were removed, use a more lenient threshold
            filtered_pcd, _ = point_cloud_outlier_removal(pcd_tensor, threshold=0.1, K=20)
        
        return filtered_pcd.cpu().numpy()