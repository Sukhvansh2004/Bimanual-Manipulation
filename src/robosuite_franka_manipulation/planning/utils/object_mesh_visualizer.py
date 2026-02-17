"""
Object Mesh Visualizer for Trajectory Planning.

This module provides utilities to visualize object meshes along trajectories
in MeshCat with color-coded feasibility (matching robot feasibility).
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union
from contextlib import nullcontext
import mujoco

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.planning.utils.trajectory_utils import WaypointFeasibilityResult

logger = logging.getLogger(__name__)


class ObjectMeshVisualizer:
    """
    Visualizes object meshes along trajectories in MeshCat.
    
    Features:
    - Renders object meshes at each waypoint
    - Colors meshes based on feasibility status of the waypoint
    - Organizes meshes under trajectory ID paths in MeshCat hierarchy
    """
    
    # Color constants (matching RobotMeshVisualizer)
    COLOR_FEASIBLE = 0x00ff00  # Green
    COLOR_INFEASIBLE = 0xff0000  # Red
    COLOR_IK_FAILED = 0xff8800  # Orange
    COLOR_DEFAULT = 0xcccccc    # Grey
    
    def __init__(
        self, 
        vis: meshcat.Visualizer,
        tf_manager: TransformManager,
        base_path: str = "trajectory_configs"
    ):
        """
        Initialize the ObjectMeshVisualizer.
        
        Args:
            vis: MeshCat visualizer instance
            tf_manager: TransformManager for frame conversions
            base_path: Base path in MeshCat hierarchy
        """
        self.vis = vis
        self.tf_manager = tf_manager
        self.env = tf_manager.env
        self.env_lock = tf_manager.env_lock
        
        if base_path == "trajectory_configs":
            self.base_path = f"{constants.VIS_PLANNING_ROOT}/{constants.VIS_ROBOT_CONFIG_PATH}"
        else:
            self.base_path = base_path
            
        self._mesh_cache: Dict[str, g.Geometry] = {}
        
    def _get_meshcat_geometry(self, model, geom_id: int) -> Optional[g.Geometry]:
        """Convert MuJoCo geom to MeshCat geometry. (Adapted from RobotMeshVisualizer)"""
        geom_type = model.geom_type[geom_id]
        cache_key = f"geom_{geom_id}_type_{geom_type}"
        
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]
        
        geometry = None
        
        if geom_type == 7:  # Mesh
            mesh_id = model.geom_dataid[geom_id]
            vert_adr = model.mesh_vertadr[mesh_id]
            vert_num = model.mesh_vertnum[mesh_id]
            face_adr = model.mesh_faceadr[mesh_id]
            face_num = model.mesh_facenum[mesh_id]
            
            vertices = np.array(model.mesh_vert[vert_adr:vert_adr + vert_num])
            faces = np.array(model.mesh_face[face_adr:face_adr + face_num])
            
            geometry = g.TriangularMeshGeometry(vertices, faces)
            
        elif geom_type == 2:  # Sphere
            radius = float(model.geom_size[geom_id][0])
            geometry = g.Sphere(radius)
            
        elif geom_type == 3:  # Capsule
            radius = float(model.geom_size[geom_id][0])
            height = float(model.geom_size[geom_id][1]) * 2
            geometry = g.Cylinder(height, radius)
            
        elif geom_type == 5:  # Cylinder
            radius = float(model.geom_size[geom_id][0])
            height = float(model.geom_size[geom_id][1]) * 2
            geometry = g.Cylinder(height, radius)
            
        elif geom_type == 6:  # Box
            size = np.array(model.geom_size[geom_id]) * 2
            geometry = g.Box(size.tolist())
        
        if geometry is not None:
            self._mesh_cache[cache_key] = geometry
            
        return geometry

    def _get_object_geoms(self, object_name: str) -> List[Tuple[int, np.ndarray]]:
        """
        Find all geoms associated with an object and their relative transforms to the root body.
        
        Returns:
            List of (geom_id, relative_transform)
        """
        ctx = self.env_lock if self.env_lock is not None else nullcontext()
        
        with ctx:
            model = self.env.sim.model
            
            # Find body ID for the object
            # We try standard naming candidates
            body_id = -1
            body_name_candidates = [object_name, f"{object_name}_main"]
            for bn in body_name_candidates:
                try:
                    body_id = model.body_name2id(bn)
                    break
                except Exception:
                    continue
            
            if body_id == -1:
                # Try to find body by partial match if exact fails
                for i in range(model.nbody):
                    name = model.body_id2name(i)
                    if name and object_name in name:
                        body_id = i
                        break
            
            if body_id == -1:
                logger.warning(f"Could not find body for object '{object_name}'")
                return []
            
            # Find all geoms belonging to this body or its children
            # For simplicity, we just look at geoms whose name starts with object_name
            # or belong to the body_id.
            geoms = []
            for geom_id in range(model.ngeom):
                if model.geom_bodyid[geom_id] == body_id:
                    # Get relative transform
                    pos = model.geom_pos[geom_id]
                    quat = model.geom_quat[geom_id] # [w, x, y, z]
                    
                    # Convert to 4x4
                    T_rel = np.eye(4)
                    # MuJoCo uses w-first quat
                    from robosuite.utils import transform_utils as rs_T
                    quat_xyzw = rs_T.convert_quat(quat, to="xyzw")
                    from scipy.spatial.transform import Rotation as R
                    T_rel[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
                    T_rel[:3, 3] = pos
                    
                    geoms.append((geom_id, T_rel))
                    
            return geoms

    def visualize_object_at_pose(
        self,
        object_name: str,
        pose: StampedPose,
        path: str,
        color: int,
        opacity: float = 0.6
    ):
        """Visualize object at a specific pose."""
        geoms = self._get_object_geoms(object_name)
        if not geoms:
            return
            
        # Ensure pose is in world frame
        if pose.frame_id != constants.WORLD_FRAME:
            pose = self.tf_manager.transform_pose(pose, constants.WORLD_FRAME)
            
        T_world = pose.to_matrix()
        
        model = self.env.sim.model
        
        for geom_id, T_rel in geoms:
            geom_name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
            geom_path = f"{path}/{geom_name}"
            
            geometry = self._get_meshcat_geometry(model, geom_id)
            if geometry is None:
                continue
                
            material = g.MeshPhongMaterial(
                color=color,
                opacity=opacity,
                transparent=True
            )
            
            self.vis[geom_path].set_object(geometry, material)
            
            # World transform for geom
            T_geom = T_world @ T_rel
            
            # Handle cylinder/capsule orientation difference (matching RobotMeshVisualizer)
            geom_type = model.geom_type[geom_id]
            if geom_type in [3, 5]:  # Capsule or Cylinder
                R_align = np.array([
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]
                ])
                T_geom = T_geom @ R_align
                
            self.vis[geom_path].set_transform(T_geom)

    def visualize_trajectory(
        self,
        object_name: str,
        object_trajectory: List[StampedPose],
        step_interval: int = 5,
        opacity: float = 0.5,
        color: int = 0x0000ff
    ) -> None:
        """Visualize object along a trajectory."""
        base_traj_path = f"{self.base_path}/object_meshes/{object_name}"
        
        for i, pose in enumerate(object_trajectory):
            if i % step_interval != 0 and i != len(object_trajectory) - 1:
                continue
                
            waypoint_path = f"{base_traj_path}/waypoint_{i:04d}"
            
            self.visualize_object_at_pose(
                object_name=object_name,
                pose=pose,
                path=waypoint_path,
                color=color,
                opacity=opacity
            )
        
        logger.info(f"Visualized object '{object_name}' along trajectory")
