"""
Object Mesh Visualizer for Trajectory Planning.

This module provides utilities to visualize object meshes along trajectories
in MeshCat. Feasibility colours are exposed as class constants, but the
drawing methods use a single caller-supplied colour.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from contextlib import nullcontext

import meshcat
import meshcat.geometry as g

from bimanual_manipulation.transforms import TransformManager, StampedPose
from bimanual_manipulation import constants
from bimanual_manipulation.utils.meshcat_mj_utils import mujoco_geom_to_meshcat
from robosuite.utils import transform_utils as rs_T
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class ObjectMeshVisualizer:
    """Visualizes object meshes along trajectories in MeshCat.

    Features:
    - Renders object meshes at sampled waypoints
    - Colors meshes with a single caller-supplied colour
    - Organizes meshes under ``{base_path}/{object_name}`` in the MeshCat
      hierarchy

    Attributes:
        COLOR_FEASIBLE: Colour constant for feasible waypoints.
        COLOR_INFEASIBLE: Colour constant for waypoints in collision.
        COLOR_IK_FAILED: Colour constant for IK-failed waypoints.
        COLOR_DEFAULT: Default colour constant.
    """
    
    # Color constants (from centralized config)
    COLOR_FEASIBLE = constants.VIS_COLOR_FEASIBLE
    COLOR_INFEASIBLE = constants.VIS_COLOR_INFEASIBLE
    COLOR_IK_FAILED = constants.VIS_COLOR_IK_FAILED
    COLOR_DEFAULT = constants.VIS_COLOR_DEFAULT
    
    def __init__(
        self, 
        vis: meshcat.Visualizer,
        tf_manager: TransformManager,
        base_path: str = "trajectory_configs",
        mesh_sample_ratio: float = 1.0
    ):
        """Initialize the ObjectMeshVisualizer.

        Args:
            vis: MeshCat visualizer instance.
            tf_manager: TransformManager for frame conversions; its ``env``
                and ``env_lock`` are also used.
            base_path: Base path in MeshCat hierarchy. The default
                ``"trajectory_configs"`` is replaced by
                ``f"{constants.VIS_PLANNING_ROOT}/{constants.VIS_ROBOT_CONFIG_PATH}"``.
            mesh_sample_ratio: Fraction of mesh faces to keep, clamped to
                ``[0.01, 1.0]``. Lower values speed up MeshCat rendering
                at the cost of visual quality.
        """
        self.vis = vis
        self.tf_manager = tf_manager
        self.env = tf_manager.env
        self.env_lock = tf_manager.env_lock
        self.mesh_sample_ratio = max(0.01, min(1.0, mesh_sample_ratio))
        
        if base_path == "trajectory_configs":
            self.base_path = f"{constants.VIS_PLANNING_ROOT}/{constants.VIS_ROBOT_CONFIG_PATH}"
        else:
            self.base_path = base_path
            
        self._mesh_cache: Dict[str, g.Geometry] = {}
        
    def _get_meshcat_geometry(self, model, geom_id: int) -> Optional[g.Geometry]:
        """Convert a MuJoCo geom to MeshCat geometry (cached).

        Args:
            model: MuJoCo model.
            geom_id: Geom ID.

        Returns:
            MeshCat geometry, or ``None`` if the geom type is unsupported.
        """
        return mujoco_geom_to_meshcat(model, geom_id, self._mesh_cache, self.mesh_sample_ratio)

    def _get_object_geoms(self, object_name: str) -> List[Tuple[int, np.ndarray]]:
        """Find the geoms attached to an object's body and their body-relative transforms.

        The body is looked up by exact name (*object_name*, then
        ``f"{object_name}_main"``), falling back to the first body whose
        name contains *object_name*. Only geoms whose ``geom_bodyid`` is
        that body are returned (geoms on child bodies are not included).

        Args:
            object_name: Object name.

        Returns:
            List of ``(geom_id, T_rel)`` with ``T_rel`` the ``(4, 4)``
            geom-in-body transform; empty if no body is found.
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
                    
                    quat_xyzw = rs_T.convert_quat(quat, to="xyzw")
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
        """Visualize object at a specific pose.

        Each geom is drawn at ``f"{path}/{geom_name}"``; the pose is first
        converted to the world frame if needed. Does nothing if the object
        has no geoms.

        Args:
            object_name: Object name.
            pose: Object body pose.
            path: Absolute MeshCat path for this visualization.
            color: RGB color as hex integer.
            opacity: Opacity in ``[0, 1]``.
        """
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
        """Visualize object along a trajectory.

        Draws every *step_interval*-th pose plus the last one under
        ``{base_path}/{object_name}/waypoint_XXXX``.

        Args:
            object_name: Object name.
            object_trajectory: List of object poses.
            step_interval: Visualize every Nth pose.
            opacity: Mesh opacity.
            color: RGB color as hex integer (default blue).
        """
        base_traj_path = f"{self.base_path}/{object_name}"
        
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
