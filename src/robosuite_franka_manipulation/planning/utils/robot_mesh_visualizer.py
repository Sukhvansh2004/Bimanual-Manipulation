"""
Robot Mesh Visualizer for Trajectory Planning.

This module provides utilities to visualize robot configurations along trajectories
in MeshCat with color-coded feasibility (green for feasible, red for infeasible).

Note: This module does NOT perform feasibility checks. It expects pre-computed
joint configurations and their feasibility status as input.
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
from robosuite_franka_manipulation.transforms import TransformManager
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_joint_qpos_indices,
    set_qpos,
    get_native_mj_object
)
# Import WaypointFeasibilityResult from trajectory_utils (the source of truth)
from robosuite_franka_manipulation.planning.utils.trajectory_utils import WaypointFeasibilityResult

logger = logging.getLogger(__name__)


class RobotMeshVisualizer:
    """
    Visualizes robot configurations along trajectories in MeshCat.
    
    Features:
    - Renders robot meshes at each waypoint
    - Colors meshes green (feasible) or red (infeasible)
    - Organizes meshes under trajectory ID paths in MeshCat hierarchy
    """
    
    # Color constants
    COLOR_FEASIBLE = 0x00ff00  # Green
    COLOR_INFEASIBLE = 0xff0000  # Red
    COLOR_IK_FAILED = 0xff8800  # Orange (IK failed, different from collision)
    
    def __init__(
        self, 
        vis: meshcat.Visualizer,
        env: DualArmEnvironment,
        env_lock=None,
        tf_manager: Optional[TransformManager] = None,
        base_path: str = "trajectory_configs"
    ):
        """
        Initialize the RobotMeshVisualizer.
        
        Args:
            vis: MeshCat visualizer instance
            env: DualArmEnvironment instance
            env_lock: Optional threading lock for env access
            tf_manager: Optional TransformManager for frame conversions
            base_path: Base path in MeshCat hierarchy for all trajectory visualizations.
                       If None, uses constants.VIS_PLANNING_ROOT/constants.VIS_ROBOT_CONFIG_PATH
        """
        self.vis = vis
        self.env = env
        self.env_lock = env_lock
        self.tf_manager = tf_manager or TransformManager(env, env_lock=env_lock)
        # Use planning root path for organization
        if base_path == "trajectory_configs":
            # Default value - use constants
            self.base_path = f"{constants.VIS_PLANNING_ROOT}/{constants.VIS_ROBOT_CONFIG_PATH}"
        else:
            self.base_path = base_path
        
        # Cache for mesh data to avoid reloading
        self._mesh_cache: Dict[str, g.Geometry] = {}
        self._geom_to_body: Dict[int, int] = {}  # Maps geom_id to body_id
        self._robot_geom_ids: Dict[int, List[int]] = {0: [], 1: []}  # Geom IDs per robot
        
        # Initialize mesh cache
        self._initialize_mesh_cache()
    
    def _initialize_mesh_cache(self):
        """Pre-cache robot mesh geometries from MuJoCo model."""
        ctx = self.env_lock if self.env_lock is not None else nullcontext()
        
        with ctx:
            model = self.env.sim.model
            
            # Build geom->body mapping and identify robot geoms
            for geom_id in range(model.ngeom):
                body_id = model.geom_bodyid[geom_id]
                self._geom_to_body[geom_id] = body_id
                
                geom_name = model.geom_id2name(geom_id) or ""
                
                # Identify robot geoms by name prefix
                if geom_name.startswith("robot0_") or geom_name.startswith("gripper0_"):
                    self._robot_geom_ids[0].append(geom_id)
                elif geom_name.startswith("robot1_") or geom_name.startswith("gripper1_"):
                    self._robot_geom_ids[1].append(geom_id)
                    
            logger.debug(f"Found {len(self._robot_geom_ids[0])} geoms for robot0")
            logger.debug(f"Found {len(self._robot_geom_ids[1])} geoms for robot1")
    
    def _get_meshcat_geometry(self, model, geom_id: int) -> Optional[g.Geometry]:
        """Convert MuJoCo geom to MeshCat geometry."""
        geom_type = model.geom_type[geom_id]
        
        # Unique key for caching
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
    
    def _compute_geom_transforms(
        self, 
        robot_id: int, 
        joint_config: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Compute world transforms for all robot geoms given a joint configuration.
        
        Args:
            robot_id: Robot ID (0 or 1)
            joint_config: Joint angles for the robot arm
            
        Returns:
            Dictionary mapping geom_id to 4x4 transform matrix
        """
        ctx = self.env_lock if self.env_lock is not None else nullcontext()
        
        with ctx:
            model = self.env.sim.model
            data = self.env.sim.data
            
            # Backup state
            qpos_backup = np.array(data.qpos, copy=True)
            qvel_backup = np.array(data.qvel, copy=True)
            
            try:
                # Set joint configuration
                jids = get_arm_joint_ids(model, robot_id)
                qpos_idx = get_joint_qpos_indices(model, jids)
                set_qpos(data, qpos_idx, joint_config)
                
                # Run forward kinematics
                mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
                
                # Extract transforms for robot geoms
                transforms = {}
                for geom_id in self._robot_geom_ids[robot_id]:
                    pos = np.array(data.geom_xpos[geom_id])
                    rot_mat = np.array(data.geom_xmat[geom_id]).reshape(3, 3)
                    
                    T = np.eye(4)
                    T[:3, :3] = rot_mat
                    T[:3, 3] = pos
                    
                    transforms[geom_id] = T
                    
                return transforms
                
            finally:
                # Restore state
                data.qpos[:] = qpos_backup
                data.qvel[:] = qvel_backup
                mujoco.mj_forward(get_native_mj_object(model), get_native_mj_object(data))
    
    def visualize_robot_at_config(
        self,
        robot_id: int,
        joint_config: np.ndarray,
        path: str,
        color: int,
        opacity: float = 0.6
    ):
        """
        Visualize a single robot configuration in MeshCat.
        
        Args:
            robot_id: Robot ID (0 or 1)
            joint_config: Joint angles
            path: MeshCat path for this visualization
            color: RGB color as hex integer
            opacity: Transparency (0-1)
        """
        ctx = self.env_lock if self.env_lock is not None else nullcontext()
        
        with ctx:
            model = self.env.sim.model
            
            # Compute transforms for this configuration
            transforms = self._compute_geom_transforms(robot_id, joint_config)
            
            # Visualize each geom
            for geom_id, T in transforms.items():
                geom_name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
                geom_path = f"{path}/{geom_name}"
                
                # Get geometry
                geometry = self._get_meshcat_geometry(model, geom_id)
                if geometry is None:
                    continue
                
                # Create material with specified color
                material = g.MeshPhongMaterial(
                    color=color,
                    opacity=opacity,
                    transparent=True
                )
                
                # Set object and transform
                self.vis[geom_path].set_object(geometry, material)
                
                # Handle cylinder/capsule orientation difference
                geom_type = model.geom_type[geom_id]
                if geom_type in [3, 5]:  # Capsule or Cylinder
                    R_align = np.array([
                        [1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]
                    ])
                    T = T @ R_align
                
                self.vis[geom_path].set_transform(T)
    
    def visualize_trajectory_configs(
        self,
        trajectory_id: str,
        robot_id: int,
        waypoint_results: List[WaypointFeasibilityResult],
        step_interval: int = 5,
        opacity: float = 0.5
    ) -> None:
        """
        Visualize robot configurations along a trajectory with feasibility coloring.
        
        Args:
            trajectory_id: Unique identifier for this trajectory (used in MeshCat path)
            robot_id: Robot ID (0 or 1)
            waypoint_results: Pre-computed waypoint results with joint configs and feasibility
            step_interval: Visualize every Nth waypoint (default: 5)
            opacity: Mesh opacity
        """
        # Create base path for this trajectory
        traj_path = f"{self.base_path}/{trajectory_id}/robot{robot_id}"
        
        # Visualize selected waypoints
        for result in waypoint_results:
            if result.waypoint_index % step_interval != 0:
                continue
            
            waypoint_path = f"{traj_path}/waypoint_{result.waypoint_index:04d}"
            
            if result.joint_config is None:
                # IK failed - skip visualization
                logger.debug(f"Skipping waypoint {result.waypoint_index} - IK failed")
                continue
            
            # Determine color based on feasibility
            if result.is_feasible:
                color = self.COLOR_FEASIBLE
            elif result.ik_success and not result.collision_free:
                color = self.COLOR_INFEASIBLE
            else:
                color = self.COLOR_IK_FAILED
            
            # Visualize robot at this configuration
            self.visualize_robot_at_config(
                robot_id=robot_id,
                joint_config=result.joint_config,
                path=waypoint_path,
                color=color,
                opacity=opacity
            )
        
        # Log summary
        feasible_count = sum(1 for r in waypoint_results if r.is_feasible)
        ik_failed_count = sum(1 for r in waypoint_results if not r.ik_success)
        collision_count = sum(1 for r in waypoint_results if r.ik_success and not r.collision_free)
        
        logger.info(
            f"Trajectory '{trajectory_id}' robot{robot_id}: "
            f"{feasible_count}/{len(waypoint_results)} feasible, "
            f"{ik_failed_count} IK failures, "
            f"{collision_count} collisions"
        )
    
    def visualize_dual_arm_trajectory(
        self,
        trajectory_id: str,
        waypoint_results_robot0: List[WaypointFeasibilityResult],
        waypoint_results_robot1: List[WaypointFeasibilityResult],
        step_interval: int = 5,
        opacity: float = 0.5
    ) -> None:
        """
        Visualize both robot arms along their trajectories.
        
        Args:
            trajectory_id: Unique identifier for this trajectory
            waypoint_results_robot0: Pre-computed results for robot 0
            waypoint_results_robot1: Pre-computed results for robot 1
            step_interval: Visualize every Nth waypoint
            opacity: Mesh opacity
        """
        self.visualize_trajectory_configs(
            trajectory_id=trajectory_id,
            robot_id=0,
            waypoint_results=waypoint_results_robot0,
            step_interval=step_interval,
            opacity=opacity
        )
        
        self.visualize_trajectory_configs(
            trajectory_id=trajectory_id,
            robot_id=1,
            waypoint_results=waypoint_results_robot1,
            step_interval=step_interval,
            opacity=opacity
        )
    
    def clear_trajectory(self, trajectory_id: str):
        """Clear visualization for a specific trajectory."""
        path = f"{self.base_path}/{trajectory_id}"
        try:
            self.vis[path].delete()
        except Exception as e:
            logger.warning(f"Failed to clear trajectory '{trajectory_id}': {e}")
    
    def clear_all(self):
        """Clear all trajectory visualizations."""
        try:
            self.vis[self.base_path].delete()
        except Exception as e:
            logger.warning(f"Failed to clear all trajectories: {e}")
