"""
Robot Mesh Visualizer for Trajectory Planning.

This module provides utilities to visualize robot configurations along trajectories
in MeshCat with color-coded feasibility (green for feasible, red for infeasible).

Note: This module does NOT perform feasibility checks. It expects pre-computed
joint configurations and their feasibility status as input.
"""

import numpy as np
import logging
from typing import List, Dict, Optional
from contextlib import nullcontext
import mujoco

import meshcat
import meshcat.geometry as g

from bimanual_manipulation.transforms import TransformManager
from bimanual_manipulation import constants
from bimanual_manipulation.planning.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_joint_qpos_indices,
    set_qpos,
    get_native_mj_object
)
# Import WaypointFeasibilityResult from trajectory_utils (the source of truth)
from bimanual_manipulation.planning.utils.trajectory_utils import WaypointFeasibilityResult

logger = logging.getLogger(__name__)


class RobotMeshVisualizer:
    """
    Visualizes robot configurations along trajectories in MeshCat.
    
    Features:
    - Renders robot meshes at each waypoint
    - Colors meshes green (feasible) or red (infeasible)
    - Organizes meshes under trajectory ID paths in MeshCat hierarchy
    """
    
    # Color constants (from centralized config)
    COLOR_FEASIBLE = constants.VIS_COLOR_FEASIBLE
    COLOR_INFEASIBLE = constants.VIS_COLOR_INFEASIBLE
    COLOR_IK_FAILED = constants.VIS_COLOR_IK_FAILED
    
    def __init__(
        self, 
        vis: meshcat.Visualizer,
        tf_manager: TransformManager,
        base_path: str = "trajectory_configs",
        mesh_sample_ratio: float = 1.0
    ):
        """
        Initialize the RobotMeshVisualizer.
        
        Args:
            vis: MeshCat visualizer instance
            tf_manager: TransformManager for frame conversions
            base_path: Base path in MeshCat hierarchy for all trajectory visualizations.
                       If None, uses constants.VIS_PLANNING_ROOT/constants.VIS_ROBOT_CONFIG_PATH
            mesh_sample_ratio: Fraction of mesh faces to keep (0-1). Lower values
                               speed up MeshCat rendering at the cost of visual quality.
        """
        self.vis = vis
        self.tf_manager = tf_manager
        self.env = tf_manager.env
        self.env_lock = tf_manager.env_lock
        self.mesh_sample_ratio = max(0.01, min(1.0, mesh_sample_ratio))
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
        from bimanual_manipulation.utils.meshcat_mj_utils import mujoco_geom_to_meshcat
        return mujoco_geom_to_meshcat(model, geom_id, self._mesh_cache, self.mesh_sample_ratio)
    
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
                mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
                
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
                mujoco.mj_fwdPosition(get_native_mj_object(model), get_native_mj_object(data))
    
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
        traj_path = f"{self.base_path}/{robot_id}/{trajectory_id}"
        
        def visualize_waypoint(result: WaypointFeasibilityResult):
            waypoint_path = f"{traj_path}/waypoint_{result.waypoint_index:04d}"
            
            if result.joint_config is None:
                # IK failed - skip visualization
                logger.debug(f"Skipping waypoint {result.waypoint_index} - IK failed")
                return
            
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

        # Visualize selected waypoints
        for result in waypoint_results:
            if result.waypoint_index % step_interval != 0:
                continue
            visualize_waypoint(result)
        
        visualize_waypoint(waypoint_results[-1])
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
