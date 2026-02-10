"""
Visualization utilities for planning.
"""
import numpy as np
import logging
import meshcat
import meshcat.geometry as g
from typing import List

from grasp_gen.utils.meshcat_utils import make_frame
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose
from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.planning.utils.trajectory_utils import WaypointFeasibilityResult
from typing import List, Optional

logger = logging.getLogger(__name__)

def visualize_trajectory(
    vis: meshcat.Visualizer, 
    trajectory: List[StampedPose], 
    waypoint_results: Optional[List[WaypointFeasibilityResult]] = None,
    name: str = "trajectory", 
    color: int = 0x0000ff,
    marker_size: float = 0.01
) -> None:
    """Visualize trajectory as a line and markers in Meshcat.
    
    Colors for markers:
    - Green: Feasible (IK success + No collision)
    - Red: Infeasible (In collision)
    - Orange: Unreachable (IK failed)
    - Blue: Default (no feasibility data)
    
    Args:
        vis: Meshcat visualizer instance
        trajectory: List of StampedPose objects
        waypoint_results: Optional list of feasibility results
        name: Name for the visualization group
        color: Default color of the trajectory line
        marker_size: Size of the waypoint spheres
    """
    if not trajectory:
        return

    # Use planning root path for organization
    full_path = f"{constants.VIS_PLANNING_ROOT}/{constants.VIS_TRAJECTORY_PATH}/{name}"
    
    # Delete existing visualization
    vis[full_path].delete()
    
    points = np.array([p.position for p in trajectory]).T
    
    # 1. Draw trajectory line
    vis[f"{full_path}/line"].set_object(g.Line(
        g.PointsGeometry(points),
        g.LineBasicMaterial(linewidth=2, color=color)
    ))
    
    # 2. Draw waypoints and frames
    for i, pose in enumerate(trajectory):
        waypoint_path = f"{full_path}/waypoints/wp_{i}"
        
        # Determine status color
        # Default to blue
        wp_color = 0x0000ff
        
        if waypoint_results and i < len(waypoint_results):
            res = waypoint_results[i]
            if not res.ik_success:
                wp_color = 0xffa500 # Orange (Unreachable)
            elif not res.collision_free:
                wp_color = 0xff0000 # Red (Infeasible/Collision)
            elif res.is_feasible:
                wp_color = 0x00ff00 # Green (Feasible)
        
        # Add sphere marker
        vis[f"{waypoint_path}/marker"].set_object(
            g.Sphere(marker_size),
            g.MeshBasicMaterial(color=wp_color)
        )
        
        # Add coordinate frame
        # Draw frame for every N waypoints or if it's start/end to avoid clutter
        if i % 10 == 0 or i == 0 or i == len(trajectory) - 1:
            frame_name = f"{waypoint_path}/frame"
            
            # Construct transform matrix
            mat = pose.to_matrix()
            
            # Use make_frame from grasp_gen to draw axis
            make_frame(vis, frame_name, h=0.04, radius=0.0015, o=0.8)
            vis[frame_name].set_transform(mat)
        
        # Set transform for the whole waypoint group (marker + potentially frame)
        # Note: we already used mat for frame, but markers also need to be at position
        vis[waypoint_path].set_transform(pose.to_matrix())
