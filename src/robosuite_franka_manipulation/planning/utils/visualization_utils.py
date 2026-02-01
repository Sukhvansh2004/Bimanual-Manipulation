"""
Visualization utilities for planning.
"""
import numpy as np
import logging
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from typing import List

from grasp_gen.utils.meshcat_utils import make_frame
import robosuite.utils.transform_utils as T
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose

logger = logging.getLogger(__name__)

def visualize_trajectory(vis: meshcat.Visualizer, trajectory: List[StampedPose], name: str = "trajectory", color: int = 0x00ff00) -> None:
    """Visualize trajectory as a line/points in Meshcat.
    
    Args:
        vis: Meshcat visualizer instance
        trajectory: List of StampedPose objects
        name: Name for the visualization group
        color: Color of the trajectory line
    """
    if not trajectory:
        return

    points = np.array([p.position for p in trajectory]).T
    
    # Draw line
    vis[name].set_object(g.Line(
        g.PointsGeometry(points),
        g.LineBasicMaterial(linewidth=5, color=color)
    ))
    
    # Draw key frames
    for i, pose in enumerate(trajectory):
        if i % 5 == 0: # Draw every 5th frame
            frame_name = f"{name}/frame_{i}"
            
            # Construct transform matrix
            mat = np.eye(4)
            mat[:3, 3] = pose.position
            # Convert quat [x,y,z,w] to matrix
            # Use T.convert_quat to handle xyzw -> wxyz for meshcat
            quat_wxyz = T.convert_quat(pose.quaternion, to="wxyz")
            r = tf.quaternion_matrix(quat_wxyz) 
            mat[:3, :3] = r[:3, :3]
            
            # Use make_frame from grasp_gen to draw axis
            make_frame(vis, frame_name, h=0.05, radius=0.002, o=1.0)
            vis[frame_name].set_transform(mat)
