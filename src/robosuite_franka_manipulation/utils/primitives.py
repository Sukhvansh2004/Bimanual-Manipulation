"""
Motion primitives for robotic manipulation.

This module provides utility functions for common manipulation motion patterns
such as approach motions, retreat motions, and linear trajectories.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from typing import Tuple

from robosuite_franka_manipulation.transforms import StampedPose


def compute_approach_pose(target_pose: StampedPose, 
                          approach_distance: float = 0.1,
                          approach_axis: str = 'z') -> StampedPose:
    """
    Compute an approach pose by offsetting along the target pose's local axis.
    
    The approach pose has the same orientation as the target but is offset
    along the specified local axis of the target frame. This is useful for
    approaching a grasp pose from "above" (along the gripper's approach direction).
    
    Args:
        target_pose: The target pose to approach
        approach_distance: Distance to offset along the approach axis (meters).
                          Positive values move in the positive axis direction.
        approach_axis: Which local axis to offset along ('x', 'y', or 'z').
                      For gripper approach, typically 'z' (gripper pointing direction).
    
    Returns:
        StampedPose: The approach pose with same orientation but offset position
    """
    # Get rotation matrix from target quaternion
    rot = R.from_quat(target_pose.quaternion)
    rot_matrix = rot.as_matrix()
    
    # Get the local axis direction in world frame
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(approach_axis.lower(), 2)
    
    # The approach direction is the specified axis of the target frame
    # For a gripper, Z typically points along the approach direction
    approach_direction = rot_matrix[:, axis_idx]
    
    # Compute approach position (offset backwards along the approach axis)
    # Negative because we want to approach FROM a distance
    approach_position = np.array(target_pose.position) - approach_distance * approach_direction
    
    return StampedPose(
        position=approach_position,
        quaternion=target_pose.quaternion,  # Same orientation
        frame_id=target_pose.frame_id
    )


def compute_retreat_pose(current_pose: StampedPose,
                         retreat_distance: float = 0.1,
                         retreat_axis: str = 'z') -> StampedPose:
    """
    Compute a retreat pose by offsetting along the current pose's local axis.
    
    Similar to approach, but typically used after grasping to move away from
    the object/surface.
    
    Args:
        current_pose: The current pose to retreat from
        retreat_distance: Distance to offset along the retreat axis (meters)
        retreat_axis: Which local axis to retreat along ('x', 'y', or 'z')
    
    Returns:
        StampedPose: The retreat pose
    """
    # Get rotation matrix from current quaternion
    rot = R.from_quat(current_pose.quaternion)
    rot_matrix = rot.as_matrix()
    
    # Get the local axis direction in world frame
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(retreat_axis.lower(), 2)
    
    # The retreat direction is the specified axis of the current frame
    retreat_direction = rot_matrix[:, axis_idx]
    
    # Compute retreat position (offset along the axis - opposite of approach)
    retreat_position = np.array(current_pose.position) + retreat_distance * retreat_direction
    
    return StampedPose(
        position=retreat_position,
        quaternion=current_pose.quaternion,  # Same orientation
        frame_id=current_pose.frame_id
    )


def compute_linear_waypoints(start_pose: StampedPose,
                              end_pose: StampedPose,
                              num_waypoints: int = 10) -> list:
    """
    Compute linear interpolation waypoints between two poses.
    
    Interpolates position linearly and orientation using SLERP.
    
    Args:
        start_pose: Starting pose
        end_pose: Ending pose
        num_waypoints: Number of waypoints including start and end
    
    Returns:
        List of StampedPose waypoints
    """
    if num_waypoints < 2:
        return [end_pose]
    
    waypoints = []
    
    start_pos = np.array(start_pose.position)
    end_pos = np.array(end_pose.position)
    
    start_rot = R.from_quat(start_pose.quaternion)
    end_rot = R.from_quat(end_pose.quaternion)
    
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)
        
        # Linear position interpolation
        pos = start_pos + t * (end_pos - start_pos)
        
        # SLERP for orientation
        key_rots = R.from_quat([
            start_rot.as_quat(),
            end_rot.as_quat()
        ])
        rot = Slerp([0, 1], key_rots)([t])[0]
        
        waypoints.append(StampedPose(
            position=pos,
            quaternion=rot.as_quat(),
            frame_id=start_pose.frame_id
        ))
    
    return waypoints
