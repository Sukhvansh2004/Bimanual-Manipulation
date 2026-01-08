"""
Transforms Module - Coordinate frame management.

This module provides tools for managing coordinate frames and transforming
poses between different reference frames in the robosuite simulation.

Key Components:
- StampedPose: A pose with an associated frame_id
- Transform: A transformation between two frames
- TransformManager: Central class for managing frame trees and transformations

Usage:
    from robosuite_franka_manipulation.transforms import TransformManager, StampedPose

    # Create manager with robosuite environment
    tf_manager = TransformManager(env)

    # Create a pose in camera frame
    pose = StampedPose(
        position=[0.5, 0.0, 0.3],
        quaternion=[0, 0, 0, 1],
        frame_id="agentview"
    )

    # Transform to world frame
    world_pose = tf_manager.transform_pose(pose, "world")

    # Get current EEF pose
    eef_pose = tf_manager.get_frame_pose("left_eef")
"""

from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose, Transform
from robosuite_franka_manipulation.transforms.transform_manager import TransformManager
from robosuite_franka_manipulation.transforms.frame_utils import (
    detection_to_stamped_pose,
    grasp_to_stamped_pose,
    stamped_pose_to_dict,
    register_detection_frames,
    transform_grasps_to_frame,
    get_eef_pose,
    compute_approach_pose,
    FrameAwareGraspPipeline,
)

from robosuite_franka_manipulation.transforms.meshcat_tf_visualizer import (
    MeshcatTFVisualizerConfig,
    stream_transforms_to_meshcat,
)

__all__ = [
    "StampedPose",
    "Transform",
    "TransformManager",
    "detection_to_stamped_pose",
    "grasp_to_stamped_pose",
    "stamped_pose_to_dict",
    "register_detection_frames",
    "transform_grasps_to_frame",
    "get_eef_pose",
    "compute_approach_pose",
    "FrameAwareGraspPipeline",
    "MeshcatTFVisualizerConfig",
    "stream_transforms_to_meshcat",
]
