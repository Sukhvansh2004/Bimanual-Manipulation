"""
Frame utilities for integrating TransformManager with detection and grasping.

This module provides helper functions for working with frames in the perception
and manipulation pipeline.
"""

import numpy as np
from typing import Dict, List, Optional, Any

from robosuite_franka_manipulation.transforms import TransformManager, StampedPose, Transform


def detection_to_stamped_pose(detection: Dict, frame_id: str = "world") -> StampedPose:
    """
    Convert a detection dictionary to a StampedPose.
    
    Args:
        detection: Detection dict from ObjectDetector with 'pose' key
        frame_id: Frame ID for the pose (default: world)
        
    Returns:
        StampedPose representing the object's pose
    """
    pose = detection.get('pose', {})
    return StampedPose(
        position=np.array(pose.get('pos', [0, 0, 0])),
        quaternion=np.array(pose.get('quat', [0, 0, 0, 1])),
        frame_id=frame_id
    )


def grasp_to_stamped_pose(grasp: Dict, frame_id: str = "world") -> StampedPose:
    """
    Convert a grasp dictionary to a StampedPose.
    
    Args:
        grasp: Grasp dict from GraspPredictor with 'pos' and 'quat' keys
        frame_id: Frame ID for the pose (default: world)
        
    Returns:
        StampedPose representing the grasp pose
    """
    return StampedPose(
        position=np.array(grasp.get('pos', [0, 0, 0])),
        quaternion=np.array(grasp.get('quat', [0, 0, 0, 1])),
        frame_id=frame_id
    )


def stamped_pose_to_dict(pose: StampedPose) -> Dict:
    """
    Convert a StampedPose to a dictionary format.
    
    Args:
        pose: StampedPose to convert
        
    Returns:
        Dictionary with 'pos', 'quat', and 'frame_id' keys
    """
    return {
        'pos': pose.position,
        'quat': pose.quaternion,
        'frame_id': pose.frame_id
    }


def register_detection_frames(tf_manager: TransformManager, 
                              detections: List[Dict],
                              source_frame: str = "world") -> List[str]:
    """
    Register all detected objects as frames in the transform manager.
    
    Args:
        tf_manager: TransformManager instance
        detections: List of detection dictionaries
        source_frame: Frame the detections are expressed in
        
    Returns:
        List of registered frame names
    """
    frame_names = []
    
    for detection in detections:
        object_id = detection.get('id', detection.get('name', len(frame_names)))
        pose = detection_to_stamped_pose(detection, source_frame)
        
        frame_name = tf_manager.add_object_frame(
            object_id=str(object_id),
            pose=pose,
            parent_frame=source_frame
        )
        frame_names.append(frame_name)
    
    return frame_names


def transform_grasps_to_frame(tf_manager: TransformManager,
                              grasps: List[Dict],
                              source_frame: str,
                              target_frame: str) -> List[Dict]:
    """
    Transform a list of grasps from one frame to another.
    
    Args:
        tf_manager: TransformManager instance
        grasps: List of grasp dictionaries with 'pos' and 'quat'
        source_frame: Current frame of the grasps
        target_frame: Desired frame for the grasps
        
    Returns:
        List of grasp dictionaries in the target frame
    """
    transformed = []
    
    for grasp in grasps:
        pose = grasp_to_stamped_pose(grasp, source_frame)
        transformed_pose = tf_manager.transform_pose(pose, target_frame)
        
        new_grasp = grasp.copy()
        new_grasp['pos'] = transformed_pose.position
        new_grasp['quat'] = transformed_pose.quaternion
        new_grasp['frame_id'] = target_frame
        
        transformed.append(new_grasp)
    
    return transformed


def get_eef_pose(tf_manager: TransformManager, 
                 robot_name: str = "robot0",
                 arm: str = "right",
                 frame: str = "world") -> StampedPose:
    """
    Get the current end-effector pose.
    
    Args:
        tf_manager: TransformManager instance
        robot_name: Name of the robot
        arm: Which arm ("left" or "right")
        frame: Reference frame for the pose
        
    Returns:
        StampedPose of the end-effector
    """
    eef_frame = f"{robot_name}_{arm}_eef"
    return tf_manager.get_frame_pose(eef_frame, frame)


def compute_approach_pose(grasp_pose: StampedPose, 
                         approach_distance: float = 0.1,
                         approach_axis: str = 'z') -> StampedPose:
    """
    Compute an approach pose offset from a grasp pose.
    
    The approach pose is along the negative of the specified axis of the grasp orientation.
    
    Args:
        grasp_pose: The target grasp pose
        approach_distance: Distance to offset along approach axis
        approach_axis: Which axis to approach along ('x', 'y', or 'z')
        
    Returns:
        StampedPose for the approach position
    """
    from scipy.spatial.transform import Rotation as R
    
    rot = R.from_quat(grasp_pose.quaternion)
    
    # Get the approach axis in world frame
    axis_map = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    local_axis = np.array(axis_map.get(approach_axis.lower(), [0, 0, 1]))
    
    # Transform to world frame and negate for approach direction
    world_axis = rot.apply(local_axis)
    
    # Compute approach position (offset along negative axis)
    approach_pos = np.asarray(grasp_pose.position) - approach_distance * world_axis
    
    return StampedPose(
        position=approach_pos,
        quaternion=np.asarray(grasp_pose.quaternion).copy(),
        frame_id=grasp_pose.frame_id
    )


class FrameAwareGraspPipeline:
    """
    A grasp pipeline wrapper that uses TransformManager for all frame operations.
    
    This provides a cleaner interface for grasp planning with automatic frame handling.
    """
    
    def __init__(self, tf_manager: TransformManager, grasp_predictor: Any):
        """
        Initialize the pipeline.
        
        Args:
            tf_manager: TransformManager instance
            grasp_predictor: GraspPredictor instance
        """
        self.tf_manager = tf_manager
        self.grasp_predictor = grasp_predictor
    
    def predict_grasps_for_detection(self, 
                                     detection: Dict,
                                     num_grasps: int = 10,
                                     output_frame: str = "world") -> List[StampedPose]:
        """
        Predict grasps for a detection and return as StampedPoses.
        
        Args:
            detection: Detection dictionary with 'pcd' and 'pose'
            num_grasps: Number of grasps to predict
            output_frame: Frame for output poses
            
        Returns:
            List of grasp poses as StampedPoses
        """
        # Get point cloud and pose
        pcd = detection.get('pcd', np.array([]))
        pose = detection.get('pose', {'pos': [0, 0, 0], 'quat': [0, 0, 0, 1]})
        
        # Predict grasps (returns in world frame)
        grasps = self.grasp_predictor.predict_grasps(
            object_pcd=pcd,
            object_pose=pose,
            num_grasps=num_grasps
        )
        
        # Convert to StampedPoses and transform if needed
        result = []
        for grasp in grasps:
            stamped = grasp_to_stamped_pose(grasp, "world")
            
            if output_frame != "world":
                stamped = self.tf_manager.transform_pose(stamped, output_frame)
            
            result.append(stamped)
        
        return result
    
    def get_best_grasp_for_arm(self,
                               grasps: List[StampedPose],
                               arm: str = "right",
                               robot_name: str = "robot0") -> Optional[StampedPose]:
        """
        Select the best grasp for a specific arm based on reachability.
        
        Args:
            grasps: List of candidate grasp poses
            arm: Which arm to use
            robot_name: Robot name
            
        Returns:
            Best grasp pose or None if no suitable grasp found
        """
        # Get current EEF pose
        eef_pose = get_eef_pose(self.tf_manager, robot_name, arm)
        
        # Score grasps by distance from current EEF position
        scored = []
        for grasp in grasps:
            # Transform to same frame as EEF for comparison
            grasp_in_world = self.tf_manager.transform_pose(grasp, "world")
            distance = np.linalg.norm(np.asarray(grasp_in_world.position) - np.asarray(eef_pose.position))
            scored.append((distance, grasp))
        
        # Sort by distance (closest first)
        scored.sort(key=lambda x: x[0])
        
        if scored:
            return scored[0][1]
        return None
