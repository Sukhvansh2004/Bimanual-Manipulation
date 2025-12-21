"""
TransformManager - Central class for managing coordinate frame transformations.

Similar to tf2 in ROS, this class maintains a tree of coordinate frames and
provides methods for transforming poses between frames.
"""

import numpy as np
from typing import Dict, Optional, List, Any, Set, Tuple
from scipy.spatial.transform import Rotation as R
import logging
import time
from collections import deque

from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose, Transform

logger = logging.getLogger(__name__)


class TransformManager:
    """
    Central manager for coordinate frame transformations.
    
    Similar to tf2_ros in ROS, this class:
    - Maintains a tree of static transforms
    - Queries dynamic transforms from the simulation
    - Provides transform_pose() to convert between any two connected frames
    - Tracks frame relationships for path finding
    
    Standard frame hierarchy for dual-arm robosuite:
        world (root)
        ├── robot0_base (usually identity to world)
        │   ├── robot0_left_eef
        │   └── robot0_right_eef
        ├── robot1_base (for second robot if present)
        │   ├── robot1_left_eef
        │   └── robot1_right_eef
        ├── agentview (camera)
        ├── frontview (camera)
        └── object_* (detected objects)
    
    Usage:
        tf_manager = TransformManager(env)
        
        # Get camera pose in world frame
        cam_pose = tf_manager.get_frame_pose("agentview")
        
        # Transform pose from camera to world frame
        world_pose = tf_manager.transform_pose(camera_pose, "world")
        
        # Add static transform
        tf_manager.add_static_transform(...)
    """
    
    # Standard frame names
    WORLD_FRAME = "world"
    
    def __init__(self, env: Any = None, auto_discover: bool = True):
        """
        Initialize TransformManager.
        
        Args:
            env: Robosuite environment instance (optional, can be set later)
            auto_discover: Whether to automatically discover frames from environment
        """
        self.env = env
        
        # Static transforms (don't change over time)
        self._static_transforms: Dict[Tuple[str, str], Transform] = {}
        
        # Frame parent mapping for tree traversal
        self._frame_parents: Dict[str, str] = {}
        
        # Known dynamic frame providers (frame_id -> callable that returns Transform)
        self._dynamic_frame_providers: Dict[str, callable] = {}
        
        # Cache for dynamic transforms with timestamp
        self._transform_cache: Dict[str, Tuple[Transform, float]] = {}
        self._cache_duration: float = 0.01  # 10ms cache duration for dynamic transforms
        
        # Discovered frames
        self._known_frames: Set[str] = {self.WORLD_FRAME}
        
        if env is not None and auto_discover:
            self._discover_frames()
    
    def set_environment(self, env: Any, auto_discover: bool = True) -> None:
        """
        Set or update the robosuite environment.
        
        Args:
            env: Robosuite environment instance
            auto_discover: Whether to discover frames from environment
        """
        self.env = env
        if auto_discover:
            self._discover_frames()
    
    def _discover_frames(self) -> None:
        """Discover available frames from the robosuite environment."""
        if self.env is None:
            return
        
        logger.info("Discovering frames from environment...")
        
        # Discover robot frames
        if hasattr(self.env, 'robots'):
            for robot_idx, robot in enumerate(self.env.robots):
                robot_name = getattr(robot, 'name', f'robot{robot_idx}')
                
                # Robot base frame (usually identity to world in robosuite)
                base_frame = f"{robot_name}_base"
                self.add_static_transform(Transform.identity(
                    self.WORLD_FRAME, base_frame, is_static=True
                ))
                
                # Register EEF frames as dynamic
                if hasattr(robot, '_hand_pos'):
                    for arm_id in robot._hand_pos.keys():
                        eef_frame = f"{robot_name}_{arm_id}_eef"
                        self._register_eef_provider(robot_idx, arm_id, eef_frame)
                        self._frame_parents[eef_frame] = base_frame
                        self._known_frames.add(eef_frame)
                        logger.debug(f"Registered EEF frame: {eef_frame}")
        
        # Discover camera frames
        if hasattr(self.env, 'sim'):
            for cam_id in range(self.env.sim.model.ncam):
                cam_name = self.env.sim.model.camera_id2name(cam_id)
                if cam_name:
                    self._register_camera_provider(cam_name)
                    self._frame_parents[cam_name] = self.WORLD_FRAME
                    self._known_frames.add(cam_name)
                    logger.debug(f"Registered camera frame: {cam_name}")
        
        logger.info(f"Discovered frames: {sorted(self._known_frames)}")
    
    def _register_eef_provider(self, robot_idx: int, arm_id: str, frame_id: str) -> None:
        """Register a dynamic transform provider for an end-effector."""
        def get_eef_transform():
            robot = self.env.robots[robot_idx]
            pos = robot._hand_pos[arm_id]
            rot_mat = robot._hand_orn[arm_id]
            quat = R.from_matrix(rot_mat).as_quat()
            
            robot_name = getattr(robot, 'name', f'robot{robot_idx}')
            parent_frame = f"{robot_name}_base"
            
            return Transform(
                translation=pos,
                rotation=quat,
                parent_frame=parent_frame,
                child_frame=frame_id,
                is_static=False
            )
        
        self._dynamic_frame_providers[frame_id] = get_eef_transform
    
    def _register_camera_provider(self, camera_name: str) -> None:
        """Register a dynamic transform provider for a camera."""
        def get_camera_transform():
            pos = self.env.sim.data.get_camera_xpos(camera_name)
            rot_mat = self.env.sim.data.get_camera_xmat(camera_name).reshape(3, 3)
            quat = R.from_matrix(rot_mat).as_quat()
            
            return Transform(
                translation=pos,
                rotation=quat,
                parent_frame=self.WORLD_FRAME,
                child_frame=camera_name,
                is_static=False  # Cameras could potentially move
            )
        
        self._dynamic_frame_providers[camera_name] = get_camera_transform
    
    def add_static_transform(self, transform: Transform) -> None:
        """
        Add a static transform to the tree.
        
        Args:
            transform: The transform to add
        """
        key = (transform.parent_frame, transform.child_frame)
        self._static_transforms[key] = transform
        self._frame_parents[transform.child_frame] = transform.parent_frame
        self._known_frames.add(transform.parent_frame)
        self._known_frames.add(transform.child_frame)
        
        logger.debug(f"Added static transform: {transform.parent_frame} -> {transform.child_frame}")
    
    def add_object_frame(self, object_id: str, pose: StampedPose, 
                        parent_frame: str = None) -> str:
        """
        Add a frame for a detected object.
        
        Args:
            object_id: Unique identifier for the object
            pose: The object's pose
            parent_frame: Parent frame (default: pose's frame_id)
            
        Returns:
            The created frame name
        """
        if parent_frame is None:
            parent_frame = pose.frame_id
        
        frame_name = f"object_{object_id}"
        
        transform = Transform(
            translation=pose.position,
            rotation=pose.quaternion,
            parent_frame=parent_frame,
            child_frame=frame_name,
            is_static=False  # Objects can move
        )
        
        # Store as "semi-static" - updated when explicitly set
        self._static_transforms[(parent_frame, frame_name)] = transform
        self._frame_parents[frame_name] = parent_frame
        self._known_frames.add(frame_name)
        
        return frame_name
    
    def update_object_frame(self, object_id: str, pose: StampedPose) -> None:
        """
        Update an existing object frame.
        
        Args:
            object_id: Object identifier
            pose: New pose for the object
        """
        frame_name = f"object_{object_id}"
        if frame_name not in self._known_frames:
            self.add_object_frame(object_id, pose)
            return
        
        parent_frame = self._frame_parents.get(frame_name, self.WORLD_FRAME)
        
        transform = Transform(
            translation=pose.position,
            rotation=pose.quaternion,
            parent_frame=parent_frame,
            child_frame=frame_name,
            is_static=False
        )
        
        self._static_transforms[(parent_frame, frame_name)] = transform
    
    def get_transform(self, target_frame: str, source_frame: str) -> Transform:
        """
        Get the transform from source_frame to target_frame.
        
        This is the transform that converts poses from source_frame to target_frame:
        pose_in_target = transform * pose_in_source
        
        Args:
            target_frame: The frame to transform TO
            source_frame: The frame to transform FROM
            
        Returns:
            Transform from source_frame to target_frame
            
        Raises:
            ValueError: If no path exists between the frames
        """
        if source_frame == target_frame:
            return Transform.identity(target_frame, source_frame)
        
        # Find path from source to target
        path = self._find_path(source_frame, target_frame)
        
        if path is None:
            raise ValueError(f"No transform path from '{source_frame}' to '{target_frame}'")
        
        # Compose transforms along the path
        result = Transform.identity(path[0], path[0])
        
        for i in range(len(path) - 1):
            from_frame = path[i]
            to_frame = path[i + 1]
            
            # Get transform from from_frame to to_frame
            t = self._get_direct_transform(from_frame, to_frame)
            
            if result.child_frame == result.parent_frame:
                # First real transform
                result = t
            else:
                result = result.compose(t)
        
        return result
    
    def _get_direct_transform(self, from_frame: str, to_frame: str) -> Transform:
        """Get a direct transform between adjacent frames."""
        # Check static transforms
        key = (from_frame, to_frame)
        if key in self._static_transforms:
            return self._static_transforms[key]
        
        # Check inverse static transform
        inv_key = (to_frame, from_frame)
        if inv_key in self._static_transforms:
            return self._static_transforms[inv_key].inverse()
        
        # Check dynamic providers
        if to_frame in self._dynamic_frame_providers:
            # Check cache
            if to_frame in self._transform_cache:
                cached_tf, cached_time = self._transform_cache[to_frame]
                if time.time() - cached_time < self._cache_duration:
                    if cached_tf.parent_frame == from_frame:
                        return cached_tf
                    elif cached_tf.child_frame == from_frame:
                        return cached_tf.inverse()
            
            # Get fresh transform
            tf = self._dynamic_frame_providers[to_frame]()
            self._transform_cache[to_frame] = (tf, time.time())
            
            if tf.parent_frame == from_frame:
                return tf
            elif tf.child_frame == from_frame:
                return tf.inverse()
        
        # Check if from_frame is dynamic
        if from_frame in self._dynamic_frame_providers:
            if from_frame in self._transform_cache:
                cached_tf, cached_time = self._transform_cache[from_frame]
                if time.time() - cached_time < self._cache_duration:
                    if cached_tf.child_frame == to_frame:
                        return cached_tf.inverse()
                    elif cached_tf.parent_frame == to_frame:
                        return cached_tf
            
            tf = self._dynamic_frame_providers[from_frame]()
            self._transform_cache[from_frame] = (tf, time.time())
            
            if tf.child_frame == to_frame:
                return tf.inverse()
            elif tf.parent_frame == to_frame:
                return tf
        
        raise ValueError(f"No direct transform from '{from_frame}' to '{to_frame}'")
    
    def _find_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Find a path between two frames using BFS.
        
        Returns list of frames from source to target, or None if no path exists.
        """
        if source == target:
            return [source]
        
        # Build adjacency list
        adjacency: Dict[str, Set[str]] = {f: set() for f in self._known_frames}
        
        # Add edges from static transforms
        for (parent, child) in self._static_transforms.keys():
            if parent in adjacency:
                adjacency[parent].add(child)
            if child in adjacency:
                adjacency[child].add(parent)
        
        # Add edges from frame parents
        for child, parent in self._frame_parents.items():
            if parent in adjacency and child in adjacency:
                adjacency[parent].add(child)
                adjacency[child].add(parent)
        
        # BFS to find shortest path
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in adjacency.get(current, []):
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def transform_pose(self, pose: StampedPose, target_frame: str) -> StampedPose:
        """
        Transform a pose to a different frame.
        
        Args:
            pose: The pose to transform
            target_frame: The target frame
            
        Returns:
            The pose expressed in the target frame
        """
        if pose.frame_id == target_frame:
            return pose.copy()
        
        # Get transform from pose's frame to target frame
        transform = self.get_transform(target_frame, pose.frame_id)
        
        # Apply transform (note: transform goes from child to parent)
        # We need to construct a transform that goes from pose.frame_id to target_frame
        transform_matrix = self._compute_transform_matrix(pose.frame_id, target_frame)
        
        # Apply to position
        pos_homogeneous = np.append(pose.position, 1.0)
        new_pos = (transform_matrix @ pos_homogeneous)[:3]
        
        # Apply to orientation
        pose_rot = R.from_quat(pose.quaternion)
        transform_rot = R.from_matrix(transform_matrix[:3, :3])
        new_rot = transform_rot * pose_rot
        
        return StampedPose(
            position=new_pos,
            quaternion=new_rot.as_quat(),
            frame_id=target_frame
        )
    
    def _compute_transform_matrix(self, source_frame: str, target_frame: str) -> np.ndarray:
        """Compute the 4x4 transformation matrix from source to target frame."""
        path = self._find_path(source_frame, target_frame)
        
        if path is None:
            raise ValueError(f"No path from '{source_frame}' to '{target_frame}'")
        
        result = np.eye(4)
        
        for i in range(len(path) - 1):
            from_frame = path[i]
            to_frame = path[i + 1]
            
            t = self._get_direct_transform(from_frame, to_frame)
            result = t.to_matrix() @ result
        
        return result
    
    def get_frame_pose(self, frame_id: str, reference_frame: str = None) -> StampedPose:
        """
        Get the pose of a frame relative to another frame.
        
        Args:
            frame_id: The frame to get the pose of
            reference_frame: The reference frame (default: world)
            
        Returns:
            StampedPose of frame_id in reference_frame
        """
        if reference_frame is None:
            reference_frame = self.WORLD_FRAME
        
        transform = self.get_transform(reference_frame, frame_id)
        
        # The transform describes how to go from frame_id to reference_frame
        # The "pose" of frame_id in reference_frame is the origin of frame_id
        # transformed to reference_frame
        
        return StampedPose(
            position=transform.translation,
            quaternion=transform.rotation,
            frame_id=reference_frame
        )
    
    def get_known_frames(self) -> List[str]:
        """Get list of all known frame names."""
        return sorted(list(self._known_frames))
    
    def frame_exists(self, frame_id: str) -> bool:
        """Check if a frame exists."""
        return frame_id in self._known_frames
    
    def clear_cache(self) -> None:
        """Clear the dynamic transform cache."""
        self._transform_cache.clear()
    
    def print_frame_tree(self, root: str = None, indent: int = 0) -> None:
        """Print the frame tree structure."""
        if root is None:
            root = self.WORLD_FRAME
        
        prefix = "  " * indent
        print(f"{prefix}{root}")
        
        # Find children
        children = [f for f, p in self._frame_parents.items() if p == root]
        for child in sorted(children):
            self.print_frame_tree(child, indent + 1)
    
    def __repr__(self) -> str:
        return f"TransformManager(frames={len(self._known_frames)}, env={'set' if self.env else 'None'})"
