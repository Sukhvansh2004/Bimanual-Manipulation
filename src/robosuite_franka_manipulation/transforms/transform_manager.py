"""
TransformManager - Central class for managing coordinate frame transformations.

This class maintains a tree of coordinate frames and
provides methods for transforming poses between frames.
"""

import numpy as np
from typing import Dict, Optional, List, Set, Tuple, Callable
from scipy.spatial.transform import Rotation as R
import logging
import time
from collections import deque
import threading

from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.transforms.stamped_pose import StampedPose, Transform

logger = logging.getLogger(__name__)


class TransformManager:
    """
    Central manager for coordinate frame transformations.
    
    - Maintains a tree of static transforms
    - Queries dynamic transforms from the simulation
    - Provides transform_pose() to convert between any two connected frames
    - Tracks frame relationships for path finding
    
    Standard frame hierarchy for dual-arm robosuite:
        world (root)
        ├── robot0_base (robot's base frame, may have offset from world)
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
    
    # Standard frame names (from constants)
    WORLD_FRAME = constants.WORLD_FRAME
    
    def __init__(self, env: DualArmEnvironment, auto_discover: bool = True, env_lock: Optional[threading.Lock] = None):
        """
        Initialize TransformManager.
        
        Args:
            env: Robosuite environment instance (optional, can be set later)
            auto_discover: Whether to automatically discover frames from environment
            env_lock: Optional threading.Lock used to guard access to MuJoCo sim.data
        """
        self.env: DualArmEnvironment = env
        # Optional lock to guard simulation access when multiple threads use the env
        self._env_lock: Optional[threading.Lock] = env_lock
        
        # Static transforms (don't change over time)
        self._static_transforms: Dict[Tuple[str, str], Transform] = {}
        
        # Frame parent mapping for tree traversal
        self._frame_parents: Dict[str, str] = {}
        
        # Known dynamic frame providers (frame_id -> callable that returns Transform)
        self._dynamic_frame_providers: Dict[str, Callable] = {}
        
        # Cache for dynamic transforms with timestamp
        self._transform_cache: Dict[str, Tuple[Transform, float]] = {}
        self._cache_duration: float = 0.01  # 10ms cache duration for dynamic transforms
        
        # Discovered frames
        self._known_frames: Set[str] = {self.WORLD_FRAME}
        
        if auto_discover:
            self._discover_frames()
    
    def _discover_frames(self) -> None:
        """Discover available frames from the robosuite environment."""
        if self.env is None:
            return
        
        logger.info("Discovering frames from environment...")
        
        # Discover robot frames first, as eye-in-hand cameras are relative to them
        for robot_idx, robot in enumerate(self.env.robots):  # type: ignore[union-attr]
            robot_name = f"robot{robot_idx}"

            # Robot base frame
            base_frame = f"{robot_name}_base"
            self._register_robot_base_provider(robot_idx, base_frame)
            
            # EEF frames
            try:
                arm_ids = list(robot.robot_model.eef_name.keys())
            except Exception:
                arm_ids = list(getattr(robot, "arms", []) or [])

            for arm_id in arm_ids:
                eef_frame = f"{robot_name}_{arm_id}_eef"
                self._register_eef_provider(robot_idx, arm_id, eef_frame)
                self._frame_parents[eef_frame] = base_frame
                self._known_frames.add(eef_frame)
                logger.debug(f"Registered EEF frame: {eef_frame}")
        
        # Discover camera frames
        for cam_id in range(self.env.sim.model.ncam):
            cam_name = self.env.sim.model.camera_id2name(cam_id)
            if not cam_name:
                continue

            parent_frame = self.WORLD_FRAME
            if "eye_in_hand" in cam_name:
                try:
                    parts = cam_name.split('_')
                    robot_name_part = parts[0]
                    robot_idx = int(robot_name_part.replace('robot', ''))
                    robot = self.env.robots[robot_idx]
                    arm_id = list(robot.robot_model.eef_name.keys())[0]  # Assume first/only arm
                    parent_frame = f"{robot_name_part}_{arm_id}_eef"
                except (IndexError, ValueError) as e:
                    logger.warning(f"Could not determine parent for eye_in_hand camera '{cam_name}': {e}. Defaulting to world.")
                    parent_frame = self.WORLD_FRAME

            # This will handle both static (eye-in-hand) and dynamic (fixed) cameras
            self._register_camera_provider(cam_name, parent_frame=parent_frame)
            
            cam_mj = f"{cam_name}_mj"
            self._frame_parents[cam_mj] = parent_frame
            self._frame_parents[cam_name] = cam_mj
            self._known_frames.add(cam_mj)
            self._known_frames.add(cam_name)

            cam_opt = f"{cam_name}_optical"
            self._frame_parents[cam_opt] = cam_mj
            self._known_frames.add(cam_opt)
            logger.debug(f"Registered camera frame: {cam_name}")
        
        logger.info(f"Discovered frames: {sorted(self._known_frames)}")
    
    def _register_robot_base_provider(self, robot_idx: int, frame_id: str) -> None:
        """Register a dynamic transform provider for a robot's base frame."""
        def get_base_transform() -> Transform:
            robot = self.env.robots[robot_idx]  # type: ignore[union-attr, index]
            
            # Get robot base position and orientation from simulation
            # The robot's base body name varies by robosuite model; try multiple fallbacks.
            robot_name = getattr(robot, 'name', f'robot{robot_idx}')
            
            # Try to get base pose from the robot's root body in MuJoCo
            try:
                # Get the robot's base body position and orientation
                base_body_name = robot.robot_model.root_body
                if self._env_lock:
                    with self._env_lock:
                        pos = self.env.sim.data.get_body_xpos(base_body_name)
                        rot_mat = self.env.sim.data.get_body_xmat(base_body_name).reshape(3, 3)
                else:
                    pos = self.env.sim.data.get_body_xpos(base_body_name)
                    rot_mat = self.env.sim.data.get_body_xmat(base_body_name).reshape(3, 3)
                quat = R.from_matrix(rot_mat).as_quat()
            except (AttributeError, KeyError):
                body_candidates = [frame_id, f"{robot_name}_base"]
                last_err: Optional[Exception] = None
                for body_name in body_candidates:
                    try:
                        if self._env_lock:
                            with self._env_lock:
                                pos = self.env.sim.data.get_body_xpos(body_name)
                                rot_mat = self.env.sim.data.get_body_xmat(body_name).reshape(3, 3)
                        else:
                            pos = self.env.sim.data.get_body_xpos(body_name)
                            rot_mat = self.env.sim.data.get_body_xmat(body_name).reshape(3, 3)
                        quat = R.from_matrix(rot_mat).as_quat()
                        break
                    except Exception as e:  # pragma: no cover
                        last_err = e
                else:
                    raise last_err if last_err else RuntimeError("Failed to resolve base body name")
            
            return Transform(
                translation=pos,
                rotation=quat,
                parent_frame=self.WORLD_FRAME,
                child_frame=frame_id,
                is_static=False  # Could change if robot is mobile
            )
        
        self._dynamic_frame_providers[frame_id] = get_base_transform
        self._frame_parents[frame_id] = self.WORLD_FRAME
        self._known_frames.add(frame_id)
        logger.debug(f"Registered robot base frame: {frame_id}")
    
    def _register_eef_provider(self, robot_idx: int, arm_id: str, frame_id: str) -> None:
        """Register a dynamic transform provider for an end-effector."""
        def get_eef_transform() -> Transform:
            robot = self.env.robots[robot_idx]  # type: ignore[union-attr, index]
            # robosuite provides hand pose in the ROBOT BASE frame.
            # We therefore publish base <- eef directly.
            # Parent frame is derived from the discovered TF naming, not robot.name.
            # E.g., for frame_id="robot0_right_eef" -> parent_frame="robot0_base".
            parent_frame = frame_id.replace(f"_{arm_id}_eef", "_base")

            # Use robosuite's public API to avoid underscore-prefixed internals.
            eef_body = robot.robot_model.eef_name[arm_id]
            if self._env_lock:
                with self._env_lock:
                    pose4 = robot.pose_in_base_from_name(eef_body)
            else:
                pose4 = robot.pose_in_base_from_name(eef_body)
            pos_b = np.asarray(pose4[:3, 3], dtype=np.float64)
            quat_b_e = R.from_matrix(np.asarray(pose4[:3, :3], dtype=np.float64)).as_quat()

            return Transform(
                translation=pos_b,
                rotation=quat_b_e,
                parent_frame=parent_frame,
                child_frame=frame_id,
                is_static=False,
            )
        
        self._dynamic_frame_providers[frame_id] = get_eef_transform
    
    def _register_camera_provider(self, camera_name: str, parent_frame: str) -> None:
        """
        Register a provider for a camera.
        If parent_frame is 'world', the camera is dynamic (fixed to world).
        If parent_frame is another frame (e.g. an EEF), the transform is static.
        """
        cam_mj = f"{camera_name}_mj"

        # Static transform for eye-in-hand cameras
        if parent_frame != self.WORLD_FRAME:
            # Get camera pose in world
            if self._env_lock:
                with self._env_lock:
                    self.env.sim.forward()
                    cam_pos_world = self.env.sim.data.get_camera_xpos(camera_name)
                    cam_rot_world = self.env.sim.data.get_camera_xmat(camera_name).reshape(3, 3)
            else:
                self.env.sim.forward()
                cam_pos_world = self.env.sim.data.get_camera_xpos(camera_name)
                cam_rot_world = self.env.sim.data.get_camera_xmat(camera_name).reshape(3, 3)
            
            T_world_cam = Transform(
                translation=cam_pos_world,
                rotation=R.from_matrix(cam_rot_world).as_quat(),
                parent_frame=self.WORLD_FRAME,
                child_frame=cam_mj
            )

            # Get parent pose in world
            T_world_parent = self.get_transform(self.WORLD_FRAME, parent_frame)

            # Compute T_parent_cam = (T_world_parent)^-1 * T_world_cam
            T_parent_cam = T_world_parent.inverse().compose(T_world_cam)
            
            self.add_static_transform(Transform(
                translation=T_parent_cam.translation,
                rotation=T_parent_cam.rotation,
                parent_frame=parent_frame,
                child_frame=cam_mj,
                is_static=True
            ))

        # Dynamic transform provider for fixed-to-world cameras
        else:
            def get_camera_transform() -> Transform:
                if self._env_lock:
                    with self._env_lock:
                        pos = self.env.sim.data.get_camera_xpos(camera_name)
                        rot_mat = self.env.sim.data.get_camera_xmat(camera_name).reshape(3, 3)
                else:
                    pos = self.env.sim.data.get_camera_xpos(camera_name)
                    rot_mat = self.env.sim.data.get_camera_xmat(camera_name).reshape(3, 3)

                quat = R.from_matrix(rot_mat).as_quat()
                return Transform(
                    translation=pos,
                    rotation=quat,
                    parent_frame=self.WORLD_FRAME,
                    child_frame=cam_mj,
                    is_static=False
                )
            self._dynamic_frame_providers[cam_mj] = get_camera_transform

        # Ancillary frames (camera_link, optical) are always static relative to the _mj frame
        R_cam_mj_link = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]], dtype=float)
        R_cam_mj_opt = np.diag([1.0, -1.0, -1.0])

        self.add_static_transform(
            Transform(
                translation=np.zeros(3),
                rotation=R.from_matrix(R_cam_mj_link).as_quat(),
                parent_frame=cam_mj,
                child_frame=camera_name,
                is_static=True,
            )
        )
        self.add_static_transform(
            Transform(
                translation=np.zeros(3),
                rotation=R.from_matrix(R_cam_mj_opt).as_quat(),
                parent_frame=cam_mj,
                child_frame=f"{camera_name}_optical",
                is_static=True,
            )
        )
    
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
                        parent_frame: Optional[str] = None) -> str:
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
        Get the transform that maps coordinates from source_frame into target_frame.

        In other words, this returns a Transform with:
        - parent_frame = target_frame
        - child_frame  = source_frame

        so that applying it yields:
        pose_in_target = T(target<-source) * pose_in_source
        
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
        
        # We need target<-source. _get_direct_transform(parent, child) returns parent<-child.
        # Therefore we traverse the path backwards (target -> ... -> source) and compose.
        result: Optional[Transform] = None
        for i in range(len(path) - 1, 0, -1):
            parent = path[i]
            child = path[i - 1]
            t = self._get_direct_transform(parent, child)  # parent<-child
            result = t if result is None else result.compose(t)

        if result is None:
            return Transform.identity(target_frame, source_frame)

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
                    if cached_tf.parent_frame == from_frame and cached_tf.child_frame == to_frame:
                        return cached_tf
                    elif cached_tf.parent_frame == to_frame and cached_tf.child_frame == from_frame:
                        return cached_tf.inverse()
            
            # Get fresh transform
            tf = self._dynamic_frame_providers[to_frame]()
            self._transform_cache[to_frame] = (tf, time.time())
            
            if tf.parent_frame == from_frame and tf.child_frame == to_frame:
                return tf
            elif tf.parent_frame == to_frame and tf.child_frame == from_frame:
                return tf.inverse()
        
        # Check if from_frame is dynamic
        if from_frame in self._dynamic_frame_providers:
            if from_frame in self._transform_cache:
                cached_tf, cached_time = self._transform_cache[from_frame]
                if time.time() - cached_time < self._cache_duration:
                    if cached_tf.parent_frame == from_frame and cached_tf.child_frame == to_frame:
                        return cached_tf
                    elif cached_tf.parent_frame == to_frame and cached_tf.child_frame == from_frame:
                        return cached_tf.inverse()
            
            tf = self._dynamic_frame_providers[from_frame]()
            self._transform_cache[from_frame] = (tf, time.time())
            
            if tf.parent_frame == from_frame and tf.child_frame == to_frame:
                return tf
            elif tf.parent_frame == to_frame and tf.child_frame == from_frame:
                return tf.inverse()
        
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
        
        # Get transform target<-source and apply directly
        transform = self.get_transform(target_frame, pose.frame_id)
        return transform.apply_to_pose(pose)
    
    def _compute_transform_matrix(self, source_frame: str, target_frame: str) -> np.ndarray:
        """Compute the 4x4 matrix that maps source_frame coordinates into target_frame."""
        path = self._find_path(source_frame, target_frame)
        
        if path is None:
            raise ValueError(f"No path from '{source_frame}' to '{target_frame}'")
        
        # target<-source = (target<-...) @ ... @ (...<-source)
        result = np.eye(4)
        for i in range(len(path) - 1, 0, -1):
            parent = path[i]
            child = path[i - 1]
            t = self._get_direct_transform(parent, child)  # parent<-child
            result = t.to_matrix() @ result

        return result
    
    def get_frame_pose(self, frame_id: str, reference_frame: Optional[str] = None) -> StampedPose:
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
    
    def print_frame_tree(self, root: Optional[str] = None, indent: int = 0) -> None:
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
