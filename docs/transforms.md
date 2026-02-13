# Transforms Module

The transforms module provides coordinate frame management similar to ROS tf/tf2. It allows you to:
- Track poses with their associated coordinate frames
- Transform poses between different frames
- Query dynamic transforms from the simulation (cameras, end-effectors)
- Build and traverse frame trees

## Quick Start

```python
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation import constants

# Create environment and transform manager
env = make_dual_arm_env()
env.reset()

# Initialize with auto-discovery enabled
tf_manager = TransformManager(
    env=env,
    auto_discover=True,  # Automatically discover robot and camera frames
    env_lock=None        # Optional lock for thread-safe simulation access
)

# Print discovered frames
print(tf_manager.get_known_frames())
# ['agentview', 'frontview', 'robot0_base', 'robot0_left_eef', 'robot0_right_eef', 'world']

# Print frame tree
tf_manager.print_frame_tree()
# world
#   robot0_base
#     robot0_left_eef
#     robot0_right_eef
#   agentview
#   frontview
```

## Core Classes

### StampedPose

A pose (position + orientation) with an associated coordinate frame and timestamp. It is a Pydantic model with field aliases for flexibility.

```python
class StampedPose(BaseModel):
    pos: np.ndarray         # [x, y, z] (alias: 'position')
    quat: np.ndarray        # [x, y, z, w] - scipy convention (alias: 'quaternion')
    frame_id: str = constants.WORLD_FRAME # Frame this pose is expressed in
    timestamp: float = time.time()        # Time of creation

    # Properties for convenient access
    @property
    def position(self) -> np.ndarray: ...
    @property
    def quaternion(self) -> np.ndarray: ...
    
    # Class methods
    @classmethod
    def from_matrix(cls, matrix: np.ndarray, frame_id: str) -> "StampedPose": ...
    def to_matrix(self) -> np.ndarray: ...
    def copy(self) -> "StampedPose": ...
```

Usage:

```python
from robosuite_franka_manipulation.transforms import StampedPose
from robosuite_franka_manipulation import constants

# Create a pose using aliases (pos/quat) or property names (position/quaternion)
camera_pose = StampedPose(
    pos=[0.5, 0.0, 0.3],
    quat=[0, 0, 0, 1],
    frame_id="agentview"
)

# Alternative: using property names
camera_pose = StampedPose(
    position=[0.5, 0.0, 0.3],
    quaternion=[0, 0, 0, 1],
    frame_id="agentview"
)

# Access fields using either name
print(camera_pose.pos)       # np.array([0.5, 0., 0.3])
print(camera_pose.position)  # Same as .pos

# Create from 4x4 transformation matrix
matrix = np.eye(4)
matrix[:3, 3] = [1, 2, 3]
pose = StampedPose.from_matrix(matrix, constants.WORLD_FRAME)

# Convert to matrix
T = pose.to_matrix()  # Returns 4x4 homogeneous matrix

# Create from dictionary (for API responses)
detection = {'pos': [0.5, 0.0, 0.3], 'quat': [0, 0, 0, 1], 'frame_id': constants.WORLD_FRAME}
pose = StampedPose(**detection)
```

### Transform

A transformation between two coordinate frames. Similar to `geometry_msgs/TransformStamped` in ROS.

```python
@dataclass
class Transform:
    translation: np.ndarray # [x, y, z]
    rotation: np.ndarray    # [x, y, z, w]
    parent_frame: str      # Source/Reference frame
    child_frame: str       # Target frame
    timestamp: float       # Time of creation
    is_static: bool = False # Whether transform is constant
```

Usage:

```python
from robosuite_franka_manipulation.transforms import Transform
from robosuite_franka_manipulation import constants
import numpy as np

# Create a transform from world to object frame
transform = Transform(
    translation=[0.5, 0.0, 0.3],
    rotation=[0, 0, 0, 1],
    parent_frame=constants.WORLD_FRAME,
    child_frame="object_0",
    is_static=False
)
```

### TransformManager

Central manager for coordinate frame transformations. Maintains a tree of frames and provides transform lookup.

```python
from robosuite_franka_manipulation.transforms import TransformManager
from robosuite_franka_manipulation import constants
import threading

# Create with environment (auto-discovers frames)
env_lock = threading.RLock()  # Optional lock for thread safety
tf_manager = TransformManager(env, auto_discover=True, env_lock=env_lock)

# Check available frames
frames = tf_manager.get_known_frames()
tf_manager.frame_exists("agentview")  # True

# Get transform between any two frames
transform = tf_manager.get_transform(constants.WORLD_FRAME, "agentview")

# Transform a pose between frames
camera_pose = StampedPose(pos=[0.5, 0.0, 0.3], quat=[0, 0, 0, 1], frame_id="agentview")
world_pose = tf_manager.transform_pose(camera_pose, constants.WORLD_FRAME)

# Get current pose of a frame
eef_pose = tf_manager.get_frame_pose("robot0_right_eef", constants.WORLD_FRAME)
print(f"EEF position: {eef_pose.position}")

# Add static transforms
tf_manager.add_static_transform(Transform(
    translation=[1, 0, 0],
    rotation=[0, 0, 0, 1],
    parent_frame="world",
    child_frame="custom_frame"
))
```

## Standard Frame Hierarchy

The transform manager automatically discovers these frames from robosuite:

```
world (root)
├── robot0_base (usually identity to world)
│   ├── robot0_left_eef
│   └── robot0_right_eef
├── agentview (camera)
├── frontview (camera)
├── birdview (camera)
└── object_* (detected objects, added dynamically)
```

## Working with Objects

Register detected objects as frames:

```python
# Add an object frame
detection = {'pos': [0.5, 0.0, 0.1], 'quat': [0, 0, 0, 1]}
pose = StampedPose.from_dict(detection, constants.WORLD_FRAME)
frame_name = tf_manager.add_object_frame("cube", pose)  # Returns "object_cube"

# Update object frame as it moves
new_pose = StampedPose([0.6, 0.1, 0.1], [0, 0, 0, 1], constants.WORLD_FRAME)
tf_manager.update_object_frame("cube", new_pose)

# Transform to object's local frame
world_point = StampedPose([0.55, 0.05, 0.15], [0, 0, 0, 1], constants.WORLD_FRAME)
object_local = tf_manager.transform_pose(world_point, "object_cube")
```

## Frame Utilities

Helper functions for common operations:

```python
from robosuite_franka_manipulation.transforms import (
    detection_to_stamped_pose,
    grasp_to_stamped_pose,
    register_detection_frames,
    transform_grasps_to_frame,
    get_eef_pose,
    compute_approach_pose,
)

# Convert detection dict to StampedPose
detection = {'pose': {'pos': [0.5, 0, 0.1], 'quat': [0, 0, 0, 1]}}
pose = detection_to_stamped_pose(detection, constants.WORLD_FRAME)

# Register multiple detections
frame_names = register_detection_frames(tf_manager, detections, constants.WORLD_FRAME)

# Transform grasp list to different frame
robot_frame_grasps = transform_grasps_to_frame(
    tf_manager, grasps, constants.WORLD_FRAME, "robot0_base"
)

# Get current EEF pose
eef_pose = get_eef_pose(tf_manager, "robot0", "right", constants.WORLD_FRAME)

# Compute approach pose (offset along grasp z-axis)
approach = compute_approach_pose(grasp_pose, approach_distance=0.1, approach_axis='z')
```

## Integration with Manipulation

Example of using transforms for pick-and-place:

```python
from robosuite_franka_manipulation.transforms import TransformManager, compute_approach_pose
from robosuite_franka_manipulation.controllers import RobosuiteArmController

# Setup
tf_manager = TransformManager(env)
arm_controller = RobosuiteArmController(env, arm_id="right")

# Get grasp pose (from grasp service or predictor)
grasp_pose = StampedPose.from_dict(grasp, constants.WORLD_FRAME)

# Compute approach pose
approach_pose = compute_approach_pose(grasp_pose, approach_distance=0.1)

# Execute approach
arm_controller.move_to_pose(
    target_pos=approach_pose.position,
    target_quat=approach_pose.quaternion
)

# Execute grasp
arm_controller.move_to_pose(
    target_pos=grasp_pose.position,
    target_quat=grasp_pose.quaternion
)

# Close gripper...

# Lift (transform grasp pose up in world frame)
lift_pose = StampedPose(
    position=grasp_pose.position + [0, 0, 0.1],  # Lift 10cm
    quaternion=grasp_pose.quaternion,
    frame_id=constants.WORLD_FRAME
)
arm_controller.move_to_pose(lift_pose.position, lift_pose.quaternion)
```

## Debugging

```python
# Print frame tree
tf_manager.print_frame_tree()

# Get all known frames
print(tf_manager.get_known_frames())

# Check if path exists between frames
try:
    tf = tf_manager.get_transform("frame_a", "frame_b")
except ValueError as e:
    print(f"No path: {e}")

# Clear transform cache (for debugging stale data)
tf_manager.clear_cache()
```
