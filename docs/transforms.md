# Transforms Module

The transforms module provides coordinate frame management similar to ROS tf/tf2. It allows you to:
- Track poses with their associated coordinate frames
- Transform poses between different frames
- Query dynamic transforms from the simulation (cameras, end-effectors)
- Build and traverse frame trees
- Stream live frame axes to Meshcat for debugging

All classes are exported from `robosuite_franka_manipulation.transforms`:

```python
from robosuite_franka_manipulation.transforms import (
    StampedPose,
    Transform,
    TransformManager,
    MeshcatTFVisualizerConfig,
    stream_transforms_to_meshcat,
)
```

## Quick Start

```python
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation import constants

env = make_dual_arm_env()
env.reset()

# Initialize with auto-discovery enabled
tf_manager = TransformManager(
    env=env,
    auto_discover=True,  # Automatically discover robot and camera frames
    env_lock=None        # Optional RLock for thread-safe simulation access
)

# Print discovered frames
print(tf_manager.get_known_frames())

# Print frame tree
tf_manager.print_frame_tree()
```

---

## Core Classes

### StampedPose

A Pydantic `BaseModel` holding a position, orientation quaternion, coordinate
frame, and optional timestamp.

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `pos` | `np.ndarray` (3,) | — | Accepts alias `position` on **input** |
| `quat` | `np.ndarray` (4,) | — | Accepts alias `quaternion` on **input** — `[x, y, z, w]` (scipy convention) |
| `frame_id` | `str` | `constants.WORLD_FRAME` | Coordinate frame the pose is expressed in |
| `timestamp` | `float` | `time.time()` | Auto-set at creation |

Quaternions are automatically normalised on assignment.  NumPy arrays are
converted from any array-like input and made immutable.

```python
# Both styles are equivalent for construction:
pose_a = StampedPose(pos=[0.5, 0.0, 0.3], quat=[0, 0, 0, 1], frame_id="agentview")
pose_b = StampedPose(position=[0.5, 0.0, 0.3], quaternion=[0, 0, 0, 1], frame_id="agentview")

# Read-only aliases
pose_a.position   # same as pose_a.pos
pose_a.quaternion  # same as pose_a.quat

# Writable via property setters
pose_a.position = [0.6, 0.0, 0.3]
```

#### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `from_matrix(matrix, frame_id)` | `StampedPose` | Class method — construct from a 4×4 homogeneous matrix |
| `to_matrix()` | `np.ndarray` (4, 4) | Convert to homogeneous transformation matrix |
| `to_dict()` | `dict` | Serialize via Pydantic `model_dump()` |
| `copy()` | `StampedPose` | Deep copy via `model_copy(deep=True)` |

```python
# From / to 4×4 matrices
T = np.eye(4); T[:3, 3] = [1, 2, 3]
pose = StampedPose.from_matrix(T, constants.WORLD_FRAME)
T_back = pose.to_matrix()

# From a dictionary (e.g., API responses)
d = {'pos': [0.5, 0.0, 0.3], 'quat': [0, 0, 0, 1], 'frame_id': 'world'}
pose = StampedPose(**d)
```

> **Note:** `StampedPose` is a Pydantic model — it does **not** accept
> positional arguments.  Always use keyword arguments.

---

### Transform

A dataclass describing a rigid transformation between two coordinate frames
(analogous to `geometry_msgs/TransformStamped` in ROS).

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `translation` | `np.ndarray` (3,) | — | `[x, y, z]` |
| `rotation` | `np.ndarray` (4,) | — | `[x, y, z, w]` (scipy convention) |
| `parent_frame` | `str` | — | Reference frame |
| `child_frame` | `str` | — | Frame described relative to parent |
| `timestamp` | `Optional[float]` | `None` | Auto-set to `time.time()` in `__post_init__` |
| `is_static` | `bool` | `False` | Whether this transform is constant |

Arrays are converted to `np.float64` and quaternions normalised in `__post_init__`.

#### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `identity(parent, child, is_static=True)` | `Transform` | Class method — create the identity transform |
| `from_matrix(matrix, parent, child, is_static=False)` | `Transform` | Class method — create from a 4×4 matrix |
| `to_matrix()` | `np.ndarray` (4, 4) | Convert to 4×4 homogeneous matrix |
| `inverse()` | `Transform` | Return the inverse transform with frames swapped |
| `compose(other)` | `Transform` | Compose `self * other` (`self.child_frame` must equal `other.parent_frame`) |
| `apply_to_pose(pose)` | `StampedPose` | Apply this transform to a pose (pose must be in `child_frame`) |

```python
from robosuite_franka_manipulation.transforms import Transform

# Create a transform
tf = Transform(
    translation=[0.5, 0.0, 0.3],
    rotation=[0, 0, 0, 1],
    parent_frame="world",
    child_frame="object_0",
)

# Inverse
tf_inv = tf.inverse()  # child_frame ↔ parent_frame swapped

# Compose
tf_ab = tf_a.compose(tf_b)   # tf_a.child_frame must == tf_b.parent_frame

# Apply to a pose
pose_in_child = StampedPose(pos=[0.1, 0, 0], quat=[0, 0, 0, 1], frame_id="object_0")
pose_in_parent = tf.apply_to_pose(pose_in_child)
```

---

### TransformManager

Central class that maintains a tree of frames with BFS-based transform lookup
and a 10 ms cache for dynamic providers.

```python
tf_manager = TransformManager(env, auto_discover=True, env_lock=env_lock)
```

#### Public Methods

| Method | Description |
|--------|-------------|
| `get_known_frames()` | Return a sorted list of all known frame names |
| `frame_exists(frame_id)` | Check if a frame is registered |
| `get_transform(target_frame, source_frame)` | Get the `Transform` mapping source → target (BFS path) |
| `transform_pose(pose, target_frame)` | Transform a `StampedPose` into a different frame |
| `get_frame_pose(frame_id, reference_frame=None)` | Get pose of a frame in `reference_frame` (defaults to world) |
| `add_static_transform(transform)` | Register a constant transform |
| `add_dynamic_transform(frame_id, parent_frame, provider)` | Register a callable that returns a fresh `Transform` on each query |
| `add_object_frame(object_id, pose, parent_frame=None)` | Create an `object_{id}` frame from a detection pose; returns frame name |
| `update_object_frame(object_id, pose)` | Update an existing object frame (creates it if missing) |
| `remove_frame(frame_id)` | Remove a frame and its static transform |
| `clear_cache()` | Flush the dynamic-transform cache (10 ms TTL) |
| `print_frame_tree(root=None, indent=0)` | Pretty-print the frame tree to stdout |

```python
# Transform a pose between frames
camera_pose = StampedPose(pos=[0.5, 0.0, 0.3], quat=[0, 0, 0, 1], frame_id="agentview")
world_pose = tf_manager.transform_pose(camera_pose, constants.WORLD_FRAME)

# Get current EEF pose (reference_frame defaults to world)
eef_pose = tf_manager.get_frame_pose("robot0_right_eef")

# Register a detected object
frame_name = tf_manager.add_object_frame("cube", detection_pose)  # "object_cube"
tf_manager.update_object_frame("cube", new_pose)

# Custom static frame
tf_manager.add_static_transform(Transform(
    translation=[1, 0, 0],
    rotation=[0, 0, 0, 1],
    parent_frame="world",
    child_frame="my_custom_frame",
    is_static=True,
))
```

---

## Standard Frame Hierarchy

When `auto_discover=True`, the manager discovers frames from robosuite.
Cameras are wrapped in a three-level chain so that the final `*_optical`
frame follows the OpenCV convention (z-forward, x-right, y-down).

```
world (root)
├── robot0_base
│   ├── robot0_left_eef
│   └── robot0_right_eef
├── robot1_base                       (if dual-arm)
│   ├── robot1_left_eef
│   └── robot1_right_eef
├── agentview_mj          ← raw MuJoCo camera pose
│   └── agentview          ← renamed / adjusted
│       └── agentview_optical  ← OpenCV optical frame
├── frontview_mj
│   └── frontview
│       └── frontview_optical
├── birdview_mj
│   └── birdview
│       └── birdview_optical
├── robot0_eye_in_hand_mj   (eye-in-hand cameras)
│   └── robot0_eye_in_hand
│       └── robot0_eye_in_hand_optical
└── object_* (added dynamically via add_object_frame)
```

---

## Meshcat TF Visualizer

`stream_transforms_to_meshcat` runs a continuous loop that draws coloured
coordinate axes and optional text labels for every frame in the
`TransformManager`, streamed to a Meshcat viewer over ZMQ.

```python
import threading
from robosuite_franka_manipulation.transforms import (
    stream_transforms_to_meshcat,
    MeshcatTFVisualizerConfig,
)

config = MeshcatTFVisualizerConfig(
    fps=30.0,                   # Update rate
    zmq_url="tcp://127.0.0.1:6000",
    clear=True,                 # Clear scene on start
    axis_length=0.12,
    axis_radius=0.008,
    labels=True,                # Draw text labels next to axes
    label_offset=0.05,          # Label Z offset (m)
    visualize_scene=True,       # Also render robot meshes / geoms
)

stop = threading.Event()
t = threading.Thread(
    target=stream_transforms_to_meshcat,
    kwargs=dict(
        env=env,
        env_lock=env_lock,
        tf_manager=tf_manager,
        config=config,
        stop_event=stop,
    ),
    daemon=True,
)
t.start()

# ... later ...
stop.set()  # gracefully stop the visualizer thread
```

Frames are grouped by type in the Meshcat scene tree:

| Group | Frames |
|-------|--------|
| `world` | World origin |
| `robots/` | Robot base and EEF frames |
| `cameras/raw` | `*_mj` raw MuJoCo camera frames |
| `cameras/adjusted` | Named camera frames (e.g. `agentview`) |
| `cameras/optical` | `*_optical` OpenCV-convention frames |
| `objects/` | Object frames |

---

## Working with Objects

```python
# Add an object frame from a detection
detection = {'pos': [0.5, 0.0, 0.1], 'quat': [0, 0, 0, 1]}
pose = StampedPose(**detection, frame_id=constants.WORLD_FRAME)
frame_name = tf_manager.add_object_frame("cube", pose)  # "object_cube"

# Update as the object moves
new_pose = StampedPose(pos=[0.6, 0.1, 0.1], quat=[0, 0, 0, 1], frame_id=constants.WORLD_FRAME)
tf_manager.update_object_frame("cube", new_pose)

# Transform a world-frame point into the object's local frame
world_point = StampedPose(pos=[0.55, 0.05, 0.15], quat=[0, 0, 0, 1], frame_id=constants.WORLD_FRAME)
object_local = tf_manager.transform_pose(world_point, "object_cube")
```

---

## Debugging

```python
# Print the full frame tree
tf_manager.print_frame_tree()

# List all known frames
print(tf_manager.get_known_frames())

# Check whether a path exists
try:
    tf = tf_manager.get_transform("frame_a", "frame_b")
except ValueError as e:
    print(f"No path: {e}")

# Clear the dynamic-transform cache (10 ms TTL)
tf_manager.clear_cache()
```
