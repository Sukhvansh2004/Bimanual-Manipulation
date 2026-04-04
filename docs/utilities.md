# Utilities Module

The `utils` package provides reusable utility functions organized by domain. All public functions are re-exported from the package `__init__.py` for convenience.

## Overview

```python
from bimanual_manipulation.utils import (
    # SE(3) Lie algebra (from utils.se3)
    skew, unskew, SE3_log, SE3_exp, pose_to_SE3, SE3_to_pose,
    compute_se3_distance, interpolate_SE3, se3_to_matrix, matrix_to_se3,

    # Point cloud processing (from utils.pointcloud)
    downsample_point_cloud, transform_point_cloud, validate_point_cloud,
    deproject_depth_to_point_cloud, estimate_pose_from_point_cloud,
    parse_camera_intrinsics, ensure_depth_2d, compute_point_cloud_bounds,
    filter_point_cloud_by_range,

    # Environment helpers (from utils.env_utils)
    env_step, set_robot_config, ActionPersistenceManager,
    create_ycb_object, create_primitive_object,
    get_available_ycb_objects, list_ycb_objects, resolve_mesh_path,

    # Mesh utilities (from utils.mesh_utils)
    load_mesh_and_sample,

    # Motion primitives (from utils.primitives)
    DualArmHandoverPlanner, ObjectGraspPrimitive, ReleaseRetreatPrimitive,
    compute_ee_pose, extract_grasp_transform,

    # Suction gripper (from utils.suction_weld)
    apply_suction_weld, clear_suction_cache,
)
```

## SE(3) Lie Algebra Utilities (`se3.py`)

Utilities for working with rigid body transformations using the SE(3) Lie group and se(3) Lie algebra. Used throughout the planner for proper coupled rotation-translation interpolation.

### Theory

SE(3) is the Special Euclidean group representing rigid body transformations (rotation + translation). The Lie algebra se(3) is the tangent space at the identity, representing "velocities" or small transformations.

- **SE3_log**: Maps SE(3) → se(3) (matrix logarithm)
- **SE3_exp**: Maps se(3) → SE(3) (matrix exponential)

### Usage

```python
from bimanual_manipulation.utils.se3 import (
    pose_to_SE3, SE3_to_pose, SE3_log, SE3_exp,
    compute_se3_distance, interpolate_SE3, interpolate_pose,
    skew, unskew, se3_to_matrix, matrix_to_se3,
)
import numpy as np

# Convert pose to SE(3) matrix
position = [0.5, 0.0, 0.3]
quaternion = [0, 0, 0, 1]  # (x, y, z, w) — scipy convention
T = pose_to_SE3(position, quaternion)  # 4x4 matrix

# Convert back to pose
pos, quat = SE3_to_pose(T)

# Compute twist (se(3) representation) from transform
twist = SE3_log(T)  # 6D vector [omega, v]

# Compute transform from twist
T_from_twist = SE3_exp(twist)

# Compute geodesic distance between two poses
T1 = pose_to_SE3([0, 0, 0], [0, 0, 0, 1])
T2 = pose_to_SE3([1, 0, 0], [0, 0, 0.707, 0.707])  # 90° yaw
T_rel = np.linalg.inv(T1) @ T2
twist = SE3_log(T_rel)
distance = compute_se3_distance(twist, translation_weight=1.0, rotation_weight=1.0)

# Interpolate on the SE(3) manifold (4x4 matrices)
T_mid = interpolate_SE3(T1, T2, s=0.5)

# Interpolate between poses (position + quaternion)
# Supports 'linear' (decoupled lerp+slerp) and 'SE3' (geodesic) modes
mid_pos, mid_quat = interpolate_pose(
    start_pos, start_quat, end_pos, end_quat, t=0.5,
    interpolation_type='SE3',  # or 'linear' (default from constants.INTERPOLATION_TYPE)
)

# Skew-symmetric matrix (so(3) representation)
omega = [0, 0, 1]  # Rotation around z-axis
omega_hat = skew(omega)        # 3x3 skew-symmetric matrix
omega_rec = unskew(omega_hat)  # Recover vector

# Twist conversions
twist = np.array([0, 0, 1, 0.1, 0, 0])  # [omega, v]
mat = se3_to_matrix(twist)     # 4x4 se(3) matrix
twist_rec = matrix_to_se3(mat) # Recover twist
```

### Trajectory Planning

The SE(3) utilities are used in both the arm controller for smooth trajectory tracking and the RRT-Connect planner for geodesic interpolation:

```python
from bimanual_manipulation.controllers import RobosuiteArmController

# The controller uses SE(3) interpolation internally
arm = RobosuiteArmController(env, arm_id="right")
arm.move_to_pose(target_pos, target_quat)  # Uses trapezoidal velocity profile on SE(3)
```

## Point Cloud Utilities (`pointcloud.py`)

Common point cloud operations for the perception pipeline.

### Functions

```python
from bimanual_manipulation.utils.pointcloud import (
    downsample_point_cloud,
    estimate_pose_from_point_cloud,
    transform_point_cloud,
    compute_point_cloud_bounds,
    filter_point_cloud_by_range,
    validate_point_cloud,
    deproject_depth_to_point_cloud,
    ensure_depth_2d,
    parse_camera_intrinsics,
)

# Downsample point cloud
pcd = downsample_point_cloud(pcd, target_points=2048, method='random')

# Estimate 6-DOF pose using PCA
position, quaternion = estimate_pose_from_point_cloud(pcd)

# Transform point cloud with 4x4 matrix
pcd_transformed = transform_point_cloud(pcd, transform_matrix)

# Get bounding box
min_bounds, max_bounds = compute_point_cloud_bounds(pcd)

# Filter by distance from origin
pcd_filtered = filter_point_cloud_by_range(pcd, min_range=0.1, max_range=2.0)

# Validate point cloud data
is_valid = validate_point_cloud(pcd, max_range=10.0, min_points=100)

# Ensure depth map is 2D
depth_2d = ensure_depth_2d(depth, height=480, width=640)

# Parse camera intrinsics (accepts 1D [fx,fy,cx,cy] or 3x3 matrix)
fx, fy, cx, cy = parse_camera_intrinsics(intrinsics)

# Deproject depth image to 3D points
pcd = deproject_depth_to_point_cloud(
    depth,           # HxW depth image
    mask,            # Optional binary mask
    fx, fy, cx, cy,  # Camera intrinsics
    max_depth=10.0,
    mujoco_convention=False,
    image_flipped=False,
)
```

### Depth Deprojection

The deprojection uses the standard pinhole camera model:

```
Z = depth[y, x]
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
```

When `image_flipped=True`, the function internally converts v coordinates back to the original OpenGL space before applying the pinhole formula, ensuring correct 3D reconstruction when working with robosuite's camera extrinsic matrices. The system typically handles image flipping before calling this function, so it is usually called with `image_flipped=False` (default).

## Environment Utilities (`env_utils.py`)

Helpers for stepping and managing the robosuite simulation environment.

### Core Functions

```python
from bimanual_manipulation.utils.env_utils import (
    env_step,                         # Step the environment with action persistence
    set_robot_config,                 # Set robot joint configuration
    ActionPersistenceManager,         # Manage action persistence across steps
    reset_action_persistence,
    get_action_persistence_manager,
)
```

### Object Factory (`object_factory.py`)

YCB and primitive object creation utilities.

```python
from bimanual_manipulation.utils.object_factory import (
    get_available_ycb_objects,   # List available YCB objects in the dataset
    list_ycb_objects,            # Print available objects
    resolve_mesh_path,           # Get mesh file path for a YCB object
    create_ycb_object,           # Create a YCB object specification dict
    create_primitive_object,     # Create a primitive (box, cylinder, etc.) spec
    register_mesh_extraction_env,# Register env for mesh extraction
)

# List available objects
objects = get_available_ycb_objects()

# Create specs for environment construction
ycb_spec = create_ycb_object("mug", position=[0.5, 0, 0.85])
prim_spec = create_primitive_object("box", "plate", size=[0.2, 0.2, 0.02], position=[0.3, 0, 0.85])
```

### Monitors (`monitors.py`)

Runtime collision detection and object trajectory tracking.

```python
from bimanual_manipulation.utils.monitors import (
    CollisionMonitor,            # Counts collisions during execution
    ObjectTrajectoryMonitor,     # Records object poses over time
)
```

## Mesh Utilities (`mesh_utils.py`)

Mesh loading and point sampling for collision checking and grasp planning.

```python
from bimanual_manipulation.utils.mesh_utils import load_mesh_and_sample

# Load a mesh file and sample points on its surface
points = load_mesh_and_sample(mesh_path, num_samples=2000, scale=1.0)
```

## Motion Primitives (`primitives.py`)

High-level manipulation primitive classes for common dual-arm tasks.

```python
from bimanual_manipulation.utils.primitives import (
    DualArmHandoverPlanner,     # Plan and execute dual-arm handover sequences
    ObjectGraspPrimitive,       # Grasp an object with approach + close
    ReleaseRetreatPrimitive,    # Release object and retreat to safe pose
    compute_ee_pose,            # Get current end-effector pose from env
    extract_grasp_transform,    # Extract grasp → object transform
)

# Plan a handover between two arms
planner = DualArmHandoverPlanner(env, tf_manager)
# ... configure and execute handover sequence
```

## Suction Gripper Utilities (`suction_weld.py`)

Functions for managing suction gripper weld constraints in MuJoCo.

```python
from bimanual_manipulation.utils.suction_weld import (
    apply_suction_weld,    # Create a weld constraint between gripper and object
    clear_suction_cache,   # Clear cached weld data
)
```

## Meshcat ↔ MuJoCo Bridge (`meshcat_mj_utils.py`)

Low-level utilities for rendering MuJoCo model elements in Meshcat. Not re-exported from `utils/__init__.py` — import directly when needed:

```python
from bimanual_manipulation.utils.meshcat_mj_utils import (
    # Functions for loading MuJoCo geoms/bodies into Meshcat
)
```

## Integration Example

Complete example using multiple utility modules:

```python
import numpy as np
from bimanual_manipulation.utils import (
    deproject_depth_to_point_cloud,
    estimate_pose_from_point_cloud,
    transform_point_cloud,
    pose_to_SE3,
    SE3_to_pose,
    interpolate_SE3,
)
from bimanual_manipulation.transforms import StampedPose
from bimanual_manipulation import constants

# Get point cloud from depth
pcd_camera = deproject_depth_to_point_cloud(depth, mask, fx, fy, cx, cy)

# Transform to world frame
camera_to_world = np.eye(4)
camera_to_world[:3, :3] = camera_rotation_matrix
camera_to_world[:3, 3] = camera_position
pcd_world = transform_point_cloud(pcd_camera, camera_to_world)

# Estimate object pose
obj_pos, obj_quat = estimate_pose_from_point_cloud(pcd_world)

# Plan trajectory to object
start_pos, start_quat = current_eef_pose
T_start = pose_to_SE3(start_pos, start_quat)
T_goal = pose_to_SE3(obj_pos, obj_quat)

# Generate waypoints via geodesic interpolation
waypoints = []
for alpha in np.linspace(0, 1, 10):
    T_interp = interpolate_SE3(T_start, T_goal, alpha)
    pos, quat = SE3_to_pose(T_interp)
    waypoints.append(StampedPose(position=pos, quaternion=quat, frame_id="world"))
```

## Performance Notes

- All utilities use vectorized numpy operations for efficiency
- SE(3) operations avoid gimbal lock issues inherent in Euler angles
- Point cloud functions handle empty arrays gracefully
- `interpolate_pose` supports both decoupled `'linear'` (fast) and coupled `'SE3'` (geodesic) modes, selectable via `constants.INTERPOLATION_TYPE`
