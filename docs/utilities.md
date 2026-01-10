# Utilities Module

The `utils` module provides reusable utility functions organized by domain. These utilities are used throughout the codebase and can be imported for custom applications.

## Overview

```python
from robosuite_franka_manipulation.utils import (
    # SE(3) Lie algebra
    skew, unskew, SE3_log, SE3_exp, pose_to_SE3, SE3_to_pose,
    compute_se3_distance, interpolate_SE3,
    # Point cloud processing
    downsample_point_cloud, transform_point_cloud, validate_point_cloud,
    deproject_depth_to_point_cloud, estimate_pose_from_point_cloud,
    # Serialization
    to_json_serializable, convert_grasps_to_json, convert_detections_to_json,
)
```

## SE(3) Lie Algebra Utilities

The `se3` module provides utilities for working with rigid body transformations using Lie algebra. This enables proper trajectory interpolation on the SE(3) manifold.

### Theory

SE(3) is the Special Euclidean group representing rigid body transformations (rotation + translation). The Lie algebra se(3) is the tangent space at the identity, representing "velocities" or small transformations.

- **SE3_log**: Maps SE(3) → se(3) (matrix logarithm)
- **SE3_exp**: Maps se(3) → SE(3) (matrix exponential)

### Usage

```python
from robosuite_franka_manipulation.utils import (
    pose_to_SE3, SE3_to_pose, SE3_log, SE3_exp,
    compute_se3_distance, interpolate_SE3, skew
)
import numpy as np

# Convert pose to SE(3) matrix
position = [0.5, 0.0, 0.3]
quaternion = [0, 0, 0, 1]  # (x, y, z, w)
T = pose_to_SE3(position, quaternion)  # 4x4 matrix

# Convert back to pose
pos, quat = SE3_to_pose(T)

# Compute geodesic distance between two poses
T1 = pose_to_SE3([0, 0, 0], [0, 0, 0, 1])
T2 = pose_to_SE3([1, 0, 0], [0, 0, 0.707, 0.707])  # 90° yaw
distance = compute_se3_distance(T1, T2)

# Interpolate between poses (geodesic on SE(3) manifold)
T_interp = interpolate_SE3(T1, T2, alpha=0.5)  # Midpoint

# Skew-symmetric matrix (so(3) representation)
omega = [0, 0, 1]  # Rotation around z-axis
omega_hat = skew(omega)  # 3x3 skew-symmetric matrix
```

### Trajectory Planning

The SE(3) utilities are used in the arm controller for smooth trajectory planning:

```python
from robosuite_franka_manipulation.controllers import RobosuiteArmController

# The controller uses SE(3) interpolation internally
arm = RobosuiteArmController(env, arm_id="right")
arm.move_to_pose(target_pos, target_quat)  # Uses trapezoidal velocity profile on SE(3)
```

## Point Cloud Utilities

The `pointcloud` module provides common point cloud operations.

### Functions

```python
from robosuite_franka_manipulation.utils import (
    downsample_point_cloud,
    estimate_pose_from_point_cloud,
    transform_point_cloud,
    compute_point_cloud_bounds,
    filter_point_cloud_by_range,
    validate_point_cloud,
    deproject_depth_to_point_cloud,
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

# Deproject depth image to 3D points
pcd = deproject_depth_to_point_cloud(
    depth,           # HxW depth image
    mask,            # Optional binary mask
    fx, fy, cx, cy,  # Camera intrinsics
    max_depth=10.0,
    mujoco_convention=False,  # standard pinhole / optical frame (x right, y down, z forward)
    image_flipped=True  # Set True if depth was flipped from OpenGL to standard image coords
)
```

### Depth Deprojection

The deprojection uses standard pinhole camera model:

```
Z = depth[y, x]
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
```

When `image_flipped=True`, the function internally converts v coordinates back to the original
OpenGL space before applying the pinhole formula, ensuring correct 3D reconstruction when working
with robosuite's camera extrinsic matrices.

## Serialization Utilities

The `serialization` module handles JSON/numpy conversion for ZMQ communication.

### Functions

```python
from robosuite_franka_manipulation.utils import (
    to_json_serializable,
    from_json_to_numpy,
    convert_grasps_to_json,
    convert_detections_to_json,
    convert_pose_to_json,
    numpy_to_list,
    ensure_numpy,
)

# Convert any object to JSON-serializable format
data = {'array': np.array([1, 2, 3]), 'value': np.float32(1.5)}
json_data = to_json_serializable(data)

# Convert JSON back to numpy
numpy_data = from_json_to_numpy(json_data)

# Convert grasp list for ZMQ transmission
grasps = [{'pos': np.array([0, 0, 0]), 'quat': np.array([0, 0, 0, 1]), 'score': 0.9}]
json_grasps = convert_grasps_to_json(grasps)

# Convert detections for ZMQ transmission
detections = [{'pose': {'pos': [...], 'quat': [...]}, 'pcd': np.array([...])}]
json_detections = convert_detections_to_json(detections)

# Ensure value is numpy array
arr = ensure_numpy([1, 2, 3])  # Returns np.array([1, 2, 3])
```

## Integration Example

Complete example using all utilities:

```python
import numpy as np
from robosuite_franka_manipulation.utils import (
    deproject_depth_to_point_cloud,
    estimate_pose_from_point_cloud,
    transform_point_cloud,
    pose_to_SE3,
    SE3_to_pose,
    interpolate_SE3,
    convert_grasps_to_json,
)
from robosuite_franka_manipulation.transforms import Transform

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

# Generate waypoints
waypoints = []
for alpha in np.linspace(0, 1, 10):
    T_interp = interpolate_SE3(T_start, T_goal, alpha)
    pos, quat = SE3_to_pose(T_interp)
    waypoints.append({'pos': pos, 'quat': quat})

# Convert for transmission
json_waypoints = convert_grasps_to_json(waypoints)
```

## Performance Notes

- All utilities use vectorized numpy operations for efficiency
- SE(3) operations avoid gimbal lock issues inherent in Euler angles
- Point cloud functions handle empty arrays gracefully
- Serialization handles nested structures recursively
