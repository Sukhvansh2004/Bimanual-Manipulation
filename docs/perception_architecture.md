# Perception Architecture

The perception system provides real-time object detection, segmentation, and 3D point cloud extraction for robotic manipulation.

## Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  RGB-D Camera   │────▶│  FastSAM Service │────▶│  Grasp Service  │
│  (Robosuite)    │     │  (Detection)     │     │  (GraspGen)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │                       │
        ▼                        ▼                       ▼
   RGB + Depth            Masks + Point           6-DOF Grasp
     Images               Clouds + Poses             Poses
```

## Components

### ObjectDetector

The `ObjectDetector` class handles FastSAM inference and point cloud extraction.

```python
from robosuite_franka_manipulation.perception import ObjectDetector

detector = ObjectDetector(
    camera_intrinsics=intrinsics,  # [fx, fy, cx, cy] or 3x3 matrix
    camera_width=640,
    camera_height=480,
    prompt="object",
    debug=True,
    visualize=True
)

# Run detection
detections = detector.detect_objects(
    rgb_image,       # HxWx3 uint8 array
    depth_image,     # HxW float32 array (meters)
    camera_pos,      # [x, y, z] camera position
    camera_rot       # scipy.spatial.transform.Rotation
)
```

#### Detection Pipeline

1. **FastSAM Inference**: Runs on GPU (if available) with text prompt
2. **Mask Extraction**: Gets binary masks for each detected object
3. **Point Cloud Extraction**: Vectorized depth deprojection using camera intrinsics
4. **Pose Estimation**: PCA-based 6-DOF pose from point cloud
5. **Visualization**: Optional OpenCV windows showing results

The detection prompt can be changed at runtime via `detector.set_prompt("new prompt")`.

#### Output Format

The detector returns a list of `DetectionSchema` objects (Pydantic models). Each detection contains:

```python
from robosuite_franka_manipulation.schema import DetectionSchema

# Each detection is a DetectionSchema instance:
detection = DetectionSchema(
    name='object_0',
    id=0,
    pose=StampedPose(position=[x, y, z], quaternion=[x, y, z, w], frame_id=constants.WORLD_FRAME),
    pcd=[[x1, y1, z1], ...],  # World frame point cloud (list format)
    mask=[[0, 1, ...], ...],  # Binary segmentation mask (optional)
)

# Access fields:
detection.name       # 'object_0'
detection.pose       # StampedPose instance
detection.pcd        # Point cloud as list
detection.mask       # Segmentation mask
```

### GraspPredictor

The `GraspPredictor` class wraps GraspGen for 6-DOF grasp prediction.

```python
from robosuite_franka_manipulation.perception import GraspPredictor

predictor = GraspPredictor(
    gripper_type='finger',        # or 'suction' — mapped internally via GRIPPER_TYPE_MAPPING
    model_path='',                # auto-resolved from GraspGenModels
    debug=True,
    visualize=True
)

# Predict grasps from a point cloud
grasps = predictor.predict_grasps(
    object_pose=pose_stamped,  # StampedPose object (required, first positional arg)
    num_grasps=5,              # Number of grasps to generate
    object_pcd=pcd,            # Nx3 array (optional if mesh_file given)
)

# Or predict grasps from a mesh file
grasps = predictor.predict_grasps(
    object_pose=pose_stamped,
    num_grasps=5,
    mesh_file='path/to/object.obj',
    mesh_scale=1.0,
)

# Get the gripper collision mesh (for downstream collision checking)
collision_mesh = predictor.get_gripper_collision_mesh()  # returns trimesh.Trimesh
```

#### Grasp Output Format

Grasps are returned as `GraspSchema` objects (which extend `StampedPose`):

```python
from robosuite_franka_manipulation.schema import GraspSchema

# Each grasp is a GraspSchema instance:
grasp = GraspSchema(
    pos=[x, y, z],                    # Grasp position in world frame
    quat=[qx, qy, qz, qw],            # Grasp orientation (scipy convention)
    score=0.95,                       # Confidence score [0, 1]
    id=0,                             # Grasp ID
    frame_id=constants.WORLD_FRAME,
    gripper_type='franka_panda',      # Gripper type used for prediction
    pcd_size=1024,                    # Size of input point cloud
)

# Access fields (inherited from StampedPose):
grasp.position      # np.array([x, y, z])
grasp.quaternion    # np.array([qx, qy, qz, qw])
grasp.score         # 0.95
grasp.gripper_type  # 'franka_panda'
```

## Camera Intrinsics

The system supports two intrinsics formats:

### 1D Array Format (Robosuite default)
```python
intrinsics = [fx, fy, cx, cy]  # e.g., [320, 320, 320, 240]
```

### 3x3 Matrix Format
```python
intrinsics = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])
```

The `DualArmEnvironment` automatically sets camera intrinsics from `constants.DEFAULT_CAMERA_INTRINSICS`:
```python
fx = 800.0, fy = 800.0  # Focal lengths
cx = 320.0, cy = 240.0  # Principal point (image center)
```

These values are configured in `constants.py` and can be customized for your specific camera setup.

## GPU Acceleration

The system automatically uses CUDA if available:

```python
from robosuite_franka_manipulation import constants

print(constants.DEVICE)  # 'cuda' or 'cpu'
```

FastSAM and GraspGen both run on the detected device.

## Detection Parameters

Key parameters in `constants.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `FASTSAM_CONF_THRESHOLD` | 0.9 | Minimum detection confidence |
| `FASTSAM_IOU_THRESHOLD` | 0.4 | NMS IoU threshold |
| `MAX_POINTCLOUD_RANGE` | 10.0 | Maximum depth for valid points (m) |
| `DEFAULT_DETECTION_PROMPT` | "" | Empty string detects all objects |

## Visualization

When `visualize=True`, the system shows:

1. **Raw Camera View**: Unprocessed RGB image
2. **FastSAM Segmentation**: Annotated image with masks, bounding boxes, and confidence scores

Press any key to continue when visualization windows are active.

## Point Cloud Processing

Point cloud processing is handled by utilities in `utils/pointcloud.py`:

### Depth to 3D Projection

```python
from robosuite_franka_manipulation.utils import deproject_depth_to_point_cloud

# Deproject depth image to 3D points
pcd = deproject_depth_to_point_cloud(
    depth,           # HxW depth image (meters)
    mask,            # Optional binary mask
    fx, fy, cx, cy,  # Camera intrinsics
    max_depth=10.0,
    mujoco_convention=True,   # (default) negate Y & Z for MuJoCo camera convention
    image_flipped=False       # Set True if depth was flipped from OpenGL to standard image coords
)
```

The deprojection uses the standard pinhole camera model:
```
Z = depth[y, x]
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
```

When `mujoco_convention=True` (the default), Y and Z are negated to match MuJoCo's
camera coordinate system.

When `image_flipped=True`, the function internally converts v coordinates back to the original
OpenGL space before applying the pinhole formula, ensuring correct 3D reconstruction when working
with robosuite's camera extrinsic matrices.

### Additional Point Cloud Utilities

```python
from robosuite_franka_manipulation.utils import (
    transform_point_cloud,         # Apply 4×4 transform to Nx3 point cloud
    parse_camera_intrinsics,       # Extract (fx, fy, cx, cy) from 1D or 3×3 intrinsics
    ensure_depth_2d,               # Reshape depth to (H, W)
    downsample_point_cloud,        # Farthest-point or random downsampling
    estimate_pose_from_point_cloud,# PCA-based 6-DOF pose from points
    compute_point_cloud_bounds,    # Axis-aligned bounding box
    filter_point_cloud_by_range,   # Remove points outside a distance range
    validate_point_cloud,          # Shape / NaN / size checks
)

# Camera frame to world frame
pcd_world = transform_point_cloud(pcd, camera_to_world_matrix)
```

## Grasp Selection

The perception module also exports `maximize_entropy_grasps`, which selects a
diverse subset of grasps by maximising spatial entropy:

```python
from robosuite_franka_manipulation.perception import maximize_entropy_grasps

selected = maximize_entropy_grasps(grasps, num_grasps=3, break_at_neg=False)
```

## Integration with Services

The perception components are wrapped by services for distributed operation
(see the [Services docs](services.md) for full details):

```python
from robosuite_franka_manipulation.services import FastSAMService, ServiceClient

# FastSAM Service continuously runs detection
fastsam_service = FastSAMService(env=env, visualize=True)
fastsam_service.start()

# Query detections via ServiceClient
client = ServiceClient()
response = client.call('fastsam_service', {'command': 'GET_LATEST_DETECTION'})
detections = response['detections']

# Generate grasps via Grasp Service
response = client.call('grasp_service', {
    'command': 'GET_GRASPS',
    'pcd': detections[0]['pcd'],
    'object_pose': detections[0]['pose'],
    'num_grasps': 5
})
grasps = response['grasps']
```

## Troubleshooting

### Common Issues

1. **No detections**: 
   - Check prompt matches visible objects
   - Lower `FASTSAM_CONF_THRESHOLD` if needed
   - Ensure camera has clear view of objects

2. **Invalid depth**:
   - Check depth image is 2D array
   - Verify depth values are in meters (not mm)
   - Ensure `MAX_POINTCLOUD_RANGE` is appropriate

3. **Empty point clouds**:
   - Verify camera intrinsics are correct
   - Check depth alignment with RGB
   - Ensure objects are within depth range

4. **Slow performance**:
   - Verify CUDA is being used (`constants.DEVICE == 'cuda'`)
   - Reduce image resolution if needed
   - Lower `num_grasps` for faster grasp prediction
