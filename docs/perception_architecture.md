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

#### Output Format

```python
detection = {
    'name': 'object_0',
    'pose': {
        'pos': np.array([x, y, z]),      # World frame position
        'quat': np.array([qx, qy, qz, qw])  # Quaternion orientation
    },
    'pcd': np.array([[x1,y1,z1], ...]),   # World frame point cloud
    'pcd_local': np.array([[...], ...]),  # Object-local point cloud
    'mask': np.array([[0,1,...], ...]),   # Binary segmentation mask
    'id': 0
}
```

### GraspPredictor

The `GraspPredictor` class wraps GraspGen for 6-DOF grasp prediction.

```python
from robosuite_franka_manipulation.perception import GraspPredictor

predictor = GraspPredictor(
    gripper_type='franka_panda',  # or 'single_suction_cup_30mm'
    debug=True,
    visualize=True
)

# Predict grasps
grasps = predictor.predict_grasps(
    object_pcd=point_cloud,    # Nx3 array
    object_pose=pose_dict,     # {'pos': [...], 'quat': [...]}
    num_grasps=5
)
```

#### Grasp Output Format

```python
grasp = {
    'pos': np.array([x, y, z]),         # Grasp position in world frame
    'quat': np.array([qx, qy, qz, qw]), # Grasp orientation
    'score': 0.95                        # Confidence score [0, 1]
}
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

The `DualArmEnvironmentWrapper` automatically sets realistic intrinsics:
- `fx = width / 2`
- `fy = height / 2`
- `cx = width / 2`
- `cy = height / 2`

## GPU Acceleration

The system automatically uses CUDA if available:

```python
from robosuite_franka_manipulation import constants

print(constants.DEVICE)  # 'cuda' or 'cpu'
```

FastSAM and GraspGen both run on the detected device.

## Detection Parameters

Key parameters in `constants/__init__.py`:

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
    max_depth=10.0
)
```

The deprojection uses standard pinhole camera model:
```
Z = depth[y, x]
X = (x - cx) * Z / fx
Y = (y - cy) * Z / fy
```

### Frame Transformations

```python
from robosuite_franka_manipulation.utils import transform_point_cloud

# Camera frame to world frame
pcd_world = transform_point_cloud(pcd_camera, camera_to_world_matrix)
```

## Integration with Services

The perception components are wrapped by services for distributed operation:

```python
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
    'pointcloud': detections[0]['pointcloud'],
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
