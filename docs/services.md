# Microservices Architecture

The system uses a distributed microservices architecture for robotic perception and grasp planning. All services communicate via ZeroMQ (ZMQ) with JSON message serialization.

## Service Overview

| Service | Port | Purpose |
|---------|------|---------|
| Service Manager | 5555 | Service discovery and registration |
| FastSAM Service | 5551 | Object detection and segmentation |
| Grasp Service | 5552 | 6-DOF grasp prediction |
| MoveToPose Service | 5553 | Autonomous arm control |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Service Manager (5555)                      │
│                   Service Discovery & Registry                   │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ FastSAM Service │  │  Grasp Service  │  │ MoveToPose Svc  │
│     (5551)      │  │     (5552)      │  │     (5553)      │
│                 │  │                 │  │                 │
│ - Detection     │  │ - GraspGen      │  │ - Arm control   │
│ - Segmentation  │  │ - 6-DOF poses   │  │ - Move to pose  │
│ - Point clouds  │  │ - Scoring       │  │ - Cancellation  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Quick Start

The easiest way to start all services is via the main manipulation script:

```bash
python scripts/manipulation/dual_arm_manipulation.py --visualize
```

This starts all services automatically and provides an interactive shell.

### Manual Service Startup

```python
from robosuite_franka_manipulation.services import ServiceManager, ServiceClient, FastSAMService, GraspService, MoveToPoseService
from robosuite_franka_manipulation.environments import make_dual_arm_env

# Create environment
env = make_dual_arm_env()
env.reset()

# Start Service Manager (required for service discovery)
manager = ServiceManager()
manager.start()

# Start FastSAM Service
fastsam = FastSAMService(env=env, visualize=True)
fastsam.start()

# Start Grasp Service
grasp = GraspService()
grasp.start()

# Start MoveToPose Service
move = MoveToPoseService()
# Note: You must register controllers with the service
move.register_controller(0, "right", arm_controller_0)
move.start()
```

## Service Client API

The `ServiceClient` provides a unified interface to all services:

```python
from robosuite_franka_manipulation.services import ServiceClient

client = ServiceClient()

# Call any registered service
response = client.call('service_name', {'command': 'CMD', ...})
```

### Service Registration & Discovery

```python
# Register a custom service
client.register("my_service", "localhost", 5560, "custom", "My custom service")

# Look up a service
service_info = client.lookup("fastsam_service")
```

## FastSAM Service (port 5551)

Continuous object detection using FastSAM with text prompts.

### Commands

**SET_PROMPT** - Set detection text prompt
```python
response = client.call('fastsam_service', {
    'command': 'SET_PROMPT',
    'prompt': 'red cup'
})
# Response: {'status': 'success', 'message': 'Prompt updated to: red cup'}
```

**GET_PROMPT** - Get current prompt
```python
response = client.call('fastsam_service', {
    'command': 'GET_PROMPT'
})
# Response: {'status': 'success', 'prompt': 'red cup'}
```

**GET_LATEST_DETECTION** - Get latest detections
```python
response = client.call('fastsam_service', {
    'command': 'GET_LATEST_DETECTION'
})
# Response: {
#   'status': 'success',
#   'detections': [
#     {
#       'name': 'object_0',
#       'pose': {
#           'pos': [x, y, z],
#           'quat': [qx, qy, qz, qw],
#           'frame_id': constants.WORLD_FRAME,
#           'timestamp': 123.45
#       },
#       'pcd': [[x1, y1, z1], ...],
#       'mask': [[0, 1, ...], ...],
#       'id': 0
#     },
#     ...
#   ],
#   'timestamp': 1234567890.123
# }
```

### Detection Output Format

Each detection contains:
- `name`: Object identifier (e.g., "object_0")
- `pose`: Estimated 6-DOF pose dict (position, quaternion, frame_id)
- `pcd`: Nx3 list of 3D points in world frame
- `mask`: Binary segmentation mask
- `id`: Detection index

## Grasp Service (port 5552)

GraspGen-based 6-DOF grasp prediction from point clouds.

### Commands

**GET_GRASPS** - Generate grasps for a point cloud
```python
response = client.call('grasp_service', {
    'command': 'GET_GRASPS',
    'pcd': detection['pcd'],                 # Nx3 list of points
    'object_pose': detection['pose'],       # {'pos': [...], 'quat': [...]}
    'num_grasps': 5,                         # Number of grasps to generate
    'gripper_type': ['finger', 'suction'],   # List of gripper types to predict for
    'filter_intra_collision': True,          # Enable/disable intra-gripper collision checks (recursive filtering)
    'visualize_gripper_mesh': True           # Visualize collision meshes in MeshCat
})
# Response: {
#   'status': 'success',
#   'grasps': [
#     {
#       'pos': [x, y, z],
#       'quat': [qx, qy, qz, qw],
#       'score': 0.95,
#       'id': 0,
#       'frame_id': constants.WORLD_FRAME,
#       'pcd_size': 1024,
#       'prediction_timestamp': 1234567890.123
#     },
#     ...
#   ]
# }
```

### Grasp Filtering Logic
The service collects predictions from all requested gripper types, sorts them globally by confidence score (descending), and performs sequential collision checking. This ensures that the highest-scoring grasps are prioritized, and subsequent grasps are filtered against both the environment and any previously accepted grasps (preventing inter- and intra-gripper collisions).

### Supported Gripper Types

| Type | GraspGen Config |
|------|----------------|
| `finger` | franka_panda |
| `suction` | single_suction_cup_30mm |

## MoveToPose Service (port 5553)

Autonomous robot arm control with trajectory following support.

### Commands

**MOVE_TO_POSE** - Move an arm to a target pose
```python
response = client.call('move_to_pose_service', {
    'command': 'MOVE_TO_POSE',
    'robot_id': 0,               # Robot index (0 or 1)
    'arm': 'right',              # Arm ID
    'pose': {                    # StampedPose dict
        'pos': [x, y, z],
        'quat': [qx, qy, qz, qw],
        'frame_id': 'world'
    },
    'gripper_action': 0.0,       # Gripper action during movement
    'approach': True             # Use approach motion (optional)
})
# Response: {'status': 'success', 'message': 'Move started'}
```

**FOLLOW_TRAJECTORY** - Follow a pre-defined trajectory
```python
response = client.call('move_to_pose_service', {
    'command': 'FOLLOW_TRAJECTORY',
    'robot_id': 0,
    'arm': 'right',
    'trajectory': [              # List of StampedPose dicts
        {'pos': [x1, y1, z1], 'quat': [qx, qy, qz, qw], 'frame_id': 'world'},
        {'pos': [x2, y2, z2], 'quat': [qx, qy, qz, qw], 'frame_id': 'world'},
    ],
    'gripper_action': 0.0
})
# Response: {'status': 'success', 'message': 'Trajectory started'}
```

**CANCEL_MOVE** - Cancel current movement
```python
response = client.call('move_to_pose_service', {
    'command': 'CANCEL_MOVE',
    'robot_id': 0,
    'arm': 'right'
})
# Response: {'status': 'success', 'message': 'Cancellation requested'}
```

**GET_ARM_STATUS** - Get current arm movement status
```python
response = client.call('move_to_pose_service', {
    'command': 'GET_ARM_STATUS',
    'robot_id': 0,
    'arm': 'right'
})
# Response: {'status_code': 1}  # 1=success, 0=failure, -1=moving
```

## Configuration Constants

All service ports and timeouts are defined in `constants.py`:

```python
# Network Configuration
DEFAULT_HOST = 'localhost'
SERVICE_MANAGER_PORT = 5555
FASTSAM_SERVICE_PORT = 5551
GRASP_SERVICE_PORT = 5552
MOVE_TO_POSE_SERVICE_PORT = 5553

# ZMQ Timeouts (milliseconds)
ZMQ_POLL_TIMEOUT = 1000          # 1 second for service polling
ZMQ_MANAGER_TIMEOUT = 330000     # 5.5 minutes for manager calls
ZMQ_SERVICE_CALL_TIMEOUT = 300000  # 5 minutes for service calls
ZMQ_SERVER_RECV_TIMEOUT = 1000   # 1 second for server receive polling
```

## Error Handling

All services return error responses in a consistent format:

```python
{
    'status': 'error',
    'message': 'Description of the error'
}
```

Common error scenarios:
- Service not registered: Check if ServiceManager is running
- Timeout: Service may be busy or not responding
- Invalid command: Check command spelling and required parameters

## Thread Safety

The manipulation system uses threading locks for environment access:

```python
import threading

# Create a shared lock for environment access
env_lock = threading.RLock()

# TransformManager accepts the lock
tf_manager = TransformManager(env, env_lock=env_lock)

# FastSAM service uses the lock via tf_manager
fastsam = FastSAMService(tf_manager=tf_manager, visualize=True)

# MoveToPose controllers also use locks
arm_controller = RobosuiteArmController(env, arm_id="right", env_lock=env_lock)
```

This prevents race conditions when multiple services access the robosuite environment simultaneously.
