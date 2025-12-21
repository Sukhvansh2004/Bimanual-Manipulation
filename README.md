# Dual-Arm Franka Manipulation

A comprehensive framework for dual-arm Franka robot manipulation with advanced perception, teleoperation, and autonomous control modes.

## Features

- **Dual-Arm Control**: Coordinated manipulation with two Franka Panda robots
- **Advanced Perception**: FastSAM-based object segmentation with text prompts (GPU-accelerated)
- **Grasp Generation**: GraspGen-powered 6-DOF grasp prediction
- **Microservices Architecture**: Distributed services (FastSAM, Grasp, EEF) with ZMQ interfaces
- **Teleoperation**: Keyboard-based manual control with robosuite integration
- **Interactive Shell**: Python REPL for real-time service queries and control
- **Standalone Operation**: Works out-of-the-box without external ROS dependencies

## Installation

### Quick Install

```bash
./install.sh
```

This will:
- Create the `dual_arm_manipulation` conda environment
- Install all dependencies including GraspGen
- Install the package in development mode

**Note**: GraspGenModels (pre-trained models) need to be downloaded separately.

### Manual Installation

1. **Install System Dependencies** (Linux):
```bash
sudo apt-get update
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libosmesa6-dev build-essential cmake
```

2. **Create Conda Environment**:
```bash
conda create -n dual_arm_manipulation python=3.10
conda activate dual_arm_manipulation
export MUJOCO_GL=osmesa
```

3. **Install PyTorch** (adjust for your CUDA version):
```bash
pip install torch==2.1.0 torchvision==0.16.0 torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

4. **Install GraspGen**:
```bash
cd deps/GraspGen
pip install -e .
```

5. **Download GraspGen Models**:
```bash
git clone https://huggingface.co/adithyamurali/GraspGenModels deps/GraspGenModels
```

6. **Install the Package**:
```bash
pip install -e .
```

## Project Structure

```
├── src/robosuite_franka_manipulation/    # Main package
│   ├── constants/                        # Configuration constants
│   ├── controllers/                      # Arm and gripper controllers
│   ├── environments/                     # Robosuite environment factory
│   ├── perception/                       # Object detection and grasp prediction
│   ├── services/                         # Microservices
│   │   ├── service_manager.py           # Service discovery (port 5555)
│   │   ├── fastsam_service.py           # FastSAM detection (port 5551)
│   │   ├── grasp_service.py             # GraspGen prediction (port 5552)
│   │   └── eef_service.py               # End-effector coordinates (port 5559)
│   └── teleop/                           # Keyboard teleoperation
├── scripts/
│   ├── demos/                            # Demo scripts (go_to_pose, gripper_move)
│   ├── manipulation/                     # Main manipulation script
│   └── teleop/                           # Standalone teleoperation
├── tests/                                # Pytest test suite
├── docs/                                 # Documentation
├── models/                               # Pre-trained models (FastSAM-x.pt)
└── deps/                                 # Dependencies (GraspGen, GraspGenModels)
```

## Usage

### Main Script: Dual-Arm Manipulation System

The primary entry point starts all services and provides an interactive shell:

```bash
python scripts/manipulation/dual_arm_manipulation.py --visualize
```

**Arguments**:
- `--visualize`: Enable OpenCV visualization of FastSAM detections
- `--debug`: Enable debug logging
- `--prompt <text>`: Set initial detection prompt (default: "object")

This starts:
1. Dual-arm robosuite environment with visualization
2. Service Manager for service discovery
3. FastSAM Service for object detection
4. Grasp Service for grasp prediction
5. EEF Service for end-effector coordinates
6. Keyboard teleoperation (main thread)
7. Interactive Python shell (background thread)

### Interactive Shell Commands

Once the system is running, use these in the interactive shell:

```python
# Detect objects in the scene
>>> detections = detect()

# Generate grasps for a detection
>>> grasps = get_grasps(detections[0])

# Get end-effector coordinates
>>> get_eef()

# Change detection prompt
>>> set_prompt('red cup')

# Access controllers directly
>>> system.left_arm.move_to_pose([0.5, -0.3, 0.4], [0, 0, 0, 1])
>>> system.left_gripper.open()
>>> system.right_gripper.close()
```

### Teleoperation Controls

| Key | Action |
|-----|--------|
| Arrow keys | Move in X/Y plane |
| `.` / `;` | Move up/down (Z) |
| `o` / `p` | Rotate roll |
| `y` / `h` | Rotate pitch |
| `e` / `r` | Rotate yaw |
| `Space` | Toggle gripper |
| `s` | Switch active arm |
| `q` | Exit teleoperation |

### Demo Scripts

```bash
# Test arm position control
python scripts/demos/go_to_pose.py

# Test gripper open/close
python scripts/demos/gripper_move.py

# Standalone teleoperation
python scripts/teleop/teleop_dual_arm.py
```

## Microservices Architecture

All services communicate via ZMQ with JSON messages.

### Service Manager (port 5555)
Service discovery and registration.

### FastSAM Service (port 5551)
Continuous object detection with FastSAM.

**Commands**:
```python
# Set detection prompt
client.call('fastsam_service', {'command': 'SET_PROMPT', 'prompt': 'cup'})

# Get current prompt
client.call('fastsam_service', {'command': 'GET_PROMPT'})

# Get latest detections
client.call('fastsam_service', {'command': 'GET_LATEST_DETECTION'})
```

### Grasp Service (port 5552)
GraspGen-based grasp prediction.

**Commands**:
```python
client.call('grasp_service', {
    'command': 'GET_GRASPS',
    'pointcloud': detection['pointcloud'],
    'object_pose': detection['pose'],
    'num_grasps': 5,
    'gripper_type': 'finger'  # or 'suction'
})
```

### EEF Service (port 5559)
Real-time end-effector coordinates.

**Commands**:
```python
client.call('eef_service', {'command': 'GET'})
```

## Configuration

Key constants are defined in `src/robosuite_franka_manipulation/constants/__init__.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `DEVICE` | auto | 'cuda' if available, else 'cpu' |
| `FASTSAM_CONF_THRESHOLD` | 0.9 | Detection confidence threshold |
| `FASTSAM_IOU_THRESHOLD` | 0.4 | NMS IoU threshold |
| `DEFAULT_CAMERA_NAME` | 'agentview' | Camera for perception |
| `MAX_POINTCLOUD_RANGE` | 10.0 | Maximum depth for point clouds (m) |

## API Reference

### Controllers

```python
from robosuite_franka_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController

# Arm control
arm = RobosuiteArmController(env, arm_id="right", robot_id=0)
arm.move_to_pose(position=[x, y, z], orientation=[qx, qy, qz, qw])

# Gripper control
gripper = RobosuiteGripperController(env, arm_id="right", robot_id=0)
gripper.open()
gripper.close()
```

### Perception

```python
from robosuite_franka_manipulation.perception import ObjectDetector, GraspPredictor

# Object detection
detector = ObjectDetector(camera_intrinsics, camera_width, camera_height)
objects = detector.detect_objects(rgb_image, depth_image, camera_pos, camera_rot)

# Grasp prediction
predictor = GraspPredictor(gripper_type='franka_panda')
grasps = predictor.predict_grasps(object_pcd, object_pose, num_grasps=5)
```

### Services

```python
from robosuite_franka_manipulation.services import ServiceManager, ServiceClient, FastSAMService, GraspService, EEFService

# Start service manager
manager = ServiceManager()
manager.start()

# Use service client
client = ServiceClient()
response = client.call('service_name', {'command': 'CMD', ...})
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_perception.py -v
python -m pytest tests/test_services.py -v
python -m pytest tests/test_arm_controllers.py -v
```

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{dual_arm_franka_manipulation,
  title = {Dual-Arm Franka Manipulation},
  author = {Sukhvansh Jain},
  year = {2025},
  url = {https://github.com/Sukhvansh2004/Dual-Arm-Franka-Control-MTP}
}
```
