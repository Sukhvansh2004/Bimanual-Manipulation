# Dual-Arm Franka Manipulation

A comprehensive framework for dual-arm Franka robot manipulation with advanced perception, trajectory planning, teleoperation, and autonomous control modes.

## Features

- **Dual-Arm Control**: Coordinated manipulation with two Franka Panda robots (parallel-jaw + suction grippers)
- **Advanced Perception**: FastSAM-based object segmentation with text prompts (GPU-accelerated)
- **Grasp Generation**: GraspGen-powered 6-DOF grasp prediction with collision filtering
- **RRT-Connect Trajectory Planning**: Collision-aware object trajectory planning in SE(3) with MuJoCo-based collision checking
- **Bézier Curve Smoothing**: Cubic Bézier spline smoothing for positions (R³) and rotations (SO(3) via De Casteljau SLERP)
- **SE(3) Lie Algebra Utilities**: Proper coupled rotation-translation interpolation on the SE(3) manifold
- **Experiment Configuration System**: JSON-driven experiment specification with 12 built-in presets
- **Transforms Module**: ROS tf/tf2-like coordinate frame management
- **Motion Primitives**: High-level manipulation primitives (handover, grasp, release-retreat)
- **Microservices Architecture**: Distributed services (FastSAM, Grasp, MoveToPose) with ZMQ interfaces
- **Meshcat Visualization**: 3D visualization of RRT trees, Bézier curves, robot meshes, and transforms
- **Teleoperation**: Keyboard-based manual control with robosuite integration
- **Interactive Shell**: Python REPL for real-time service queries and control
- **YCB Object Support**: Load objects from the YCB dataset into simulation environments
- **Comprehensive Test Suite**: 80 pytest tests covering all modules
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
export MUJOCO_GL="glx"
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
├── src/robosuite_franka_manipulation/       # Main package
│   ├── constants.py                         # All configuration constants
│   ├── experiment_config.py                 # JSON experiment config loader & dataclasses
│   ├── schema.py                            # Pydantic models for service APIs
│   ├── system_manager.py                    # DualArmManipulationSystem orchestrator
│   ├── controllers/                         # Arm and gripper controllers
│   │   ├── arm.py                           # SE(3) trajectory tracking with Lie algebra
│   │   └── gripper.py                       # Gripper open/close control
│   ├── environments/                        # Robosuite environment factory (YCB support)
│   ├── perception/                          # Object detection and grasp prediction
│   │   ├── object_detector.py               # FastSAM-based detection
│   │   └── grasp_predictor.py               # GraspGen wrapper
│   ├── planning/                            # Trajectory planning and validation
│   │   ├── trajectory_generator.py          # SE(3) trajectory generation & IK validation
│   │   ├── collision_checker.py             # Single/dual-arm grasp feasibility
│   │   ├── grasp_generator.py               # High-level grasp sampling API
│   │   ├── object_trajectory_planner.py     # RRT-Connect in SE(3) + Bézier smoothing
│   │   └── utils/                           # Planning sub-utilities
│   │       ├── collision_utils.py           # MuJoCo collision helpers
│   │       ├── ik_utils.py                  # Scipy-based IK solver
│   │       ├── mujoco_utils.py              # Native MuJoCo object access
│   │       ├── trajectory_utils.py          # Waypoint feasibility data structures
│   │       ├── visualization_utils.py       # Meshcat planning visualization
│   │       ├── robot_mesh_visualizer.py     # Robot mesh rendering in Meshcat
│   │       └── object_mesh_visualizer.py    # Object mesh rendering in Meshcat
│   ├── services/                            # Microservices
│   │   ├── services_core.py                 # ServiceManager & ServiceClient (port 5555)
│   │   ├── fastsam_service.py               # FastSAM detection (port 5551)
│   │   ├── grasp_service.py                 # GraspGen prediction (port 5552)
│   │   └── move_to_pose_service.py          # MoveToPose / trajectory following (port 5553)
│   ├── transforms/                          # Coordinate frame management (tf-like)
│   │   ├── stamped_pose.py                  # StampedPose and Transform classes
│   │   ├── transform_manager.py             # Frame tree and transform lookup
│   │   └── meshcat_tf_visualizer.py         # Live frame visualization in Meshcat
│   ├── utils/                               # Shared utilities
│   │   ├── se3.py                           # SE(3) / so(3) Lie algebra utilities
│   │   ├── pointcloud.py                    # Point cloud processing & deprojection
│   │   ├── env_utils.py                     # Environment helpers (YCB, primitives, stepping)
│   │   ├── mesh_utils.py                    # Mesh loading and sampling
│   │   ├── meshcat_mj_utils.py              # Meshcat ↔ MuJoCo bridge
│   │   ├── primitives.py                    # DualArmHandoverPlanner, grasp/release primitives
│   │   └── suction_weld.py                  # Suction gripper weld constraints
│   ├── models/grippers/                     # Custom gripper models (suction)
│   ├── assets/                              # Gripper XMLs and mesh files
│   └── teleop/                              # Keyboard teleoperation
├── scripts/
│   ├── run_all_experiments.py               # Batch experiment runner (--no-meshcat)
│   ├── planning/                            # Trajectory planning scripts
│   │   ├── trajectory_generation.py         # Single-mode trajectory generation
│   │   ├── dual_trajectory_generation.py    # Dual-arm trajectory generation
│   │   └── visualize_robot_trajectory.py    # Trajectory playback & visualization
│   ├── demos/                               # Demo scripts
│   │   ├── go_to_pose.py                    # Arm position control demo
│   │   ├── gripper_move.py                  # Gripper open/close demo
│   │   ├── launch_env.py                    # Environment launcher (with YCB objects)
│   │   └── visualize_rrt.py                 # RRT planner visualization demo
│   ├── manipulation/                        # Manipulation execution scripts
│   │   ├── dual_arm_manipulation.py         # Full system: services + interactive shell
│   │   └── dual_arm_handover_traj.py        # Dual-arm handover with pre-planned trajectories
│   ├── teleop/                              # Standalone teleoperation
│   │   └── teleop_dual_arm.py
│   └── transforms/
│       └── transform_visualizer_meshcat.py  # Meshcat transform visualizer
├── experiments/                             # 12 JSON experiment configurations
├── tests/                                   # Pytest test suite (80 tests)
├── data/                                    # Pre-recorded trajectory data (NPZ files)
├── models/                                  # Pre-trained models (FastSAM-x.pt)
└── deps/                                    # Dependencies (GraspGen, GraspGenModels, YCB_Dataset)
```

## Usage

### Experiment-Based Trajectory Planning

The primary workflow uses JSON experiment configs to define manipulation tasks:

```bash
# Run a single experiment
python scripts/planning/trajectory_generation.py \
    --experiment experiments/flatPlate_table_to_shelf.json

# Run dual-arm trajectory planning
python scripts/planning/dual_trajectory_generation.py \
    --experiment experiments/ycb_mug_table_to_rack.json

# Run all experiments in batch (meshcat disabled for speed)
python scripts/run_all_experiments.py
```

Each experiment JSON specifies the environment setup, object to manipulate, target pose, planning parameters, visualization settings, and more. See [Experiment Configuration](#experiment-configuration) below.

### Visualize RRT Planning

```bash
# Watch the RRT-Connect planner in Meshcat (trees, Bézier curves, collision checks)
python scripts/demos/visualize_rrt.py --experiment experiments/flatPlate_table_to_shelf.json
```

### Main Script: Dual-Arm Manipulation System

The interactive entry point starts all services and provides a shell:

```bash
python scripts/manipulation/dual_arm_manipulation.py --visualize
```

**Arguments**:
- `--visualize`: Enable OpenCV visualization of FastSAM detections
- `--debug`: Enable debug logging
- `--prompt <text>`: Set initial detection prompt (default: "object")
- `--meshcat-tf`: Enable Meshcat transform visualization

This starts:
1. Dual-arm robosuite environment with visualization
2. Service Manager for service discovery
3. FastSAM Service for object detection
4. Grasp Service for grasp prediction
5. MoveToPose Service for autonomous arm control
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

# Move an arm to a pose
>>> go_to_pose(pose, robot_id=0, arm_id='right')

# Cancel a move
>>> cancel_move(robot_id=0, arm_id='right')

# Access controllers directly
>>> system.left_arm.move_to_pose([0.5, -0.3, 0.4], [0, 0, 0, 1])
>>> system.left_gripper.open()
>>> system.right_gripper.close()
```

### Execute Pre-Planned Trajectories

```bash
# Execute a dual-arm handover using saved trajectory files
python scripts/manipulation/dual_arm_handover_traj.py \
    --experiment experiments/ycb_mug_rack_to_table.json
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

# Launch environment with custom/YCB objects
python scripts/demos/launch_env.py --with-objects --ycb apple banana

# Standalone teleoperation
python scripts/teleop/teleop_dual_arm.py

# Visualize Meshcat transforms
python scripts/transforms/transform_visualizer_meshcat.py
```

## Experiment Configuration

Experiments are defined as JSON files in the `experiments/` directory. Each config specifies:

```jsonc
{
  "name": "flatPlate_table_to_shelf",
  "description": "Move a flat plate from the table into a shelf slot.",
  "environment": {
    "table_full_size": [1.0, 0.75, 0.05],
    "ycb_objects": [{"ycb_name": "table_shelf", "position": [...], "is_static": true}],
    "custom_objects": [{"type": "box", "name": "flatPlate", "size": [...], ...}]
  },
  "task": {
    "manipulated_object": "flatPlate",
    "start_pose": null,            // null = auto-detect from simulation
    "target_pose": {"position": [...], "quaternion": [...]}
  },
  "planning": {
    "num_grasps": 200,
    "gripper_types": ["finger", "suction"],
    "num_steps": 50,
    "lift_height": 0.2,
    "approach_offset": 0.3
  },
  "collision_filter": {
    "exclude_patterns": ["flatPlate", "table_shelf"]
  },
  "visualization": {
    "visualize_mesh": true,
    "mesh_step_interval": 10,
    "zmq_url": "tcp://127.0.0.1:6000",
    "enable_meshcat": true
  },
  "output": {
    "save_trajectories": false,
    "output_dir": null
  },
  "execution": {
    "visualize": true,
    "meshcat_tf": true,
    "smoothen": false,
    "traj0": null,
    "traj1": null
  },
  "runtime": {
    "debug": false,
    "num_workers": 4
  }
}
```

The config is loaded via `load_experiment()` into typed dataclasses (`ExperimentConfig`, `TaskConfig`, `PlanningConfig`, `VisualizationConfig`, `ExecutionConfig`, etc.).

**Available experiments**: `flatPlate_shelf_to_box`, `flatPlate_shelf_to_table`, `flatPlate_table_to_shelf`, `ycb_bowl_rack_to_table`, `ycb_bowl_table_to_rack`, `ycb_can_table_to_rack`, `ycb_mug_rack_to_table`, `ycb_mug_table_to_rack`, `ycb_plate_shelf_to_table`, `ycb_plate_table_to_shelf`, `dual_handover_mug`, `dual_handover_plate`.

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
    'pcd': detection['pcd'],
    'object_pose': detection['pose'],
    'num_grasps': 5,
    'gripper_type': ['finger', 'suction']
})
```

### MoveToPose Service (port 5553)
Autonomous robot arm control.

**Commands**:
```python
# Move to pose
client.call('move_to_pose_service', {
    'command': 'MOVE_TO_POSE',
    'robot_id': 0,
    'arm': 'right',
    'pose': pose_dict,
    'gripper_action': 0.0,
    'approach': True
})

# Follow pre-planned trajectory
client.call('move_to_pose_service', {
    'command': 'FOLLOW_TRAJECTORY',
    'robot_id': 0,
    'arm': 'right',
    'trajectory': [{'pos': [...], 'quat': [...], 'frame_id': 'world'}, ...],
    'gripper_action': 0.0
})

# Cancel move
client.call('move_to_pose_service', {
    'command': 'CANCEL_MOVE',
    'robot_id': 0,
    'arm': 'right'
})
```

## Configuration

Key constants are defined in `src/robosuite_franka_manipulation/constants.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| **Perception** | | |
| `DEVICE` | auto | `'cuda'` if available, else `'cpu'` |
| `FASTSAM_CONF_THRESHOLD` | 0.9 | Detection confidence threshold |
| `FASTSAM_IOU_THRESHOLD` | 0.4 | NMS IoU threshold |
| `DEFAULT_CAMERA_NAME` | `'agentview'` | Camera for perception |
| `MAX_POINTCLOUD_RANGE` | 10.0 | Maximum depth for point clouds (m) |
| **RRT Planning** | | |
| `RRT_MAX_ITERATIONS` | 5000 | Maximum RRT-Connect iterations |
| `RRT_STEP_SIZE` | 0.05 | Extension step size (SE(3) distance) |
| `RRT_GOAL_BIAS` | 0.15 | Probability of sampling the goal |
| `RRT_ROTATION_WEIGHT` | 0.1 | Rotation distance weight |
| `RRT_SMOOTHING_ITERATIONS` | 200 | Shortcutting attempts for smoothing |
| `INTERPOLATION_TYPE` | `'linear'` | `'linear'` (lerp+slerp) or `'SE3'` (geodesic) |
| **Meshcat** | | |
| `MESHCAT_ZMQ_URL` | `tcp://127.0.0.1:6000` | Meshcat visualizer address |
| `MESHCAT_TF_VISUALIZER_FPS` | 20.0 | Transform visualizer refresh rate |

## API Reference

### Controllers

```python
from robosuite_franka_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController

# Arm control with SE(3) trajectory planning
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

### Planning

```python
from robosuite_franka_manipulation.planning import (
    plan_object_trajectory,           # RRT-Connect in SE(3)
    generate_object_trajectory,       # SE(3) interpolation (lift-approach-insert)
    generate_end_effector_trajectory, # Object → EEF trajectory via grasp transform
    check_trajectory_feasibility,     # IK + collision validation
    check_dual_trajectory_feasibility,
)

# Plan collision-free object path (RRT-Connect + Bézier smoothing)
trajectory, success = plan_object_trajectory(
    env, object_name="plate",
    start_pose=start, target_pose=goal,
    tf_manager=tf_manager,
)
```

### Transforms (tf-like API)

```python
from robosuite_franka_manipulation.transforms import TransformManager, StampedPose

# Create transform manager with environment
tf_manager = TransformManager(env)

# Create a pose in camera frame
camera_pose = StampedPose(
    position=[0.5, 0.0, 0.3],
    quaternion=[0, 0, 0, 1],  # (x, y, z, w)
    frame_id="agentview"
)

# Transform to world frame
world_pose = tf_manager.transform_pose(camera_pose, "world")

# Get current EEF pose
eef_pose = tf_manager.get_frame_pose("robot0_right_eef", "world")
```

### SE(3) Utilities

```python
from robosuite_franka_manipulation.utils.se3 import (
    pose_to_SE3, SE3_to_pose, SE3_log, SE3_exp,
    interpolate_SE3, interpolate_pose, compute_se3_distance,
)

# Geodesic interpolation on SE(3) manifold
mid_pos, mid_quat = interpolate_pose(start_pos, start_quat, end_pos, end_quat, t=0.5)

# Direct SE(3) matrix operations
T = pose_to_SE3(position, quaternion)
twist = SE3_log(T)            # 6D twist [omega, v]
T_back = SE3_exp(twist)
pos, quat = SE3_to_pose(T)
```

### Services

```python
from robosuite_franka_manipulation.services import (
    ServiceManager, ServiceClient, FastSAMService, GraspService, MoveToPoseService
)

# Start service manager
manager = ServiceManager()
manager.start()

# Use service client
client = ServiceClient()
response = client.call('service_name', {'command': 'CMD', ...})
```

## Testing

```bash
# Run all tests (80 tests)
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_arm_controllers.py -v      # 7 tests
python -m pytest tests/test_environments.py -v         # 3 tests
python -m pytest tests/test_fastsam_detection.py -v    # 3 tests
python -m pytest tests/test_grasps.py -v               # 24 tests
python -m pytest tests/test_move_to_pose_service.py -v # 2 tests
python -m pytest tests/test_planning_module.py -v      # 6 tests
python -m pytest tests/test_services.py -v             # 3 tests
python -m pytest tests/test_trajectory_planning.py -v  # 15 tests
python -m pytest tests/test_transforms.py -v           # 17 tests
```

## Documentation

- [Planning Architecture](docs/planning.md) - RRT-Connect, Bézier smoothing, trajectory generation
- [Microservices Architecture](docs/services.md) - Service APIs and usage
- [Perception Architecture](docs/perception_architecture.md) - Detection and grasp prediction
- [Transforms Module](docs/transforms.md) - Coordinate frame management
- [Utilities Module](docs/utilities.md) - SE(3), point cloud, and environment utilities

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{dual_arm_franka_manipulation,
  title = {Dual-Arm Franka Manipulation},
  author = {Sukhvansh Jain},
  year = {2026},
  url = {https://github.com/Sukhvansh2004/Dual-Arm-Franka-Control-MTP}
}
```
