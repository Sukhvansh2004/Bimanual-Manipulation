# Dual-Arm Franka Manipulation

A comprehensive framework for dual-arm Franka robot manipulation with advanced perception, teleoperation, and autonomous control modes.

## Features

- **Dual-Arm Control**: Coordinated manipulation with Franka Panda robots
- **Advanced Perception**: FastSAM-based object detection with dynamic prompts
- **Grasp Generation**: GraspGen-powered 6-DOF grasp prediction (integrated)
- **Microservices**: Distributed perception services (Prompt, Grasp, EEF) with TCP/ZMQ interfaces
- **Teleoperation**: Keyboard-based manual control with robosuite integration
- **Mode Switching**: Seamless transition between autonomous and manual control
- **Standalone Operation**: Works out-of-the-box without external ROS dependencies
- **Comprehensive Testing**: Full test suite including service integration tests
- **Modular Architecture**: Clean separation of controllers, perception, and services

## Installation

### Quick Install

Run the automated installation script:

```bash
./install.sh
```

This will:
- Create the `dual_arm_manipulation` conda environment
- Install all dependencies including GraspGen
- Install the package in development mode
- **Note**: GraspGenModels (pre-trained models) need to be downloaded separately
- **Note**: FastSAM model is included in the `models/` folder
- **Note**: No ROS dependencies required - works standalone

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
export MUJOCO_GL=osmesa  # Set rendering backend
```

3. **Install PyTorch** (adjust for your CUDA version):
```bash
pip install torch==2.1.0 torchvision==0.16.0 torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

4. **Install GraspGen** (refer to [GraspGen README](deps/GraspGen/README.md)):
```bash
cd deps/GraspGen
pip install -e .
```

5. **Download GraspGen Models**:
```bash
```bash
# Clone the models repository (or use your own fork)
git clone https://huggingface.co/adithyamurali/GraspGenModels deps/GraspGenModels
```

5. **Install the Package**:
```bash
pip install -e .
```

## Codebase Structure

```
├── src/robosuite_franka_manipulation/     # Main package
│   ├── controllers/                        # Robot control interfaces
│   │   ├── arm.py                         # Arm movement and trajectory control
│   │   └── gripper.py                     # Gripper open/close control
│   ├── environments/                      # Robosuite environment configurations
│   │   └── handoff.py                     # Dual-arm handoff environment setup
│   ├── perception/                        # Computer vision and grasp prediction
│   │   ├── object_detector.py             # FastSAM-based object detection
│   │   └── grasp_predictor.py             # GraspGen grasp prediction
│   └── teleop/                           # Teleoperation interfaces
│       └── teleop.py                      # Keyboard teleoperation
├── scripts/                               # Executable scripts
│   ├── manipulation/                      # Autonomous manipulation tasks
│   │   ├── dual_arm_manipulation.py       # Full manipulation pipeline
│   │   └── handoff_sequence.py            # Handoff task implementation
│   ├── teleop/                           # Teleoperation scripts
│   │   └── teleop_dual_arm.py             # Keyboard teleoperation
│   └── demos/                            # Demonstration scripts
├── tests/                                # Test suite (pytest)
│   ├── test_environments.py              # Environment tests
│   ├── test_arm_controllers.py           # Controller tests
│   ├── test_perception.py                # Perception tests
│   └── test_trajectory_planning.py       # Trajectory planning tests
├── docs/                                # Documentation
│   └── perception_architecture.md       # Detailed perception system docs
├── models/                              # Pre-trained models
│   └── FastSAM-x.pt                      # FastSAM segmentation model
├── deps/                                # External dependencies
│   ├── GraspGen/                        # Grasp generation framework
│   └── GraspGenModels/                  # Pre-trained grasp models
└── pyproject.toml                       # Package configuration
```

## Usage

### Quick Start

1. **Install the package**:
```bash
./install.sh
```

2. **Run teleoperation** to familiarize yourself with the interface:
```bash
python scripts/teleop/teleop_dual_arm.py
```

3. **Run autonomous manipulation**:
```bash
python scripts/manipulation/dual_arm_manipulation.py
```

### Autonomous Manipulation

The system provides autonomous dual-arm manipulation with integrated perception:

```bash
# Run the full manipulation pipeline
python scripts/manipulation/dual_arm_manipulation.py --task handoff
```

**Available Tasks**:
- `handoff`: Pick object with left arm, transfer to right arm
- `cooperative_lift`: Both arms coordinate to lift objects

**Pipeline Flow**:
1. Move arms to ready positions
2. Detect objects using FastSAM segmentation
3. Generate grasp poses with GraspGen
4. Execute coordinated manipulation sequence

### Manual Teleoperation

For manual control and testing:

```bash
# Start keyboard teleoperation
python scripts/teleop/teleop_dual_arm.py
```

**Controls** (Robosuite standard):
- `w/s/a/d`: Move in X-Y plane
- `q/e`: Move vertically (Z)
- `i/k/j/l/u/o`: Rotate (roll/pitch/yaw)
- `space`: Toggle gripper
- `r/t`: Switch between arms
- `x`: Exit

### Perception System

The perception system runs automatically but can be tested independently:

```python
from robosuite_franka_manipulation.perception import ObjectDetector
from robosuite_franka_manipulation.environments import make_dual_arm_handoff_env

# Setup
env = make_dual_arm_handoff_env()
detector = ObjectDetector(env)

# Detect objects
objects = detector.detect_objects()

# Each object contains: pose, point_cloud, mask, grasps
for obj in objects:
    print(f"Found object with {len(obj['grasps'])} grasp predictions")
```

**Architecture Details**: See [docs/perception_architecture.md](docs/perception_architecture.md) for comprehensive perception system documentation.

### Debug & Visualization

Enable debug output and visualization for perception components:

```python
from robosuite_franka_manipulation.perception import ObjectDetector, GraspPredictor

# Enable debug output and visualization
detector = ObjectDetector(env, debug=True, visualize=True)
predictor = GraspPredictor(debug=True, visualize=True)

# This will show:
# - FastSAM segmentation masks overlaid on RGB images
# - 3D point clouds of detected objects  
# - Generated grasp poses with confidence scores
objects = detector.detect_objects()
```

**Visualization Features**:
- **FastSAM**: Mask overlays on camera images with object labels
- **Point Clouds**: 3D scatter plots of object geometry
- **Grasps**: 3D visualization of predicted grasp poses colored by confidence

**Requirements**: `matplotlib` and `opencv-python` for visualization.

### Development & Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_perception.py -v
python -m pytest tests/test_arm_controllers.py -v
python -m pytest tests/test_services.py -v
```

### API Usage

**Core Components**:

```python
# Environment
from robosuite_franka_manipulation.environments import make_dual_arm_handoff_env
env = make_dual_arm_handoff_env()

# Controllers
from robosuite_franka_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController
arm = RobosuiteArmController(env, arm_id="right", robot_id=0)
gripper = RobosuiteGripperController(env, arm_id="right", robot_id=0)

# Perception
from robosuite_franka_manipulation.perception import ObjectDetector, GraspPredictor
detector = ObjectDetector(env)
predictor = GraspPredictor()

# Teleoperation
from robosuite_franka_manipulation.teleop import KeyboardTeleop
teleop = KeyboardTeleop(env)
teleop.run()
```

## Usage

### Autonomous Manipulation

Run dual-arm manipulation tasks:

```bash
python -m robosuite_franka_manipulation.scripts.manipulation.dual_arm_manipulation --task handoff
```

Available tasks:
- `handoff`: Pick with left arm, hand off to right arm
- `cooperative_lift`: Both arms lift together

### Manual Teleoperation

Run teleoperation mode:

```bash
python -m robosuite_franka_manipulation.scripts.teleop.teleop_dual_arm
```

### Microservices

Start individual services for distributed perception:

```bash
# Start all services together
python scripts/start_services.py

# Or start individual services
python -c "from robosuite_franka_manipulation.services.prompt_service import PromptService; PromptService().run()"
python -c "from robosuite_franka_manipulation.services.grasp_service import GraspService; GraspService().run()"
python -c "from robosuite_franka_manipulation.services.eef_service import EEFService; from robosuite_franka_manipulation.environments import make_dual_arm_handoff_env; env = make_dual_arm_handoff_env(); EEFService(env).run()"
```

**Service APIs**:

**Prompt Service** (TCP):
```bash
# Get current prompts
echo 'GET' | nc localhost 5557

# Set new prompts
echo 'SET ["bottle", "cup", "box"]' | nc localhost 5557
```

**Grasp Service** (ZMQ):
```python
import zmq
# Client code for grasp prediction requests
```

**EEF Service** (TCP):
```bash
# Get left arm coordinates
echo 'GET left' | nc localhost 5559

# Get right arm coordinates  
echo 'GET right' | nc localhost 5559
```

### Prompt Service

Start the SAM prompt service (optional for dynamic prompts):

```bash
python -m robosuite_franka_manipulation.services.prompt_service
```

Set prompts:
```bash
echo "SET: red cube" | nc localhost 5557
```

Get current prompt:
```bash
echo "GET" | nc localhost 5557
```

**Note**: The system works with default prompt "object" without this service.

## Microservices Architecture

The system uses an **integrated perception pipeline** with optional microservices for advanced functionality. Currently, the core perception components (`ObjectDetector` and `GraspPredictor`) are implemented as integrated classes rather than separate services.

### Current Implementation Status

- **✅ Object Detection**: Integrated FastSAM-based detection
- **✅ Grasp Prediction**: Integrated GraspGen-based prediction
- **✅ Prompt Service**: TCP socket service on port 5557 for dynamic prompts
- **✅ Grasp Service**: ZMQ REP service on port 5558 for remote grasp prediction
- **✅ EEF Service**: TCP socket service on port 5559 for end-effector coordinates

### Perception Pipeline Flow

```
1. Teleoperation → Arm positioned for object view
2. ObjectDetector.detect_objects() → Gets camera images
3. ObjectDetector → Runs FastSAM segmentation on RGB image
4. ObjectDetector → Extracts point cloud from depth + mask
5. ObjectDetector → Runs GraspGen prediction locally
6. ObjectDetector → Returns objects with grasps
```

### Available Services

The system provides three microservices for distributed perception:

#### Prompt Service (`localhost:5557`)
- **Purpose**: Manages text prompts for FastSAM object detection
- **Protocol**: TCP socket with JSON commands (GET/SET)
- **Status**: ✅ Fully implemented and tested
- **Usage**: Dynamic prompt management for object detection

#### Grasp Prediction Service (`localhost:5558`)
- **Purpose**: Provides remote grasp prediction using GraspGen
- **Protocol**: ZMQ REQ/REP with JSON messages
- **Status**: ✅ Fully implemented and tested
- **Usage**: Distributed grasp prediction with numpy array serialization

#### EEF Service (`localhost:5559`)
- **Purpose**: Real-time end-effector coordinates in world frame
- **Protocol**: TCP socket with JSON responses
- **Status**: ✅ Fully implemented and tested
- **Usage**: Coordinate retrieval for both left and right arms

### Default Behavior

The system works out-of-the-box with:
- **Default prompt**: "object" (no external service needed)
- **Local grasp prediction**: Integrated GraspGen inference
- **Fallback handling**: Graceful degradation when services unavailable

## Controls

### Manipulation Mode Switching
- `m`: Switch to manual teleoperation
- `a`: Switch back to autonomous mode

### Teleoperation Controls
- **Movement**: `w/s` (X), `a/d` (Y), `q/e` (Z)
- **Rotation**: `i/k` (roll), `j/l` (pitch), `u/o` (yaw)
- **Gripper**: `space` (close)
- **Arm Switch**: `r` (right), `t` (left)
- **Exit**: `x` or `a`

## Project Structure

```
├── src/
│   └── robosuite_franka_manipulation/     # Main package
│       ├── controllers/                   # Arm and gripper controllers
│       ├── environments/                  # Robosuite environments
│       ├── perception/                    # Object detection and grasp prediction
│       ├── services/                      # Microservices
│       └── teleop/                        # Teleoperation interfaces
├── scripts/                               # Standalone executable scripts
│   ├── demos/                            # Demonstration scripts
│   ├── manipulation/                     # Manipulation task scripts
│   └── teleop/                           # Teleoperation scripts
├── tests/                                # Test suite (pytest)
│   ├── test_environments.py              # Environment tests
│   ├── test_arm_controllers.py           # Controller tests
│   ├── test_perception.py                # Perception tests
│   ├── test_services.py                  # Service functionality tests
│   └── test_trajectory_planning.py       # Trajectory planning tests
├── docs/                                # Documentation
│   └── perception_architecture.md       # Perception system architecture
├── scratch/                             # Development helpers and debug scripts
├── models/                              # Pre-trained models (FastSAM, etc.)
├── deps/                                # Dependencies (GraspGen, GraspGenModels)
└── pyproject.toml                       # Package configuration
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dual_arm_franka_manipulation,
  title = {Dual-Arm Franka Manipulation},
  author = {Sukhvansh Jain},
  year = {2025},
  url = {https://github.com/Sukhvansh2004/Dual-Arm-Franka-Control-MTP}
}
```