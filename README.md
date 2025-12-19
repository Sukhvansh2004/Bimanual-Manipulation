# Dual-Arm Franka Manipulation

A comprehensive framework for dual-arm Franka robot manipulation with advanced perception, teleoperation, and autonomous control modes.

## Features

- **Dual-Arm Control**: Coordinated manipulation with Franka Panda robots
- **Advanced Perception**: FastSAM-based object detection with dynamic prompts
- **Grasp Generation**: GraspGen-powered 6-DOF grasp prediction (integrated)
- **Teleoperation**: Keyboard-based manual control with robosuite integration
- **Mode Switching**: Seamless transition between autonomous and manual control
- **Standalone Operation**: Works out-of-the-box without external ROS dependencies
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

### Development & Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_perception.py -v
python -m pytest tests/test_arm_controllers.py -v
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

The system uses an integrated perception pipeline with optional microservices for advanced functionality:

### Perception Pipeline Flow

```
1. Teleoperation → Arm positioned for object view
2. ObjectDetector.detect_objects() → Gets camera images
3. ObjectDetector → Runs FastSAM segmentation on RGB image
4. ObjectDetector → Extracts point cloud from depth + mask
5. ObjectDetector → Runs GraspGen prediction locally
6. ObjectDetector → Returns objects with grasps
```

### Optional Services (Development Mode)

For development and testing, optional microservices can be started:

#### Prompt Service (`localhost:5557`)
- **Purpose**: Manages text prompts for FastSAM object detection
- **Protocol**: TCP socket with simple text commands
- **Status**: Currently in development (`scratch/test_prompt_service.py`)

#### Grasp Prediction Service (`localhost:5558`)
- **Purpose**: Provides grasp prediction using GraspGen
- **Protocol**: ZMQ REQ/REP with JSON messages
- **Status**: Currently integrated directly into ObjectDetector

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
│       └── teleop/                        # Teleoperation interfaces
├── scripts/                               # Standalone executable scripts
│   ├── demos/                            # Demonstration scripts
│   ├── manipulation/                     # Manipulation task scripts
│   └── teleop/                           # Teleoperation scripts
├── tests/                                # Test suite (pytest)
│   ├── test_environments.py              # Environment tests
│   ├── test_arm_controllers.py           # Controller tests
│   ├── test_perception.py                # Perception tests
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