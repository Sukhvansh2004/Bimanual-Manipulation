# Dual-Arm Franka Manipulation

A comprehensive framework for dual-arm Franka robot manipulation with advanced perception, teleoperation, and autonomous control modes.

## Features

- **Dual-Arm Control**: Coordinated manipulation with Franka Panda robots
- **Advanced Perception**: FastSAM-based object detection with dynamic prompts
- **Grasp Generation**: GraspGen-powered 6-DOF grasp prediction
- **Teleoperation**: Keyboard-based manual control with robosuite integration
- **Mode Switching**: Seamless transition between autonomous and manual control
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

### Manual Installation

1. **Create Conda Environment**:
```bash
conda create -n dual_arm_manipulation python=3.10
conda activate dual_arm_manipulation
```

2. **Install PyTorch** (adjust for your CUDA version):
```bash
pip install torch==2.1.0 torchvision==0.16.0 torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

3. **Install GraspGen** (refer to [GraspGen README](deps/GraspGen/README.md)):
```bash
cd deps/GraspGen
pip install -e .
```

4. **Install the Package**:
```bash
pip install -e .
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

Start the SAM prompt service:

```bash
python -m robosuite_franka_manipulation.services.prompt_service
```

Set prompts:
```bash
echo "SET: red cube" | nc localhost 5557
```

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
│   └── robosuite_franka_manipulation/
│       ├── controllers/          # Robot and gripper controllers
│       ├── environments/         # Robosuite environment configurations
│       ├── perception/           # Object detection and grasp prediction
│       ├── teleop/               # Teleoperation modules
│       ├── services/             # Microservices (prompt service)
│       └── scripts/              # Executable scripts
├── deps/                         # Dependencies
│   ├── GraspGen/                 # Grasp generation framework
│   └── GraspGenModels/           # Pre-trained models
├── pyproject.toml                # Package configuration
├── install.sh                    # Automated installation
└── README.md
```

## Dependencies

- **robosuite**: Simulation framework
- **GraspGen**: Grasp generation
- **FastSAM** (via ultralytics): Object segmentation
- **numpy, scipy**: Numerical computing
- **keyboard**: Input handling
- **torch**: Deep learning

## Development

### Running Tests

```bash
python -m pytest
```

### Building Documentation

```bash
sphinx-build docs/ docs/_build/
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dual_arm_franka_manipulation,
  title = {Dual-Arm Franka Manipulation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/dual-arm-franka-control}
}
```