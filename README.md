# Dual-Arm Franka Manipulation

A comprehensive framework for dual-arm Franka robot manipulation with advanced perception, trajectory planning, teleoperation, and autonomous control modes.

## Features

- **Dual-Arm Control**: Coordinated manipulation with two Franka Panda robots (parallel-jaw + suction grippers)
- **Advanced Perception**: FastSAM-based object segmentation with text prompts (GPU-accelerated)
- **Grasp Generation**: GraspGen-powered 6-DOF grasp prediction with collision filtering
- **RRT-Connect Trajectory Planning**: Collision-aware object trajectory planning in SE(3) with MuJoCo-based collision checking
- **Bézier Curve Smoothing**: Cubic Bézier spline smoothing for positions (R³) and rotations (SO(3) via De Casteljau SLERP)
- **SE(3) Lie Algebra Utilities**: Proper coupled rotation-translation interpolation on the SE(3) manifold
- **Experiment Configuration System**: JSON-driven experiment specification with 4 main presets (+ 9 extras)
- **Transforms Module**: ROS tf/tf2-like coordinate frame management
- **Motion Primitives**: High-level manipulation primitives (handover, grasp, release-retreat)
- **Microservices Architecture**: Distributed services (FastSAM, Grasp, MoveToPose) with ZMQ interfaces
- **Meshcat Visualization**: 3D visualization of RRT trees, Bézier curves, robot meshes, and transforms
- **Teleoperation**: Keyboard-based manual control with robosuite integration
- **Interactive Shell**: Python REPL for real-time service queries and control
- **YCB Object Support**: Load objects from the YCB dataset into simulation environments
- **Comprehensive Test Suite**: Extensive pytest suite covering all modules
- **Standalone Operation**: Works out-of-the-box without external ROS dependencies

## Installation

### Prerequisites

- **conda** (Miniconda or Anaconda)
- **git** with LFS support (for HuggingFace model clone)
- **NVIDIA GPU** with a compatible driver (CUDA 12.1 is installed inside conda — no system CUDA required)

> No `sudo` is needed. All system-level libraries (Mesa, GLEW, cmake, Eigen, CUDA toolkit) are installed inside the conda environment.

### Quick Install

```bash
git clone --recursive git@github.com:Sukhvansh2004/Bimanual-Manipulation.git
cd Bimanual-Manipulation
bash install.sh
```

This will:
1. Initialize git submodules (GraspGen, YCB_Dataset)
2. Download GraspGen pre-trained models from HuggingFace (~3.9 GB)
3. Create the `dual_arm_manipulation` conda environment (Python 3.10)
4. Install CUDA 12.1 toolkit, Mesa, GLEW, cmake, and Eigen via conda (no sudo)
5. Install PyTorch 2.1.0 + CUDA 12.1, torch-scatter, torch-cluster
6. Install MuJoCo with EGL rendering (GPU-accelerated, headless-safe)
7. Build GraspGen + PointNet2 CUDA ops from source
8. Install the main package in editable mode (`pip install -e .`)
9. Verify all imports

### Manual Installation

If you prefer step-by-step control:

1. **Clone and initialize submodules**:
```bash
git clone --recursive https://github.com/Sukhvansh2004/Bimanual-Manipulation.git
cd Bimanual-Manipulation
git submodule update --init deps/GraspGen deps/YCB_Dataset
git clone https://huggingface.co/adithyamurali/GraspGenModels deps/GraspGenModels
```

2. **Create and activate conda environment**:
```bash
conda create -n dual_arm_manipulation python=3.10 -y
conda activate dual_arm_manipulation
```

3. **Install system libraries via conda** (no sudo):
```bash
conda install -y -c nvidia/label/cuda-12.1.0 cuda-toolkit
conda install -y -c conda-forge mesalib glew cmake eigen
```

4. **Set persistent environment variables**:
```bash
conda env config vars set MUJOCO_GL=egl -n dual_arm_manipulation
conda env config vars set CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/include/eigen3"
conda env config vars set C_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/include/eigen3"
conda deactivate && conda activate dual_arm_manipulation
```

5. **Install PyTorch + CUDA extensions**:
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install "numpy>=1.21.0,<2.0.0"
```

6. **Install MuJoCo**:
```bash
pip install mujoco
```

7. **Install GraspGen + PointNet2**:
```bash
cd deps/GraspGen
pip install -e .
cd pointnet2_ops && pip install --no-build-isolation . && cd ..
pip install PyOpenGL==3.1.5
cd ../..
```

8. **Install the main package**:
```bash
pip install -e .
pip install pyrender scene-synthesizer
```

9. **Verify**:
```bash
python -c "import bimanual_manipulation; print('OK')"
```

### Rendering Modes

The installer sets `MUJOCO_GL=egl` by default, which provides **GPU-accelerated offscreen rendering without an X11 display** — this works on both local and remote/headless systems.

| Mode | Env Var | Requires | Use Case |
|------|---------|----------|----------|
| **EGL** (default) | `MUJOCO_GL=egl` | NVIDIA GPU + driver | Headless servers, SSH, containers |
| **GLX** | `MUJOCO_GL=glx` | X11 display | Local desktop with monitor |
| **OSMesa** | `MUJOCO_GL=osmesa` | `libosmesa6` | CPU-only rendering (slow) |

To change: `conda env config vars set MUJOCO_GL=glx -n dual_arm_manipulation`

## Project Structure

```
├── src/bimanual_manipulation/       # Main package
│   ├── constants.py                         # All configuration constants
│   ├── experiment_config.py                 # JSON experiment config loader & dataclasses
│   ├── schema.py                            # Pydantic models for service APIs
│   ├── system_manager.py                    # DualArmManipulationSystem orchestrator
│   ├── controllers/                         # Arm and gripper controllers
│   │   ├── arm.py                           # SE(3) trajectory tracking with Lie algebra
│   │   ├── gripper.py                       # Gripper open/close control
│   │   └── trajectory_executor.py           # Joint trajectory execution
│   ├── environments/                        # Robosuite environment factory (YCB support)
│   ├── perception/                          # Object detection and grasp prediction
│   │   ├── object_detector.py               # FastSAM-based detection
│   │   └── grasp_predictor.py               # GraspGen wrapper
│   ├── planning/                            # Trajectory planning and validation
│   │   ├── collision_checker.py             # Single/dual-arm feasibility + waypoint processing
│   │   ├── assignment/                      # Segment assignment planner (Viterbi DP)
│   │   │   ├── models.py                    # ObjectTrajectory, RobotTrajectory, TrajectoryBundle
│   │   │   ├── cost.py                      # Segment cost, cost matrix, switching cost
│   │   │   ├── solver.py                    # Viterbi DP solver
│   │   │   ├── result.py                    # AssignmentResult dataclass + NPZ I/O
│   │   │   ├── trajectory_io.py             # Load trajectories from disk → TrajectoryBundle
│   │   │   ├── factories.py                 # IK / Jacobian / collision-check closures
│   │   │   └── utils.py                     # Normalization, smoothness, manipulability helpers
│   │   ├── dual_arm/                        # Reactive dual-arm parallel execution (mink QP)
│   │   ├── ik/                              # Inverse kinematics subpackage
│   │   │   ├── ik_utils.py                  # Backend-agnostic IK dispatch & orientation helpers
│   │   │   ├── eaik_solver.py               # Analytical IK via EAIK (default)
│   │   │   ├── scipy_solver.py              # Numerical IK via L-BFGS-B
│   │   │   ├── mink_solver.py               # QP-based iterative IK via mink
│   │   │   └── manipulability.py            # Yoshikawa manipulability measures
│   │   ├── rrt/                             # RRT planners
│   │   │   ├── core.py                      # Generic bi-directional RRT-Connect
│   │   │   ├── se3/                         # SE(3) object trajectory planning
│   │   │   └── joint/                       # Joint-space arm trajectory planning
│   │   ├── trajectory/                      # High-level trajectory planners & I/O
│   │   │   ├── object_planner.py            # RRT-Connect in SE(3) + Bézier smoothing
│   │   │   ├── arm_planner.py               # Joint-space RRT + smoothing pipeline
│   │   │   ├── generator.py                 # Object/EEF trajectory generation
│   │   │   └── utils.py                     # Trajectory save/load (NPZ) & WaypointFeasibilityResult
│   │   └── visualization/                   # Meshcat + matplotlib visualization
│   ├── services/                            # Microservices (ZMQ)
│   │   ├── services_core.py                 # ServiceManager & ServiceClient (port 6006)
│   │   ├── fastsam_service.py               # FastSAM detection (port 5551)
│   │   ├── grasp_service.py                 # GraspGen prediction (port 5552)
│   │   ├── grasp_generator.py               # High-level grasp sampling API
│   │   ├── move_to_pose_service.py          # MoveToPose / trajectory following (port 5553)
│   │   └── service_helpers.py               # Convenience wrappers for interactive use
│   ├── transforms/                          # Coordinate frame management (tf-like)
│   │   ├── stamped_pose.py                  # StampedPose and Transform classes
│   │   ├── transform_manager.py             # Frame tree and transform lookup
│   │   └── meshcat_tf_visualizer.py         # Live frame visualization in Meshcat
│   ├── utils/                               # Shared utilities
│   │   ├── se3.py                           # SE(3) / so(3) Lie algebra utilities
│   │   ├── pointcloud.py                    # Point cloud processing & deprojection
│   │   ├── env_utils.py                     # Environment stepping & action persistence
│   │   ├── object_factory.py                # YCB / primitive object creation & mesh extraction
│   │   ├── monitors.py                      # CollisionMonitor & ObjectTrajectoryMonitor
│   │   ├── mujoco_collision.py              # MuJoCo collision detection & geom helpers
│   │   ├── mujoco_utils.py                  # MuJoCo model/data accessors & Jacobian computation
│   │   ├── mesh_utils.py                    # Mesh loading and sampling
│   │   ├── meshcat_mj_utils.py              # Meshcat ↔ MuJoCo bridge
│   │   ├── primitives.py                    # DualArmHandoverPlanner, grasp/release primitives
│   │   ├── trajectory_math.py               # Quintic profiling, SE(3) trajectory smoothing
│   │   ├── trajectory_visualizer.py         # Meshcat trajectory visualization
│   │   └── suction_weld.py                  # Suction gripper weld constraints
│   ├── models/grippers/                     # Custom gripper models (suction)
│   ├── assets/                              # Gripper XMLs and mesh files
│   └── teleop/                              # Keyboard teleoperation
├── scripts/
│   ├── pipeline/                            # Core planning pipeline scripts
│   │   ├── trajectory_generation.py         # Single-mode trajectory generation
│   │   ├── dual_trajectory_generation.py    # Dual-arm trajectory generation
│   │   ├── object_trajectory_generation.py  # Object-only trajectory planning
│   │   ├── segment_assignment.py            # Optimal robot-to-segment assignment (Viterbi DP)
│   │   ├── dual_arm_handover_traj.py        # Dual-arm handover with pre-planned trajectories
│   │   └── find_feasible_pair.py            # Find feasible dual-arm grasp pairs
│   ├── utils/                               # Utility & visualization scripts
│   │   ├── dual_arm_manipulation.py         # Full system: services + interactive shell
│   │   ├── launch_env.py                    # Environment launcher (with YCB objects)
│   │   ├── visualize_rrt.py                 # RRT planner visualization demo
│   │   ├── visualize_robot_trajectory.py    # Trajectory playback & visualization
│   │   └── transform_visualizer_meshcat.py  # Meshcat transform visualizer
│   ├── demos/                               # Interactive demo scripts
│   │   ├── go_to_pose.py                    # Arm position control demo
│   │   ├── gripper_move.py                  # Gripper open/close demo
│   │   └── teleop_dual_arm.py               # Keyboard teleoperation
│   └── evaluation/                          # Evaluation & benchmarking
│       ├── run_all.py                       # Run all evaluations
│       ├── pipeline_iterative.py            # Multi-iteration robustness runner
│       ├── analyze.py                       # Result analysis
│       ├── profile_assignment.py            # Assignment planner profiling
│       ├── configs.py                       # Evaluation configurations
│       ├── metrics.py                       # Metric computation
│       └── results_store.py                 # Result persistence
├── experiments/                             # JSON experiment configurations
├── tests/                                   # Pytest test suite (546 tests)
├── data/                                    # Pre-recorded trajectory data (NPZ files)
├── models/                                  # Pre-trained models (FastSAM-x.pt)
└── deps/                                    # Dependencies (GraspGen, GraspGenModels, YCB_Dataset)
```

## Usage

Each experiment JSON specifies the environment setup, object to manipulate, target pose, planning parameters, visualization settings, and more. See [Experiment Configuration](#experiment-configuration) below.

---

### Pipeline Scripts

#### `scripts/pipeline/trajectory_generation.py`

Main **single-robot** trajectory planner. Loads an experiment config, builds the environment, plans an object trajectory from start to goal, generates grasps, evaluates feasible EEF trajectories with multiprocessing, visualizes results in Meshcat, and optionally saves trajectories to disk. This is the script invoked by `run_all_experiments.py`.

```bash
python scripts/pipeline/trajectory_generation.py --experiment experiments/flatPlate_box_to_box.json
python scripts/pipeline/trajectory_generation.py --experiment experiments/mug_inversion.json --save-traj --no-meshcat
python scripts/pipeline/trajectory_generation.py --experiment experiments/ycb_plate_table_to_shelf.json --debug
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | **required** | Path to experiment JSON config |
| `--debug` | `False` | Enable debug logging (overrides JSON) |
| `--save-traj` | `False` | Force-save trajectories (overrides JSON) |
| `--no-meshcat` | `False` | Disable all Meshcat visualization |

#### `scripts/pipeline/dual_trajectory_generation.py`

Generates and evaluates **dual-arm** manipulation trajectories. Creates grasps for both robots, pairs them (N×M), then evaluates simultaneous feasibility for each grasp pair using multiprocessing with persistent worker environments.

```bash
python scripts/pipeline/dual_trajectory_generation.py --experiment experiments/flatPlate_box_to_box.json
python scripts/pipeline/dual_trajectory_generation.py --experiment experiments/ycb_plate_shelf_to_table.json --save-traj --no-meshcat
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | **required** | Path to experiment JSON config |
| `--debug` | `False` | Enable debug logging (overrides JSON) |
| `--save-traj` | `False` | Force-save trajectories (overrides JSON) |
| `--no-meshcat` | `False` | Disable all Meshcat visualization |

#### `scripts/pipeline/segment_assignment.py`

Solves the **optimal robot-to-segment assignment** problem using Viterbi DP. Loads pre-generated trajectory data (`TrajectoryBundle` from NPZ files), builds a cost matrix over smoothness and manipulability, and finds the globally optimal arm assignment per segment. Optionally displays feasibility heatmaps and cost breakdowns via matplotlib.

```bash
# Full run with environment (accurate IK + Jacobian costs)
python scripts/pipeline/segment_assignment.py --experiment experiments/flatPlate_box_to_box.json

# Quick dry-run without environment
python scripts/pipeline/segment_assignment.py --experiment experiments/flatPlate_box_to_box.json --no-env

# Override cost weights
python scripts/pipeline/segment_assignment.py --experiment experiments/flatPlate_box_to_box.json --alpha 15.0 --beta 2.0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | **required** | Path to experiment JSON config |
| `--debug` | `False` | Enable debug logging |
| `--no-env` | `False` | Skip environment creation (no IK/Jacobian/collision) |
| `--alpha` | from config | Override smoothness weight |
| `--beta` | from config | Override manipulability weight |
| `--no-plot` | `False` | Skip matplotlib visualizations |

#### `scripts/utils/visualize_robot_trajectory.py`

Loads `.npz` trajectory file(s) and visualizes them in Meshcat with feasibility-colored waypoints, optional robot/object mesh overlays. Supports processing a whole directory at once.

```bash
# Single trajectory with robot meshes
python scripts/utils/visualize_robot_trajectory.py data/flatPlate_box_to_box/robot0_trajectory_0.npz \
    --experiment experiments/flatPlate_box_to_box.json --robot-mesh-vis --step 5

# All trajectories in a directory
python scripts/utils/visualize_robot_trajectory.py data/flatPlate_box_to_box/ \
    --experiment experiments/flatPlate_box_to_box.json --clear --visualize-object-trajectory
```

| Argument | Default | Description |
|----------|---------|-------------|
| `path` (positional) | **required** | `.npz` file or directory of `.npz` files |
| `--experiment` | **required** | Path to experiment JSON config |
| `--robot-id` | `0` | Robot ID (0 or 1) |
| `--name` | filename | MeshCat label for the trajectory |
| `--step` | `10` | Step interval for robot mesh visualization |
| `--robot-mesh-vis` | `False` | Visualize robot meshes at waypoints |
| `--object-mesh-vis` | `False` | Visualize object meshes at waypoints |
| `--object-name` | from config | Object name to visualize |
| `--visualize-object-trajectory` | `False` | Visualize object trajectory if available |
| `--opacity` | `0.4` | Opacity for robot meshes |
| `--clear` | `False` | Clear Meshcat before visualizing |
| `--zmq-url` | `None` | Meshcat ZMQ URL |

#### `scripts/generate_all_trajs.py`

Batch runner — sequentially invokes `trajectory_generation.py` for every experiment JSON in `experiments/`. Auto-disables visualization and enables trajectory saving.

```bash
python scripts/generate_all_trajs.py
python scripts/generate_all_trajs.py --only flatPlate_box_to_box mug_inversion
python scripts/generate_all_trajs.py --workers 4
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--only` | all | Run only the listed experiment names (without `.json`) |
| `--workers` | from config | Override `num_workers` for all experiments |

#### `scripts/evaluation/pipeline_iterative.py`

Full evaluation runner for **multi-seed robustness experiments**. For each iteration, it generates both standard and no-DPP trajectories for the requested experiments, then invokes the evaluation framework (`run_all.py`) to perform assignment and execution across the selected configs. Supports crash-resilient resume and optional reuse of cached object trajectories.

```bash
python scripts/evaluation/pipeline_iterative.py --experiments mug_inversion --iterations 5
python scripts/evaluation/pipeline_iterative.py --paper --iterations 5
python scripts/evaluation/pipeline_iterative.py --paper --iterations 5 --resume
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiments` | all experiments | Run only the listed experiment names |
| `--iterations` | `1` | Number of full pipeline iterations to run |
| `--workers` | `4` | Number of workers passed to generation/evaluation steps |
| `--paper` | `False` | Run only the paper configuration set |
| `--ablations` | all configs | Run only the named ablation groups |
| `--resume` | `False` | Skip completed iterations and resume from the first incomplete one |
| `--reuse-object-traj` | `False` | Reuse cached object trajectories across iterations |
| `--base-seed` | `497` | Base RNG seed; iteration `i` uses `base_seed + i` |
| `--log-file` | `None` | Append execution logs to the given file |

---

### Utility Scripts

#### `scripts/utils/dual_arm_manipulation.py`

Starts the full `DualArmManipulationSystem` (FastSAM, GraspGen, MeshCat TF) and drops into an interactive Python shell exposing helper functions for live experimentation.

```bash
python scripts/utils/dual_arm_manipulation.py --experiment experiments/flatPlate_box_to_box.json
python scripts/utils/dual_arm_manipulation.py --experiment experiments/mug_inversion.json --visualize --meshcat-tf
python scripts/utils/dual_arm_manipulation.py --experiment experiments/flatPlate_box_to_box.json --no-fastsam --no-graspgen
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | **required** | Path to experiment JSON config |
| `--prompt` | `""` | Default detection prompt |
| `--visualize` | `False` | Enable OpenCV visualization of FastSAM |
| `--debug` | `False` | Enable debug logging |
| `--meshcat-tf` | `False` | Stream TF frames to Meshcat |
| `--no-fastsam` | (enabled) | Disable FastSAM service |
| `--no-graspgen` | (enabled) | Disable GraspGen service |

**Interactive shell commands** (available once the system starts):

```python
detections = detect()                                # Run FastSAM detection
grasps = get_grasps(detections[0])                   # Generate grasps
get_eef()                                            # Print EEF poses
set_prompt('red cup')                                # Change detection prompt
go_to_pose(pose, robot_id=0, arm_id='right')         # Move arm to pose
cancel_move(robot_id=0, arm_id='right')              # Cancel active move
follow_eef_trajectory(traj, robot_id=0, arm='right') # Follow a trajectory
system.left_arm.move_to_pose([0.5, -0.3, 0.4], [0, 0, 0, 1])
system.left_gripper.open()
system.right_gripper.close()
```

#### `scripts/pipeline/dual_arm_handover_traj.py`

Execute a dual-arm handover using pre-computed trajectory files. By default, loads the optimal assignment result (`assignment_results.npz`) from the experiment's data directory. Use `--traj0` / `--traj1` to specify trajectory files explicitly (bypasses assignment).

```bash
python scripts/pipeline/dual_arm_handover_traj.py --experiment experiments/flatPlate_box_to_box.json
python scripts/pipeline/dual_arm_handover_traj.py --experiment experiments/mug_inversion.json --debug
# Explicit trajectories (no assignment):
python scripts/pipeline/dual_arm_handover_traj.py --experiment experiments/flatPlate_box_to_box.json \
    --traj0 data/flatPlate_box_to_box/robot0_trajectory_5.npz \
    --traj1 data/flatPlate_box_to_box/robot1_trajectory_1.npz
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | **required** | Path to experiment JSON config |
| `--debug` | `False` | Enable debug logging |

---

### Demo Scripts

#### `scripts/utils/launch_env.py`

Launch and visualize the dual-arm environment with optional primitive objects and/or YCB dataset objects.

```bash
python scripts/utils/launch_env.py --with-objects
python scripts/utils/launch_env.py --ycb mug apple banana
python scripts/utils/launch_env.py --list-ycb
python scripts/utils/launch_env.py --ycb-all 5 --random-actions --steps 500
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--with-objects` | `False` | Add sample primitive objects (red block, green cylinder, blue sphere) |
| `--ycb` | `None` | Add YCB objects by name |
| `--ycb-all N` | `0` | Add the first N available YCB objects |
| `--list-ycb` | `False` | List available YCB objects and exit |
| `--random-actions` | `False` | Run random actions instead of idle visualization |
| `--steps` | `200` | Number of random-action steps |
| `--no-render` | `False` | Disable rendering |
| `--headless` | `False` | Run in headless mode |

#### `scripts/utils/visualize_rrt.py`

Visualize RRT-Connect path planning in Meshcat in real time — search bounds, tree edges (cyan/orange), collision-rejected samples (red), raw path (green), and smoothed path (yellow).

```bash
python scripts/utils/visualize_rrt.py --experiment experiments/flatPlate_box_to_box.json
python scripts/utils/visualize_rrt.py --experiment experiments/flatPlate_box_to_box.json --delay 0.01 --max-iter 2000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | **required** | Path to experiment JSON config |
| `--delay` | `0.0` | Seconds to sleep per Meshcat draw call (slow-motion) |
| `--max-iter` | from config | Override RRT max iterations |
| `--step-size` | from config | Override RRT step size |
| `--goal-bias` | from config | Override RRT goal bias |
| `--no-smooth` | `False` | Skip path smoothing |
| `--zmq-url` | `None` | Meshcat ZMQ URL |
| `--no-scene` | `False` | Skip streaming MuJoCo scene to Meshcat |

#### `scripts/demos/go_to_pose.py`

Demonstrates Cartesian pose control. Creates a `DualArmPoseController`, starts a Meshcat TF visualizer, and commands each end-effector to move to a target pose. **No CLI arguments.**

```bash
python scripts/demos/go_to_pose.py
```

#### `scripts/demos/gripper_move.py`

Interactive gripper demo — opens and closes both robot grippers in sequence, pausing for user input between steps. **No CLI arguments.**

```bash
python scripts/demos/gripper_move.py
```

---

### Teleoperation

#### `scripts/demos/teleop_dual_arm.py`

Keyboard teleoperation of both Panda robots with configurable sensitivity.

```bash
python scripts/demos/teleop_dual_arm.py
python scripts/demos/teleop_dual_arm.py --pos-sensitivity 2.0 --rot-sensitivity 0.5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--pos-sensitivity` | `1.0` | Scale factor for position inputs |
| `--rot-sensitivity` | `1.0` | Scale factor for rotation inputs |

**Teleoperation keybindings:**

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

---

### Transform Visualization

#### `scripts/utils/transform_visualizer_meshcat.py`

RViz-like TF visualizer for Meshcat. Continuously streams all `TransformManager` frames (coordinate triads + text labels) to a running Meshcat server. Requires `meshcat-server` to be running separately.

```bash
# Terminal 1: start meshcat server
meshcat-server

# Terminal 2: run the visualizer
python scripts/utils/transform_visualizer_meshcat.py
python scripts/utils/transform_visualizer_meshcat.py --fps 10 --no-labels
python scripts/utils/transform_visualizer_meshcat.py --axis-length 0.2 --axis-radius 0.01
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--fps` | `20.0` | Update rate in Hz |
| `--zmq-url` | `tcp://127.0.0.1:6000` | Meshcat ZMQ URL |
| `--axis-length` | `0.12` | Axis triad length (metres) |
| `--axis-radius` | `0.008` | Axis triad radius (metres) |
| `--labels` / `--no-labels` | enabled | Render frame name labels |
| `--label-offset` | `0.05` | Z offset (m) for label placement |
| `--no-clear` | `False` | Don't clear visualizer on start |

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
    "relevant_patterns": ["flatPlate", "table_shelf"]
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

**Available experiments** (4 main + 9 extras):
- **Main**: `flatPlate_box_to_box`, `mug_inversion`, `ycb_plate_shelf_to_table`, `ycb_plate_table_to_shelf`
- **Extras** (in `experiments/extras/`): `flatPlate_shelf_to_box`, `flatPlate_shelf_to_shelf`, `flatPlate_shelf_to_table`, `flatPlate_table_to_shelf`, `dual_handover_mug`, `dual_handover_plate`, `ycb_bowl_rack_to_table`, `ycb_bowl_table_to_rack`, `ycb_can_table_to_rack`, `ycb_mug_rack_to_table`, `ycb_mug_table_to_rack`.

## Microservices Architecture

All services communicate via ZMQ with JSON messages.

### Service Manager (port 6006)
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

Key constants are defined in `src/bimanual_manipulation/constants.py`:

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
from bimanual_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController

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
from bimanual_manipulation.perception import ObjectDetector, GraspPredictor

# Object detection
detector = ObjectDetector(camera_intrinsics, camera_width, camera_height)
objects = detector.detect_objects(rgb_image, depth_image, camera_pos, camera_rot)

# Grasp prediction
predictor = GraspPredictor(gripper_type='franka_panda')
grasps = predictor.predict_grasps(object_pcd, object_pose, num_grasps=5)
```

### Planning

```python
from bimanual_manipulation.planning import (
    plan_object_trajectory,           # RRT-Connect in SE(3)
    generate_object_trajectory,       # SE(3) interpolation (lift-approach-insert)
    generate_end_effector_trajectory, # Object → EEF trajectory via grasp transform
    check_trajectory_feasibility,     # IK + collision validation
    check_dual_trajectory_feasibility,
    # Segment assignment
    compute_optimal_robot_assignment, # Viterbi DP solver
    AssignmentResult,                 # Result dataclass
    ObjectTrajectory,                 # Typed object trajectory
    RobotTrajectory,                  # Typed robot trajectory
    TrajectoryBundle,                 # Bundle of all trajectories
    load_all_trajectories,            # Load NPZ files → TrajectoryBundle
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
from bimanual_manipulation.transforms import TransformManager, StampedPose

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
from bimanual_manipulation.utils.se3 import (
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
from bimanual_manipulation.services import (
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
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_schema.py -v                 # Pydantic schema validation
python -m pytest tests/test_assignment_planner.py -v     # Assignment planning
python -m pytest tests/test_rrt_joint_planner.py -v      # RRT-Connect planner
python -m pytest tests/test_grasps.py -v                 # Grasp transforms
python -m pytest tests/test_transforms.py -v             # Coordinate transforms
python -m pytest tests/test_trajectory_planning.py -v    # SE(3) & trajectory planning
python -m pytest tests/test_trajectory_math.py -v        # Quintic profiling & resampling
python -m pytest tests/test_pointcloud.py -v             # Point cloud processing
python -m pytest tests/test_dual_arm_planner.py -v       # Dual-arm collision avoidance
python -m pytest tests/test_planning_module.py -v        # IK backends & trajectory feasibility
python -m pytest tests/test_experiment_config.py -v      # Experiment config loading
python -m pytest tests/test_arm_controllers.py -v        # Arm controllers
python -m pytest tests/test_environments.py -v           # Environment setup
python -m pytest tests/test_fastsam_detection.py -v      # FastSAM service
python -m pytest tests/test_services.py -v               # Service manager
python -m pytest tests/test_move_to_pose_service.py -v   # MoveToPose service
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
