#!/usr/bin/env python3
"""
Constants file for robosuite_franka_manipulation package.
"""

import logging
import numpy as np
import torch
from pathlib import Path

# =============================================================================
# Network and Service Configuration
# =============================================================================

# Default host for all services
DEFAULT_HOST = 'localhost'

# Service ports
SERVICE_MANAGER_PORT = 5555
FASTSAM_SERVICE_PORT = 5551
GRASP_SERVICE_PORT = 5552
MOVE_TO_POSE_SERVICE_PORT = 5553

# ZMQ timeouts (in milliseconds)
ZMQ_POLL_TIMEOUT = 1000  # 1 second for service polling
ZMQ_MANAGER_TIMEOUT = 5.5 * 60 * 1000  # 5.5 mins... for manager calls (must be >= service call timeout)
ZMQ_SERVER_RECV_TIMEOUT = 1000  # 1 second for server receive polling
ZMQ_SERVICE_CALL_TIMEOUT = 5 * 60 * 1000  # 5 mins... seconds for service calls

# =============================================================================
# Model and Path Configuration
# =============================================================================

# Model paths relative to project root
FASTSAM_MODEL_PATH = 'models/FastSAM-x.pt'
GRASPGEN_MODELS_DIR = 'deps/GraspGenModels/checkpoints'
GRASPGEN_DEPS_DIR = 'deps/GraspGen'
DATA_DIR = 'data'

# GraspGen config file template
GRASPGEN_CONFIG_TEMPLATE = 'graspgen_{gripper_type}.yml'

# =============================================================================
# Gripper Configuration
# =============================================================================

# Gripper action mappings
GRIPPER_TYPE_MAPPING = {
    'finger': 'franka_panda',
    'suction': 'single_suction_cup_30mm'
}

# Supported gripper types
GRIPPER_TYPES = list(GRIPPER_TYPE_MAPPING.keys())

# Gripper grasp offset transforms
# These offsets are applied to predicted grasps to align them with the gripper frame
# Format: (position_offset, quaternion_offset) where quaternion is (x, y, z, w)
# For finger gripper: 90 degree rotation around Z-axis
# For suction gripper: identity transform (no offset)
GRIPPER_GRASP_OFFSETS = {
    'franka_panda': {
        'position': np.array([0.0, 0.0, 0.02]),
        'quaternion': np.array([0.0, 0.0, 0.70710678, 0.70710678]),  # 90 deg around Z
    },
    'single_suction_cup_30mm': {
        'position': np.array([0.0, 0.0, -0.12]),
        'quaternion': np.array([0.0, 0.0, 0.0, 1.0]),  # 180 deg around Y
    },
}

# =============================================================================
# Perception and Detection Configuration
# =============================================================================

# Default camera settings
DEFAULT_CAMERA_NAME = 'agentview'
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_INTRINSICS = np.array([
    [800.0, 0, 320.0],
    [0, 800.0, 240.0],
    [0, 0, 1]
])

# Point cloud validation
MAX_POINTCLOUD_RANGE = 10.0  # meters

# Default detection prompt
DEFAULT_DETECTION_PROMPT = ""

# FastSAM detection parameters
FASTSAM_CONF_THRESHOLD = 0.9  # Confidence threshold for detections
FASTSAM_IOU_THRESHOLD = 0.4   # IoU threshold for NMS

# Compute device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# Grasp Prediction Configuration
# =============================================================================

# Default number of grasps to predict
DEFAULT_NUM_GRASPS = 200
GRASPGEN_POINTCLOUD_SIZE = 4096  # Number of points for GraspGen input
GRIPPER_COLLISION_MESH_SAMPLES = 2000  # Number of points to sample on gripper collision mesh for occupancy checking
MESH_SCALE = 1.0  # Default scale for object meshes
GRIPPER_COLLISION_THRESHOLD = 0.005  # Collision threshold (meters) for gripper occupancy checking
GRASPGEN_PREDICTION_THRESHOLD = 0.8  # Detection confidence threshold for using GraspGen
MIN_GRASPS = 50
DEFAULT_VISUALIZE_GRIPPER_MESH = True  # Whether to visualize gripper mesh during grasp prediction

ENTROPY_POSITION_SCALE = 5
ENTROPY_ORIENTATION_SCALE = 5

# =============================================================================
# Planning Configuration
# =============================================================================

DEFAULT_IK_MAX_SOLUTIONS = 5  # Max IK solutions to check for single grasp feasibility
DEFAULT_FEASIBILITY_MAX_PAIRS = 5  # Max pairs to check for dual-arm feasibility (1 = just need existence)
DEFAULT_LIFT_HEIGHT = 0.1  # Default lift height in meters
DEFAULT_LIFT_STEPS = 10  # Number of waypoints for lift trajectory

# IK Solver Parameters
IK_NUM_SEED_CONFIGS = 48  # Number of random seed configurations for IK
IK_ELBOW_DISCRETIZATION = 8  # Number of elbow angle discretization steps
IK_MAX_SOLUTIONS = 32  # Maximum IK solutions to enumerate
IK_POS_TOLERANCE = 1e-3  # Position tolerance (meters)
IK_ORI_TOLERANCE = 5e-3  # Orientation tolerance (radians)
IK_DAMPING = 1e-2  # Damping factor for DLS IK
IK_STEP_SIZE = 0.6  # Step size for IK updates
IK_UNIQUE_TOLERANCE = 1e-3  # Tolerance for unique solution filtering
IK_RANDOM_PERTURBATION_SCALE = 0.25  # Std dev for random seed perturbation

# Trajectory Feasibility Check Parameters
TRAJ_IK_NUM_SEED_CONFIGS = 32  # Seeds for trajectory waypoints
TRAJ_IK_MAX_SOLUTIONS = 5  # Check multiple solutions per waypoint

# Collision Checking Parameters
COLLISION_CONTACT_THRESHOLD = 1e-6  # Contact distance threshold

# Dual-Arm Feasibility Parameters
DUAL_ARM_MAX_PAIRS = 256  # Maximum pairs to check for dual-arm feasibility

# =============================================================================
# RRT-Connect Object Trajectory Planner
# =============================================================================

RRT_MAX_ITERATIONS = 5000       # Maximum RRT-Connect iterations
RRT_STEP_SIZE = 0.05            # Step size for tree extension (meters in SE(3) weighted distance)
RRT_GOAL_BIAS = 0.15            # Probability of sampling the goal pose
RRT_ROTATION_WEIGHT = 0.1       # Weight for rotation distance (radians -> equivalent meters)
RRT_COLLISION_RESOLUTION = 0.02 # Resolution for edge collision checking (meters)
RRT_CONNECT_MAX_STEPS = 50      # Max connect steps toward a target node
RRT_SMOOTHING_ITERATIONS = 200  # Number of shortcutting attempts for path smoothing
RRT_ORIENTATION_NOISE = 0.3     # Max random perturbation angle (radians) for orientation sampling
RRT_BOUNDS_MARGIN = [0.3, 0.3, 0.3]  # Margin around start/goal for auto workspace bounds (meters)
RRT_MIN_Z = 1.18                # Minimum z height for planning (slightly below table surface)
# Default workspace bounds [lo, hi] — used only when auto-bounds are not sufficient
RRT_WORKSPACE_BOUNDS = [[-0.8, -0.8, 1.18], [0.8, 0.8, 2.0]]

# =============================================================================
# Control and Trajectory Configuration
# =============================================================================

# Object Trajectory Generation
INTERPOLATION_TYPE = 'linear'  # 'linear' (decoupled lerp+slerp) or 'SE3' (geodesic on SE(3) manifold)
OBJECT_TRAJ_NUM_STEPS = 50  # Default number of steps for object trajectory
OBJECT_TRAJ_LIFT_HEIGHT = 0.2  # Height to lift object during trajectory (meters)
OBJECT_TRAJ_APPROACH_OFFSET = 0.3  # Approach offset distance (meters)
# Segment ratios for object trajectory (lift, approach, insert)
OBJECT_TRAJ_LIFT_RATIO = 0.2
OBJECT_TRAJ_APPROACH_RATIO = 0.5
OBJECT_TRAJ_INSERT_RATIO = 0.3

# Trajectory planning - number of waypoints
TRAJECTORY_NUM_POINTS = 200  # Number of waypoints for quintic polynomial trajectory
TRAJECTORY_TIME_STEP = 0.05  # Time step between waypoints (seconds)

# Trajectory planning thresholds
TRAJECTORY_CONVERGENCE_THRESHOLD = 0.01
TRAJECTORY_ORIENTATION_SCALING = 0.05  # Scaling for orientation in SE(3) distance calculation
TRAJECTORY_MIN_DISTANCE = 1e-6  # Minimum distance threshold to consider movement needed

# Gripper width limits (normalized)
GRIPPER_OPEN_WIDTH = 1.0
GRIPPER_CLOSE_WIDTH = 0.0

# Visualization defaults
VIS_PLANNING_ROOT = "planning"  # Root path in MeshCat for all planning visualizations
VIS_TRAJECTORY_PATH = "trajectories"  # Subpath for trajectory visualizations
VIS_ROBOT_CONFIG_PATH = "robot_configs"  # Subpath for robot mesh configurations
VIS_MESH_STEP_INTERVAL = 10  # Visualize robot mesh every N waypoints
VIS_MESH_OPACITY = 0.4  # Default opacity for robot mesh visualization

# =============================================================================
# Teleoperation Configuration
# =============================================================================

# Default sensitivity values
DEFAULT_POS_SENSITIVITY = 1.0
DEFAULT_ROT_SENSITIVITY = 1.0

# =============================================================================
# Environment Configuration
# =============================================================================

# Default environment parameters
DEFAULT_CONTROL_FREQ = 20  # Hz
DEFAULT_TELEOP_FREQ = 10  # Hz
DEFAULT_HORIZON = 1000  # steps
DEFAULT_CAMERA_DEPTH = True
DEFAULT_RENDER_CAMERA = 'free'

# Environment setup
DEFAULT_ROBOT_TYPES = ['Panda', 'Panda']
DEFAULT_GRIPPER_TYPES_ENV = ['PandaGripper', 'PandaSuctionGripper']
DEFAULT_ENV_CONFIG = 'opposed'
DEFAULT_TABLE_SIZE = (0.75, 0.75, 0.05)
DEFAULT_TABLE_FRICTION = (1.0, 5e-3, 1e-4)
DEFAULT_USE_OBJECT_OBS = True
# Number of arm joints on a Franka Panda (excludes gripper DOF).
# Per-robot action dim = FRANKA_ARM_DOF + gripper_dof and depends on the
# controller; use robot.action_dim at runtime instead of hard-coding.
FRANKA_ARM_DOF = 7

# Robot base configurations (position, quaternion) from world frame
# quaternions are [x, y, z, w]
ROBOT_CONFIGURATIONS = {
    'opposed': [
        {
            'position': np.array([0.0, -0.05, 0.0]),
            'quaternion': np.array([0.0, 0.0, 0.70710678, 0.70710678]),  # 90 deg z
        },
        {
            'position': np.array([0.0, 0.05, 0.0]),
            'quaternion': np.array([0.0, 0.0, -0.70710678, 0.70710678]),  # -90 deg z
        }
    ],
    'parallel': [
        {
            'position': np.array([0.0, -0.05, 0.0]),
            'quaternion': np.array([0.0, 0.0, 0.0, 1.0]),  # 0 deg
        },
        {
            'position': np.array([0.0, 0.05, 0.0]),
            'quaternion': np.array([0.0, 0.0, 0.0, 1.0]),  # 0 deg
        }
    ],
}

# Renderer settings
DEFAULT_HAS_RENDERER = True
DEFAULT_HAS_OFFSCREEN_RENDERER = True
DEFAULT_USE_CAMERA_OBS = True

# Controller configuration
DEFAULT_CONTROLLER_TYPE = 'BASIC'
DEFAULT_CONTROLLER_NAME = 'JOINT_POSITION'
DEFAULT_GRIPPER_CONTROLLER_TYPE = 'GRIP'
DEFAULT_INPUT_REF_FRAME = 'base'  # Controller input reference frame (base = robot base frame)
DEFAULT_INPUT_TYPE = 'absolute'

DEFAULT_KP = 1000  # Position stiffness (default is ~150)
DEFAULT_DAMPING_RATIO = 1.0  # Critical damping (default is ~1.0)
DEFAULT_KP_LIMITS = [0, 1000]  # Allow higher stiffness values
DEFAULT_DAMPING_RATIO_LIMITS = [0, 2]  # Allow higher damping

# =============================================================================
# Controller Types and Input Modes
# =============================================================================

# Supported controller names (from robosuite)
class ControllerName:
    """Supported controller names."""
    OSC_POSE = 'OSC_POSE'           # Operational Space Control for end-effector pose
    JOINT_POSITION = 'JOINT_POSITION'  # Joint position control

# Supported input types
class InputType:
    """Input types for controllers."""
    DELTA = 'delta'       # Delta (incremental) control
    ABSOLUTE = 'absolute'  # Absolute (target) control

# Valid combinations
SUPPORTED_CONTROLLER_NAMES = [ControllerName.OSC_POSE, ControllerName.JOINT_POSITION]
SUPPORTED_INPUT_TYPES = [InputType.DELTA, InputType.ABSOLUTE]

# Action dimensions for different controller types (excluding gripper)
# For Franka Panda: 7 DOF arm
CONTROLLER_ACTION_DIMS = {
    ControllerName.OSC_POSE: 6,        # [x, y, z, rx, ry, rz]
    ControllerName.JOINT_POSITION: 7,  # [j1, j2, j3, j4, j5, j6, j7]
}

# Default home joint configuration for Franka Panda (radians)
# This is a neutral pose with arm slightly bent
FRANKA_HOME_JOINT_POSITION = np.array([
    0.0,    # joint1
    -0.785, # joint2 (-45 deg)
    0.0,    # joint3
    -2.356, # joint4 (-135 deg)
    0.0,    # joint5
    1.571,  # joint6 (90 deg)
    0.785,  # joint7 (45 deg)
])

# Initial joint configuration for higher EEF pose (avoids table collision)
# Used to override the default Panda init_qpos which places grippers too low.
FRANKA_HIGH_INIT_QPOS = np.array([
    0.0,     # joint1
    -0.5,    # joint2  (more upright than default 0.196)
    0.0,     # joint3
    -2.0,    # joint4  (less bent than default -2.618)
    0.0,     # joint5
    2.0,     # joint6  (adjusted for higher EEF)
    0.785,   # joint7  (45 deg)
])

# =============================================================================
# Coordinate Frame Names
# =============================================================================

# Standard frame names used throughout the codebase
WORLD_FRAME = 'world'           # Global world frame (simulation origin)
BASE_FRAME = 'base'             # Robot base frame (may have offset from world)
EEF_FRAME_TEMPLATE = 'robot{robot_id}_{arm}_eef'  # End-effector frame template

# Reward and episode settings
DEFAULT_IGNORE_DONE = True

# =============================================================================
# Logger Configuration
# =============================================================================

LOG_LEVEL = logging.INFO  # Default logging level

# =============================================================================
# Meshcat Configuration
# =============================================================================

MESHCAT_TF_VISUALIZER_FPS = 20.0
MESHCAT_ZMQ_URL = "tcp://127.0.0.1:6000"
MESHCAT_VISUALIZE_GEOMETRY = True

# =============================================================================
# Utility Functions
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def get_graspgen_config_path(gripper_type: str) -> Path:
    """Get full path to GraspGen config for a gripper type."""
    config_name = GRASPGEN_CONFIG_TEMPLATE.format(gripper_type=gripper_type)
    return get_project_root() / GRASPGEN_MODELS_DIR / config_name

# =============================================================================
# YCB Dataset and Custom Objects Configuration
# =============================================================================

# Path to YCB dataset relative to project root
YCB_DATASET_PATH = Path(get_project_root()) / "deps" / "YCB_Dataset" / "ycb"
DEFAULT_OBJECT_POSITION = [0.0, 0.0, 1.30]
DEFAULT_OBJECT_QUATERNION = [0.0, 0.0, 0.0, 1.0]

# =============================================================================
# Asset Paths
# =============================================================================

# Path to suction gripper XML asset
SUCTION_GRIPPER_XML_PATH = get_project_root() / "src" / "robosuite_franka_manipulation" / "assets" / "grippers" / "panda_suction.xml"

# Path to custom suction gripper collision mesh
COBOT_PUMP_MESH_PATH = get_project_root() / "src" / "robosuite_franka_manipulation" / "assets" / "meshes" / "cobot_pump" / "collision" / "cobot_pump.stl"

# Suction TCP offset relative to its connection site (base)
# This is usually the distance from the flange to the suction cup tip
SUCTION_TCP_OFFSET = np.array([0.0, 0.0, 0.11]) 
