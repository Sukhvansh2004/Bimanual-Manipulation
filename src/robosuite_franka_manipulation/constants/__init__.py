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
EEF_SERVICE_PORT = 5559

# ZMQ timeouts (in milliseconds)
ZMQ_POLL_TIMEOUT = 1000  # 1 second for service polling
ZMQ_MANAGER_TIMEOUT = 120000  # 120 seconds for manager calls (must be >= service call timeout)
ZMQ_SERVER_RECV_TIMEOUT = 1000  # 1 second for server receive polling
ZMQ_SERVICE_CALL_TIMEOUT = 100000  # 100 seconds for service calls

# =============================================================================
# Model and Path Configuration
# =============================================================================

# Model paths relative to project root
FASTSAM_MODEL_PATH = 'models/FastSAM-x.pt'
GRASPGEN_MODELS_DIR = 'deps/GraspGenModels/checkpoints'
GRASPGEN_DEPS_DIR = 'deps/GraspGen'

# GraspGen config file template
GRASPGEN_CONFIG_TEMPLATE = 'graspgen_{gripper_type}.yml'

# =============================================================================
# Gripper Configuration
# =============================================================================

# Supported gripper types
GRIPPER_TYPES = ['franka_panda', 'single_suction_cup_30mm']

# Default gripper type
DEFAULT_GRIPPER_TYPE = 'franka_panda'

# Gripper action mappings
GRIPPER_TYPE_MAPPING = {
    'finger': 'franka_panda',
    'suction': 'single_suction_cup_30mm'
}

# Gripper grasp offset transforms
# These offsets are applied to predicted grasps to align them with the gripper frame
# Format: (position_offset, quaternion_offset) where quaternion is (x, y, z, w)
# For finger gripper: 90 degree rotation around Z-axis
# For suction gripper: identity transform (no offset)
GRIPPER_GRASP_OFFSETS = {
    'franka_panda': {
        'position': np.array([0.0, 0.0, 0.0]),
        'quaternion': np.array([0.0, 0.0, 0.70710678, 0.70710678]),  # 90 deg around Z
    },
    'single_suction_cup_30mm': {
        'position': np.array([0.0, 0.0, 0.0]),
        'quaternion': np.array([0.0, 0.0, 0.0, 1.0]),  # Identity
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
DEFAULT_RENDER_CAMERA = 'free'

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
DEFAULT_NUM_GRASPS = 5

# =============================================================================
# Control and Trajectory Configuration
# =============================================================================

# Trajectory planning - number of waypoints
TRAJECTORY_NUM_POINTS = 100  # Number of waypoints for quintic polynomial trajectory
TRAJECTORY_TIME_STEP = 0.05  # Time step between waypoints (seconds)

# Trajectory planning thresholds
TRAJECTORY_CONVERGENCE_THRESHOLD = 0.01  # Position convergence threshold (1cm)
TRAJECTORY_ORIENTATION_SCALING = 0.05  # Scaling for orientation in SE(3) distance calculation
TRAJECTORY_MIN_DISTANCE = 1e-6  # Minimum distance threshold to consider movement needed

# Gripper width limits (normalized)
GRIPPER_OPEN_WIDTH = 1.0
GRIPPER_CLOSE_WIDTH = 0.0

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
DEFAULT_HORIZON = 1000  # steps
DEFAULT_CAMERA_DEPTH = True

# Environment setup
DEFAULT_ROBOT_TYPES = ['Panda', 'Panda']
DEFAULT_GRIPPER_TYPES_ENV = 'default'
DEFAULT_ENV_CONFIG = 'opposed'
DEFAULT_TABLE_SIZE = (0.8, 0.8, 0.05)
DEFAULT_TABLE_FRICTION = (1.0, 5e-3, 1e-4)
DEFAULT_USE_OBJECT_OBS = True

# Renderer settings
DEFAULT_HAS_RENDERER = True
DEFAULT_HAS_OFFSCREEN_RENDERER = True
DEFAULT_USE_CAMERA_OBS = True

# Controller configuration
DEFAULT_CONTROLLER_TYPE = 'BASIC'
DEFAULT_CONTROLLER_NAME = 'OSC_POSE'
DEFAULT_GRIPPER_CONTROLLER_TYPE = 'GRIP'
DEFAULT_INPUT_REF_FRAME = 'base'  # Controller input reference frame (base = robot base frame)
DEFAULT_INPUT_TYPE = 'delta'

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

LOG_LEVEL = logging.DEBUG  # Default logging level

# =============================================================================
# Meshcat Configuration
# =============================================================================

MESHCAT_ZMQ_URL = "tcp://127.0.0.1:6000"

# =============================================================================
# Utility Functions
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent

def get_graspgen_config_path(gripper_type: str) -> Path:
    """Get full path to GraspGen config for a gripper type."""
    config_name = GRASPGEN_CONFIG_TEMPLATE.format(gripper_type=gripper_type)
    return get_project_root() / GRASPGEN_MODELS_DIR / config_name

# =============================================================================
# YCB Dataset and Custom Objects Configuration
# =============================================================================

# Path to YCB dataset relative to project root
YCB_DATASET_PATH = Path(get_project_root()) / "deps" / "YCB_Dataset" / "ycb"
DEFAULT_OBJECT_POSITION = [0.0, -0.45, 0.90]