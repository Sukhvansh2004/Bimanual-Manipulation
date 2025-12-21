#!/usr/bin/env python3
"""
Constants file for robosuite_franka_manipulation package.
"""

import numpy as np
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
ZMQ_MANAGER_TIMEOUT = 15000  # 15 seconds for manager calls
ZMQ_SERVER_RECV_TIMEOUT = 1000  # 1 second for server receive polling
ZMQ_SERVICE_CALL_TIMEOUT = 10000  # 10 seconds for service calls

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
DEFAULT_DETECTION_PROMPT = "object"

# FastSAM detection parameters
FASTSAM_CONF_THRESHOLD = 0.9  # Confidence threshold for detections
FASTSAM_IOU_THRESHOLD = 0.4   # IoU threshold for NMS

# =============================================================================
# Grasp Prediction Configuration
# =============================================================================

# Default number of grasps to predict
DEFAULT_NUM_GRASPS = 5

# =============================================================================
# Control and Trajectory Configuration
# =============================================================================

# Default control parameters
DEFAULT_MAX_VELOCITY = 0.5  # m/s
DEFAULT_MAX_ACCELERATION = 0.1  # m/s²
DEFAULT_TIME_STEP = 0.02  # seconds

# Trajectory planning
TRAJECTORY_POS_ERROR_SCALE = 5  # Scaling factor for position control
TRAJECTORY_ORI_ERROR_SCALE = 1  # Scaling factor for orientation control
TRAJECTORY_CONVERGENCE_THRESHOLD = 0.01  # Position convergence threshold
TRAJECTORY_MAX_TIMEOUT = 100  # Max steps to reach target
TRAJECTORY_ORIENTATION_SCALING = 0.05  # Scaling for orientation in distance calculation
TRAJECTORY_MIN_DISTANCE = 1e-6  # Minimum distance threshold
TRAJECTORY_MIN_STEPS = 2  # Minimum trajectory steps
TRAJECTORY_SLEEP_TIME = 2  # Sleep time after trajectory completion

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
DEFAULT_ENV_NAME = 'TwoArmHandover'
DEFAULT_ROBOT_TYPES = ['Panda', 'Panda']
DEFAULT_GRIPPER_TYPES_ENV = 'default'
DEFAULT_ENV_CONFIG = 'opposed'

# Renderer settings
DEFAULT_HAS_RENDERER = True
DEFAULT_HAS_OFFSCREEN_RENDERER = True
DEFAULT_USE_CAMERA_OBS = True

# Controller configuration
DEFAULT_CONTROLLER_TYPE = 'BASIC'
DEFAULT_CONTROLLER_NAME = 'OSC_POSE'
DEFAULT_GRIPPER_CONTROLLER_TYPE = 'GRIP'
DEFAULT_INPUT_REF_FRAME = 'base'
DEFAULT_INPUT_TYPE = 'delta'

# Reward and episode settings
DEFAULT_REWARD_SHAPING = False
DEFAULT_IGNORE_DONE = True

# =============================================================================
# Utility Functions
# =============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent

def get_model_path(model_name: str) -> Path:
    """Get full path to a model file."""
    return get_project_root() / model_name

def get_graspgen_config_path(gripper_type: str) -> Path:
    """Get full path to GraspGen config for a gripper type."""
    config_name = GRASPGEN_CONFIG_TEMPLATE.format(gripper_type=gripper_type)
    return get_project_root() / GRASPGEN_MODELS_DIR / config_name