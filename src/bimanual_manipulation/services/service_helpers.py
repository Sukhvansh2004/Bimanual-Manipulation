"""Service-client convenience wrappers for interactive / scripting use."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import meshcat
import numpy as np

import bimanual_manipulation.constants as constants
from bimanual_manipulation.planning.trajectory.utils import (
    WaypointFeasibilityResult,
    load_trajectory,
)
from bimanual_manipulation.planning.visualization.trajectory import visualize_trajectory
from bimanual_manipulation.transforms.stamped_pose import StampedPose

if TYPE_CHECKING:
    from bimanual_manipulation.system_manager import DualArmManipulationSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def detect(system: DualArmManipulationSystem):
    """Fetch the latest FastSAM detections and log a summary.

    Args:
        system: Running manipulation system with a service client.

    Returns:
        List of detection dicts, or ``[]`` on failure.

    Raises:
        AssertionError: If ``system.service_client`` is ``None``.
    """
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('fastsam_service', {'command': 'GET_LATEST_DETECTION'})
    if response.get('status') == 'success':
        detections = response.get('detections', [])
        logger.info(f"\nDetected {len(detections)} object(s):")
        for i, det in enumerate(detections):
            pc_size = len(det.get('pcd', []))
            pose = det.get('pose', {})
            logger.info(f"  Object {i+1}: {pc_size} points, pos={pose.get('pos', 'N/A')}")
        return detections
    else:
        logger.info(f"Detection failed: {response.get('message')}")
        return []


def set_prompt(system: DualArmManipulationSystem, prompt: str):
    """Set the FastSAM detection prompt (no-op if FastSAM is disabled).

    Args:
        system: Running manipulation system with a service client.
        prompt: New text prompt.

    Raises:
        AssertionError: If FastSAM is enabled and ``system.service_client``
            is ``None``.
    """
    if not system.use_fastsam:
        logger.info("FastSAM service is disabled.")
        return
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('fastsam_service', {'command': 'SET_PROMPT', 'prompt': prompt})
    if response.get('status') == 'success':
        logger.info(f"Prompt changed to: '{prompt}'")
    else:
        logger.info(f"Failed to change prompt: {response.get('message')}")


# ---------------------------------------------------------------------------
# Grasp helpers
# ---------------------------------------------------------------------------

def get_grasps(system: DualArmManipulationSystem, detection, gripper_type, num_grasps=constants.DEFAULT_NUM_GRASPS, collision_check: Optional[str] = None):
    """Request grasps for a detection's point cloud from the Grasp service.

    Args:
        system: Running manipulation system with a service client.
        detection: Detection dict with ``pcd`` and ``pose`` entries.
        gripper_type: Gripper type key(s) passed through as
            ``GraspRequest.gripper_type`` (expects a list).
        num_grasps: Number of grasps to request.
        collision_check: ``'Inter'``, ``'Intra'`` or ``None``.

    Returns:
        List of grasp dicts, or ``[]`` on invalid detection or failure.

    Raises:
        AssertionError: If ``system.service_client`` is ``None``.
    """
    pcd = detection.get('pcd', [])
    pose = detection.get('pose', {})

    if not pcd or not pose:
        logger.info("Invalid detection - missing pcd or pose")
        return []

    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('grasp_service', {
        'command': 'GET_GRASPS',
        'pcd': pcd,
        'object_pose': pose,
        'gripper_type': gripper_type,
        'num_grasps': num_grasps,
        'collision_check': collision_check
    })

    if response.get('status') == 'success':
        grasps = response.get('grasps', [])
        logger.info(f"\nGenerated {len(grasps)} grasp(s):")
        for i, grasp in enumerate(grasps[:3]):
            logger.info(f"  Grasp {i+1}: pos={grasp.get('pos')}, score={grasp.get('score', 'N/A'):.3f}")
        return grasps
    else:
        logger.info(f"Grasp generation failed: {response.get('message')}")
        return []


def get_grasps_mesh(system: DualArmManipulationSystem, pose: StampedPose, object_name: str, num_grasps=constants.DEFAULT_NUM_GRASPS, gripper_type='suction', collision_check: Optional[str] = None, visualize_gripper_mesh=True):
    """Request mesh-based grasps for a known object from the Grasp service.

    Args:
        system: Running manipulation system with a service client.
        pose: Object pose.
        object_name: Object name used to look up its mesh.
        num_grasps: Number of grasps to request.
        gripper_type: Single gripper type key (wrapped in a list).
        collision_check: ``'Inter'``, ``'Intra'`` or ``None``.
        visualize_gripper_mesh: Whether the service visualizes gripper meshes.

    Returns:
        List of grasp dicts, or ``[]`` on failure.

    Raises:
        AssertionError: If ``system.service_client`` is ``None``.
    """
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('grasp_service', {
        'command': 'GET_GRASPS',
        'object_pose': pose.model_dump(),
        'object_name': object_name,
        'num_grasps': num_grasps,
        'gripper_type': [gripper_type],
        'collision_check': collision_check,
        'visualize_gripper_mesh': visualize_gripper_mesh
    })

    if response.get('status') == 'success':
        grasps = response.get('grasps', [])
        logger.info(f"\nGenerated {len(grasps)} grasp(s):")
        for i, grasp in enumerate(grasps[:3]):
            logger.info(f"  Grasp {i+1}: pos={grasp.get('pos')}, score={grasp.get('score', 'N/A'):.3f}")
        return grasps
    else:
        logger.info(f"Grasp generation failed: {response.get('message')}")
        return []


# ---------------------------------------------------------------------------
# Transform / EEF helpers
# ---------------------------------------------------------------------------

def get_eef(system: DualArmManipulationSystem):
    """Get and log the world-frame pose of every robot EEF frame.

    Args:
        system: Manipulation system with ``tf_manager`` and ``env``.

    Returns:
        Dict mapping ``robot{i}_{arm}_eef`` to ``position``/``quaternion``
        dicts and ``frame_id``, or ``None`` if unavailable or on error.
    """
    if system.tf_manager is None:
        logger.info("TransformManager not initialized")
        return None
    try:
        eef_data = {}
        if system.env is None:
            logger.info("Environment not initialized")
            return None
        for robot_idx, robot in enumerate(system.env.robots):
            robot_name = f"robot{robot_idx}"
            try:
                arm_ids = list(robot.robot_model.eef_name.keys())
            except Exception:
                arm_ids = ["right"]
            for arm_id in arm_ids:
                frame_name = f"{robot_name}_{arm_id}_eef"
                if system.tf_manager.frame_exists(frame_name):
                    pose = system.tf_manager.get_frame_pose(frame_name, "world")
                    eef_data[frame_name] = {
                        "position": {"x": pose.position[0], "y": pose.position[1], "z": pose.position[2]},
                        "quaternion": {"x": pose.quaternion[0], "y": pose.quaternion[1], "z": pose.quaternion[2], "w": pose.quaternion[3]},
                        "frame_id": pose.frame_id
                    }
        logger.info("\nEnd-Effector Coordinates:")
        for frame_name, data in eef_data.items():
            pos = data.get('position', {})
            logger.info(f"  {frame_name}: x={pos.get('x', 0):.4f}, y={pos.get('y', 0):.4f}, z={pos.get('z', 0):.4f}")
        return eef_data
    except Exception as e:
        logger.info(f"Failed to get EEF: {e}")
        return None


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------

def go_to_pose(system: DualArmManipulationSystem, pose: StampedPose, robot_id: int, arm_id: str, gripper_action: float = 0.0, num_steps: Optional[int] = None):
    """Send a non-blocking MOVE_TO_POSE command and log the result.

    Args:
        system: Running manipulation system with a service client.
        pose: Target EEF pose.
        robot_id: Robot index.
        arm_id: Arm identifier.
        gripper_action: Gripper command during motion.
        num_steps: Number of trajectory waypoints (service default if ``None``).

    Raises:
        AssertionError: If ``system.service_client`` is ``None``.
    """
    assert system.service_client is not None, "Service client not initialized"
    pose_dict = pose.model_dump()
    response = system.service_client.call('move_to_pose_service', {
        'command': 'MOVE_TO_POSE',
        'robot_id': robot_id,
        'arm': arm_id,
        'pose': pose_dict,
        'gripper_action': gripper_action,
        'num_steps': num_steps
    })
    if response.get('status') == 'success':
        logger.info(f"Move command sent to robot {robot_id}, arm {arm_id}")
    else:
        logger.info(f"Failed to send move command: {response.get('message')}")


def follow_eef_trajectory(system: DualArmManipulationSystem, robot_id: int, arm_id: str, trajectory: List[StampedPose], gripper_action: float = 0.0, num_steps: Optional[int] = None, smoothen: bool = False):
    """Send a non-blocking FOLLOW_TRAJECTORY command and log the result.

    Args:
        system: Running manipulation system with a service client.
        robot_id: Robot index.
        arm_id: Arm identifier.
        trajectory: EEF waypoints.
        gripper_action: Gripper command during motion.
        num_steps: Number of smoothed waypoints (defaults to trajectory length).
        smoothen: Whether to apply quintic smoothing.

    Raises:
        AssertionError: If ``system.service_client`` is ``None``.
    """
    assert system.service_client is not None, "Service client not initialized"
    traj_dicts = [p.model_dump() for p in trajectory]
    response = system.service_client.call('move_to_pose_service', {
        'command': 'FOLLOW_TRAJECTORY',
        'robot_id': robot_id,
        'arm': arm_id,
        'trajectory': traj_dicts,
        'gripper_action': gripper_action,
        'num_steps': num_steps,
        'smoothen': smoothen
    })
    if response.get('status') == 'success':
        logger.info(f"Follow trajectory command sent to robot {robot_id}, arm {arm_id}")
    else:
        logger.info(f"Failed to send follow trajectory command: {response.get('message')}")


def cancel_move(system: DualArmManipulationSystem, robot_id: int, arm_id: str):
    """Send a CANCEL_MOVE command and log the result.

    Args:
        system: Running manipulation system with a service client.
        robot_id: Robot index.
        arm_id: Arm identifier.

    Raises:
        AssertionError: If ``system.service_client`` is ``None``.
    """
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('move_to_pose_service', {
        'command': 'CANCEL_MOVE',
        'robot_id': robot_id,
        'arm': arm_id
    })
    if response.get('status') == 'success':
        logger.info(f"Cancel command sent to robot {robot_id}, arm {arm_id}")
    else:
        logger.info(f"Failed to send cancel command: {response.get('message')}")


def get_arm_status(system: DualArmManipulationSystem, robot_id: int, arm_id: str):
    """Query an arm's motion status.

    Args:
        system: Running manipulation system with a service client.
        robot_id: Robot index.
        arm_id: Arm identifier.

    Returns:
        ``status_code`` (-1 moving, 1 success, 0 failure), or ``None`` on
        failure.

    Raises:
        AssertionError: If ``system.service_client`` is ``None``.
    """
    assert system.service_client is not None, "Service client not initialized"
    response = system.service_client.call('move_to_pose_service', {
        'command': 'GET_ARM_STATUS',
        'robot_id': robot_id,
        'arm': arm_id
    })
    if response.get('status') == 'success':
        return response.get('status_code')
    else:
        logger.info(f"Failed to get arm status: {response.get('message')}")
        return None


# ---------------------------------------------------------------------------
# Trajectory I/O & visualization
# ---------------------------------------------------------------------------

def load_trajectory_file(filepath: str):
    """Load a saved trajectory file.

    Args:
        filepath: Path to the trajectory file.

    Returns:
        Dict with ``eef_traj``, ``obj_traj``, ``joint_configs``,
        ``waypoint_results`` and ``feasibility`` (bool array or ``None``),
        or ``None`` on failure.
    """
    try:
        eef_traj, obj_traj, joint_configs, waypoint_results = load_trajectory(filepath)
        logger.info(f"Loaded trajectory from {filepath}: {len(eef_traj)} waypoints")
        feasibility = np.array([r.is_feasible for r in waypoint_results]) if waypoint_results else None
        return {
            "eef_traj": eef_traj,
            "obj_traj": obj_traj,
            "joint_configs": joint_configs,
            "waypoint_results": waypoint_results,
            "feasibility": feasibility
        }
    except Exception as e:
        logger.info(f"Failed to load trajectory: {e}")
        return None


def visualize_trajectory_data(system: DualArmManipulationSystem, eef_trajectory: List[StampedPose], waypoint_results: Optional[List[WaypointFeasibilityResult]] = None, name: str = "trajectory"):
    """Visualize an EEF trajectory in MeshCat.

    Args:
        system: Manipulation system with a ``tf_manager``.
        eef_trajectory: EEF waypoints.
        waypoint_results: Optional per-waypoint feasibility results for coloring.
        name: MeshCat path name.
    """
    if system.tf_manager is None:
        logger.info("TransformManager not initialized")
        return
    vis = meshcat.Visualizer(zmq_url=str(constants.MESHCAT_ZMQ_URL))
    visualize_trajectory(vis, eef_trajectory, waypoint_results, name=name, tf_manager=system.tf_manager)
    logger.info(f"Trajectory visualization sent to MeshCat as '{name}'")
