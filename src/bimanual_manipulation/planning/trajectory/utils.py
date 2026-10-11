"""Trajectory I/O and data types for trajectory feasibility.

Provides :class:`WaypointFeasibilityResult` and save/load helpers for
persisting end-effector and object trajectories to ``.npz`` files.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.transforms import StampedPose

logger = logging.getLogger(__name__)


@dataclass
class WaypointFeasibilityResult:
    """Result of feasibility check for a single trajectory waypoint.

    Attributes:
        waypoint_index: Index of the waypoint in the trajectory.
        eef_pose: End-effector pose at this waypoint.
        joint_config: IK joint configuration, or ``None`` if IK failed.
        is_feasible: ``True`` if IK succeeded and no collision was found.
        ik_success: ``True`` if IK found a solution.
        collision_free: ``True`` if no collision was detected.
    """
    waypoint_index: int
    eef_pose: StampedPose
    joint_config: Optional[np.ndarray]  # None if IK failed
    is_feasible: bool  # True if IK succeeded and no collision
    ik_success: bool  # True if IK found a solution
    collision_free: bool  # True if no collision detected


def save_trajectory(
    output_dir: str, 
    filename: str,
    eef_trajectory: Optional[List[StampedPose]] = None,
    object_trajectory: Optional[List[StampedPose]] = None,
    waypoint_results: Optional[List[WaypointFeasibilityResult]] = None,
) -> None:
    """Save trajectory to npz file with optional feasibility results.

    Creates *output_dir* if needed. Stored keys (each only when the
    corresponding input is non-empty): ``eef_positions``,
    ``eef_quaternions``, ``eef_frame_ids``, ``object_positions``,
    ``object_quaternions``, ``object_frame_ids``, ``is_feasible``,
    ``ik_success``, ``collision_free`` (``int32`` 0/1) and
    ``joint_configs`` (NaN rows of length
    :data:`constants.FRANKA_ARM_DOF` where IK failed). If
    *eef_trajectory* is empty, EEF poses are taken from
    *waypoint_results*.

    Args:
        output_dir: Output directory.
        filename: Output filename (``np.savez`` appends ``.npz`` if missing).
        eef_trajectory: List of end-effector StampedPose objects (optional
            if waypoint_results provided).
        object_trajectory: List of object StampedPose objects (optional).
        waypoint_results: List of WaypointFeasibilityResult with joint
            configs and feasibility (optional).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    
    data_dict = {}
    
    # Save EEF trajectory poses
    if eef_trajectory:
        data_dict['eef_positions'] = np.array([p.position for p in eef_trajectory])
        data_dict['eef_quaternions'] = np.array([p.quaternion for p in eef_trajectory])
        data_dict['eef_frame_ids'] = np.array([p.frame_id for p in eef_trajectory])
    
    # Save object trajectory poses
    if object_trajectory:
        data_dict['object_positions'] = np.array([p.position for p in object_trajectory])
        data_dict['object_quaternions'] = np.array([p.quaternion for p in object_trajectory])
        data_dict['object_frame_ids'] = np.array([p.frame_id for p in object_trajectory])
    
    # Save waypoint results with feasibility info
    if waypoint_results:
        # Feasibility flags (1 = feasible, 0 = infeasible)
        data_dict['is_feasible'] = np.array([int(r.is_feasible) for r in waypoint_results], dtype=np.int32)
        data_dict['ik_success'] = np.array([int(r.ik_success) for r in waypoint_results], dtype=np.int32)
        data_dict['collision_free'] = np.array([int(r.collision_free) for r in waypoint_results], dtype=np.int32)
        
        # Joint configurations (NaN for failed IK)
        joint_configs = []
        for r in waypoint_results:
            if r.joint_config is not None:
                joint_configs.append(r.joint_config)
            else:
                # Use NaN array for failed IK
                joint_configs.append(np.full(constants.FRANKA_ARM_DOF, np.nan))
        data_dict['joint_configs'] = np.array(joint_configs)
        
        # EEF poses from results (in case eef_trajectory not separately provided)
        if not eef_trajectory:
            data_dict['eef_positions'] = np.array([r.eef_pose.position for r in waypoint_results])
            data_dict['eef_quaternions'] = np.array([r.eef_pose.quaternion for r in waypoint_results])
            data_dict['eef_frame_ids'] = np.array([r.eef_pose.frame_id for r in waypoint_results])
    
    # Save as npz
    np.savez(filepath, **data_dict)
    logger.info(f"Trajectory saved to {filepath}")

def load_trajectory(
    filepath: str,
) -> Tuple[List[StampedPose], Optional[List[StampedPose]], Optional[np.ndarray], Optional[List[WaypointFeasibilityResult]]]:
    """Load trajectory from npz file written by :func:`save_trajectory`.

    Missing frame IDs default to :data:`constants.WORLD_FRAME`; missing
    ``ik_success`` / ``collision_free`` flags default to ``True``.

    Args:
        filepath: Path to the npz file.

    Returns:
        Tuple of ``(eef_trajectory, object_trajectory, joint_configs,
        waypoint_results)``:

        - eef_trajectory: List of StampedPose (empty if not stored).
        - object_trajectory: List of StampedPose, or ``None`` if not stored.
        - joint_configs: ``(N, DOF)`` array, or ``None`` if not stored.
        - waypoint_results: List of WaypointFeasibilityResult (one per
          EEF pose), or ``None`` if no ``is_feasible`` key is stored.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")
        
    data = np.load(filepath, allow_pickle=True)
    
    eef_trajectory = []
    if 'eef_positions' in data and 'eef_quaternions' in data:
        positions = data['eef_positions']
        quaternions = data['eef_quaternions']
        frame_ids = data['eef_frame_ids'] if 'eef_frame_ids' in data else [constants.WORLD_FRAME] * len(positions)
        
        for p, q, f in zip(positions, quaternions, frame_ids):
            eef_trajectory.append(StampedPose(position=p, quaternion=q, frame_id=str(f)))
            
    object_trajectory = []
    if 'object_positions' in data and 'object_quaternions' in data:
        positions = data['object_positions']
        quaternions = data['object_quaternions']
        frame_ids = data['object_frame_ids'] if 'object_frame_ids' in data else [constants.WORLD_FRAME] * len(positions)
        
        for p, q, f in zip(positions, quaternions, frame_ids):
            object_trajectory.append(StampedPose(position=p, quaternion=q, frame_id=str(f)))
    else:
        object_trajectory = None
        
    joint_configs = data['joint_configs'] if 'joint_configs' in data else None
    
    waypoint_results = None
    if 'is_feasible' in data:
        waypoint_results = []
        for i in range(len(eef_trajectory)):
            res = WaypointFeasibilityResult(
                waypoint_index=i,
                eef_pose=eef_trajectory[i],
                joint_config=joint_configs[i] if joint_configs is not None else None,
                is_feasible=bool(data['is_feasible'][i]),
                ik_success=bool(data['ik_success'][i]) if 'ik_success' in data else True,
                collision_free=bool(data['collision_free'][i]) if 'collision_free' in data else True
            )
            waypoint_results.append(res)
    
    return eef_trajectory, object_trajectory, joint_configs, waypoint_results
