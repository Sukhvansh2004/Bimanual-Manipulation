"""Common utilities and data structures for planning checks."""

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np

from robosuite_franka_manipulation.transforms import StampedPose

ArrayLike = Union[Sequence[float], np.ndarray]

@dataclass(frozen=True)
class DualArmFeasibilityResult:
    joint_angles_arm0: np.ndarray
    joint_angles_arm1: np.ndarray
    feasible: bool


def as_stamped_pose(pose: Union[StampedPose, dict, Tuple[ArrayLike, ArrayLike], ArrayLike], frame_id: str) -> StampedPose:
    """Normalize pose input into a StampedPose.

    Supported inputs:
    - StampedPose
    - dict with keys {'pos'/'position', 'quat'/'quaternion'}
    - ((pos, quat)) tuple
    - flat length-7 (pos[3] + quat[4]) array-like
    """
    if isinstance(pose, StampedPose):
        return pose

    if isinstance(pose, dict):
        pos = pose.get("pos", pose.get("position"))
        quat = pose.get("quat", pose.get("quaternion"))
        if pos is None or quat is None:
            raise ValueError("Pose dict must contain 'pos'/'position' and 'quat'/'quaternion'.")
        return StampedPose(position=np.asarray(pos, dtype=float), quaternion=np.asarray(quat, dtype=float), frame_id=frame_id)

    if isinstance(pose, tuple) and len(pose) == 2:
        pos, quat = pose
        return StampedPose(position=np.asarray(pos, dtype=float), quaternion=np.asarray(quat, dtype=float), frame_id=frame_id)

    arr = np.asarray(pose, dtype=float).reshape(-1)
    if arr.size != 7:
        raise ValueError("Pose must be a StampedPose, dict, (pos,quat) tuple, or a flat length-7 array [x,y,z,qx,qy,qz,qw].")
    return StampedPose(position=arr[:3], quaternion=arr[3:], frame_id=frame_id)
