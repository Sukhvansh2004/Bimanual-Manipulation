"""Inverse Kinematics (IK) utilities.

Provides IK solution enumeration and orientation-correction helpers
for single-arm manipulation.

Offline IK enumeration delegates to EAIK (analytical closed-form) via
:func:`enumerate_ik_solutions_eaik` for fast solving.
"""

import threading
from typing import List, Optional, Iterable

import numpy as np
import logging
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.transforms import StampedPose, TransformManager
from bimanual_manipulation.planning.utils.eaik_solver import (
    enumerate_ik_solutions_eaik,
)

logger = logging.getLogger(__name__)


def unique_solutions(solutions: Iterable[np.ndarray], atol: float = constants.IK_UNIQUE_TOLERANCE) -> List[np.ndarray]:
    uniq: List[np.ndarray] = []
    for q in solutions:
        if not any(np.allclose(q, u, atol=atol, rtol=0.0) for u in uniq):
            uniq.append(q)
    return uniq


def enumerate_ik_solutions_for_arm(
    env: DualArmEnvironment,
    *,
    robot_id: int,
    target_pose: StampedPose,
    tf_manager: TransformManager,
    arm: str = "right",
    site_name: Optional[str] = None,
    env_lock: threading.RLock = None,
    max_solutions: int = constants.IK_MAX_SOLUTIONS,
    unique_atol: float = constants.IK_UNIQUE_TOLERANCE,
    q_reference: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    f"""Enumerate IK solutions for a single arm using EAIK analytical solver.

    When *q_reference* is provided, solutions are sorted by joint-space
    proximity to the reference configuration (closest first).  This
    ensures trajectory continuity when solving IK for consecutive
    waypoints.

    Returns:
        List of joint angle arrays of shape ({constants.FRANKA_ARM_DOF},).
    """
    return enumerate_ik_solutions_eaik(
        env,
        robot_id=robot_id,
        target_pose=target_pose,
        tf_manager=tf_manager,
        arm=arm,
        site_name=site_name,
        env_lock=env_lock,
        max_solutions=max_solutions,
        unique_atol=unique_atol,
        q_reference=q_reference,
    )


def apply_ik_orientation_correction(pose: StampedPose) -> StampedPose:
    """Apply -90 degree Z rotation correction to align grasp with MuJoCo site frame.

    Converts a pose from body-frame convention to site-frame convention.
    Used before IK solving.
    """
    corrected = pose.copy()
    r_corr = R.from_euler('z', -90, degrees=True)
    r_pose = R.from_quat(corrected.quaternion)
    corrected.quaternion = (r_pose * r_corr).as_quat()
    return corrected


def reverse_ik_orientation_correction(pose: StampedPose) -> StampedPose:
    """Apply +90 degree Z rotation to convert from site-frame to body-frame convention.

    This is the inverse of apply_ik_orientation_correction.
    Used when converting stored site-frame EEF poses back to body-frame
    for use with move_to_pose / follow_trajectory.
    """
    corrected = pose.copy()
    r_corr = R.from_euler('z', 90, degrees=True)
    r_pose = R.from_quat(corrected.quaternion)
    corrected.quaternion = (r_pose * r_corr).as_quat()
    return corrected
