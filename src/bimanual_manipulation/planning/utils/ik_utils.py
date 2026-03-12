"""Inverse Kinematics (IK) utilities.

Provides IK solution enumeration, Jacobian computation, and
orientation-correction helpers for single-arm manipulation.

The IK backend is selected by :data:`constants.IK_SOLVER_BACKEND`:

* ``"eaik"`` — Analytical closed-form via EAIK (fastest, default).
* ``"scipy"`` — Numerical optimisation via L-BFGS-B.
* ``"mink"`` — QP-based iterative differential IK.
"""

import logging
import threading
from typing import Callable, Dict, List, Optional, Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.transforms import StampedPose, TransformManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend registry (lazy imports to avoid hard dependencies for unused
# backends)
# ---------------------------------------------------------------------------

def _get_enumerate_fn() -> Callable:
    """Return the enumerate function for the configured IK backend."""
    backend = constants.IK_SOLVER_BACKEND
    if backend == "eaik":
        from bimanual_manipulation.planning.utils.eaik_solver import (
            enumerate_ik_solutions_eaik,
        )
        return enumerate_ik_solutions_eaik
    elif backend == "scipy":
        from bimanual_manipulation.planning.utils.scipy_solver import (
            enumerate_ik_solutions_scipy,
        )
        return enumerate_ik_solutions_scipy
    elif backend == "mink":
        from bimanual_manipulation.planning.utils.mink_solver import (
            enumerate_ik_solutions_mink,
        )
        return enumerate_ik_solutions_mink
    else:
        raise ValueError(
            f"Unknown IK_SOLVER_BACKEND: {backend!r}. "
            "Expected 'eaik', 'scipy', or 'mink'."
        )


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
    f"""Enumerate IK solutions for a single arm.

    Dispatches to the backend selected by
    :data:`constants.IK_SOLVER_BACKEND` (``"eaik"``, ``"scipy"``, or
    ``"mink"``).

    When *q_reference* is provided, solutions are sorted by joint-space
    proximity to the reference configuration (closest first).  This
    ensures trajectory continuity when solving IK for consecutive
    waypoints.

    Returns:
        List of joint angle arrays of shape ({constants.FRANKA_ARM_DOF},).
    """
    enumerate_fn = _get_enumerate_fn()
    return enumerate_fn(
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


def _get_jacobian_fn() -> Callable:
    """Return the Jacobian function for the configured IK backend."""
    backend = constants.IK_SOLVER_BACKEND
    if backend == "eaik":
        from bimanual_manipulation.planning.utils.eaik_solver import (
            compute_arm_jacobian as _jac,
        )
        return _jac
    elif backend == "scipy":
        from bimanual_manipulation.planning.utils.scipy_solver import (
            compute_arm_jacobian as _jac,
        )
        return _jac
    elif backend == "mink":
        from bimanual_manipulation.planning.utils.mink_solver import (
            compute_arm_jacobian as _jac,
        )
        return _jac
    else:
        raise ValueError(
            f"Unknown IK_SOLVER_BACKEND: {backend!r}. "
            "Expected 'eaik', 'scipy', or 'mink'."
        )


def compute_arm_jacobian(
    env: DualArmEnvironment,
    *,
    robot_id: int,
    arm: str = "right",
    q_config: Optional[np.ndarray] = None,
    site_name: Optional[str] = None,
    env_lock=None,
) -> np.ndarray:
    """Compute the 6x7 geometric Jacobian at the EEF site.

    Dispatches to the backend selected by
    :data:`constants.IK_SOLVER_BACKEND`.

    Returns:
        ``np.ndarray`` of shape ``(6, 7)``.
    """
    jac_fn = _get_jacobian_fn()
    return jac_fn(
        env,
        robot_id=robot_id,
        arm=arm,
        q_config=q_config,
        site_name=site_name,
        env_lock=env_lock,
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
