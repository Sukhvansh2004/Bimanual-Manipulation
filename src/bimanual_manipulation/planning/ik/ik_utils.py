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
    """Return the enumerate function for the configured IK backend.

    Returns:
        The ``enumerate_ik_solutions_*`` function for
        :data:`constants.IK_SOLVER_BACKEND`.

    Raises:
        ValueError: If the backend is not ``"eaik"``, ``"scipy"`` or
            ``"mink"``.
    """
    backend = constants.IK_SOLVER_BACKEND
    if backend == "eaik":
        from bimanual_manipulation.planning.ik.eaik_solver import (
            enumerate_ik_solutions_eaik,
        )
        return enumerate_ik_solutions_eaik
    elif backend == "scipy":
        from bimanual_manipulation.planning.ik.scipy_solver import (
            enumerate_ik_solutions_scipy,
        )
        return enumerate_ik_solutions_scipy
    elif backend == "mink":
        from bimanual_manipulation.planning.ik.mink_solver import (
            enumerate_ik_solutions_mink,
        )
        return enumerate_ik_solutions_mink
    else:
        raise ValueError(
            f"Unknown IK_SOLVER_BACKEND: {backend!r}. "
            "Expected 'eaik', 'scipy', or 'mink'."
        )


def unique_solutions(solutions: Iterable[np.ndarray], atol: float = constants.IK_UNIQUE_TOLERANCE) -> List[np.ndarray]:
    """Remove near-duplicate joint solutions.

    Solutions are sorted lexicographically and a solution is dropped when
    every joint is within *atol* of its sorted predecessor.

    Args:
        solutions: Joint configurations of equal length.
        atol: Per-joint absolute tolerance for duplicates.

    Returns:
        Unique solutions in lexicographic order (input order is kept when
        there are fewer than two solutions).
    """
    sol_list = list(solutions)
    if len(sol_list) <= 1:
        return sol_list
    sol_arr = np.array(sol_list)
    order = np.lexsort(sol_arr[:, ::-1].T)
    sorted_sols = sol_arr[order]
    diffs = np.abs(sorted_sols[1:] - sorted_sols[:-1])
    is_dup = np.all(diffs <= atol, axis=1)
    mask = np.empty(len(sorted_sols), dtype=bool)
    mask[0] = True
    mask[1:] = ~is_dup
    return [sorted_sols[i] for i in np.flatnonzero(mask)]


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
    """Enumerate IK solutions for a single arm.

    Dispatches to the backend selected by
    :data:`constants.IK_SOLVER_BACKEND` (``"eaik"``, ``"scipy"``, or
    ``"mink"``).

    When *q_reference* is provided, solutions are sorted by joint-space
    proximity to the reference configuration (closest first).  This
    ensures trajectory continuity when solving IK for consecutive
    waypoints.  Otherwise they are ranked by manipulability.

    Args:
        env: The dual-arm simulation environment.
        robot_id: Robot index (0 or 1).
        target_pose: Desired EEF site pose (any frame known to
            *tf_manager*).
        tf_manager: Transform manager for frame conversions.
        arm: Which arm (``"right"`` or ``"left"``).
        site_name: Override for the MuJoCo EEF site name.
        env_lock: Optional lock guarding the environment.
        max_solutions: Maximum number of solutions to return.
        unique_atol: Tolerance for duplicate removal.
        q_reference: Optional reference configuration for ranking.

    Returns:
        List of joint angle arrays of shape ``(7,)`` (may be empty).

    Raises:
        ValueError: If :data:`constants.IK_SOLVER_BACKEND` is unknown.
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

    Delegates to the canonical implementation in
    :func:`bimanual_manipulation.utils.mujoco_utils.compute_arm_jacobian`.

    Args:
        env: The dual-arm simulation environment.
        robot_id: Robot index (0 or 1).
        arm: Which arm (``"right"`` or ``"left"``).
        q_config: Optional arm configuration; the current one is used
            when ``None``.
        site_name: Override for the MuJoCo EEF site name.
        env_lock: Optional lock guarding the environment.

    Returns:
        ``np.ndarray`` of shape ``(6, 7)``.
    """
    from bimanual_manipulation.utils.mujoco_utils import (
        compute_arm_jacobian as _compute_arm_jacobian,
    )
    return _compute_arm_jacobian(
        env,
        robot_id=robot_id,
        arm=arm,
        q_config=q_config,
        site_name=site_name,
        env_lock=env_lock,
    )


# Pre-built constant rotations (avoids R.from_euler on every call)
_R_CORR_NEG90Z = R.from_euler('z', -90, degrees=True)
_R_CORR_POS90Z = R.from_euler('z', 90, degrees=True)


def apply_ik_orientation_correction(pose: StampedPose) -> StampedPose:
    """Apply -90 degree Z rotation correction to align grasp with MuJoCo site frame.

    Converts a pose from body-frame convention to site-frame convention
    by right-multiplying its orientation by ``Rz(-90°)``.  Used before IK
    solving.

    Args:
        pose: Pose in body-frame convention.

    Returns:
        A corrected copy of *pose* (position and frame unchanged).
    """
    corrected = pose.copy()
    r_pose = R.from_quat(corrected.quaternion)
    corrected.quaternion = (r_pose * _R_CORR_NEG90Z).as_quat()
    return corrected


def reverse_ik_orientation_correction(pose: StampedPose) -> StampedPose:
    """Apply +90 degree Z rotation to convert from site-frame to body-frame convention.

    This is the inverse of apply_ik_orientation_correction.
    Used when converting stored site-frame EEF poses back to body-frame
    for use with move_to_pose / follow_trajectory.

    Args:
        pose: Pose in site-frame convention.

    Returns:
        A corrected copy of *pose* (position and frame unchanged).
    """
    corrected = pose.copy()
    r_pose = R.from_quat(corrected.quaternion)
    corrected.quaternion = (r_pose * _R_CORR_POS90Z).as_quat()
    return corrected
