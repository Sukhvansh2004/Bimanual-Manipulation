"""Analytical IK solver using EAIK (Efficient Analytical Inverse Kinematics).

Provides closed-form IK solutions for the 7-DOF Franka Panda by sweeping
a locked joint across its range and collecting all analytical solutions.
This replaces the iterative QP-based ``solve_ik_mink`` with a ~1000×
faster analytical approach for offline IK enumeration.

The 7-DOF Panda is reduced to 6-DOF by locking one joint at a time.
Joint 4 (0-indexed=3 in EAIK) produces 8 exact solutions per locked
value and is the default sweep joint.
"""

import logging
from functools import lru_cache
from pathlib import Path
import threading
from contextlib import nullcontext
from typing import List, Optional, Tuple

import mujoco
import numpy as np
from eaik.IK_URDF import UrdfRobot
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.planning.utils.manipulability import (
    compute_manipulability_from_jacobian,
)
from bimanual_manipulation.planning.utils import mujoco_utils
from bimanual_manipulation.transforms import StampedPose, TransformManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EAIK constants — imported from central config
# ---------------------------------------------------------------------------
from bimanual_manipulation.constants import (
    EAIK_SWEEP_JOINT_INDEX,
    EAIK_SWEEP_JOINT_ROBOT_INDEX,
    EAIK_NUM_SWEEP_VALUES,
    EAIK_MIN_MANIPULABILITY,
)

# Fallback joints to try if primary sweep joint fails
EAIK_FALLBACK_JOINTS = [5, 3]  # EAIK indices for joints 6, 4


def _get_urdf_path() -> str:
    """Locate the Franka Panda URDF shipped with robosuite."""
    import robosuite
    base = Path(robosuite.__file__).parent
    urdf = base / "models" / "assets" / "bullet_data" / "panda_description" / "urdf" / "panda_arm.urdf"
    if not urdf.exists():
        raise FileNotFoundError(
            f"Panda URDF not found at {urdf}. "
            "Ensure robosuite is installed with the bullet_data assets."
        )
    return str(urdf)


@lru_cache(maxsize=64)
def _build_eaik_robot(locked_joint: int, locked_value: float) -> Optional[UrdfRobot]:
    """Build an EAIK robot with a specific joint locked at a given value.

    Results are cached so repeated sweeps with the same joint/value
    reuse the C++ robot object.
    """
    urdf_path = _get_urdf_path()
    try:
        return UrdfRobot(urdf_path, fixed_axes=[(locked_joint, locked_value)])
    except Exception:
        logger.debug("EAIK: cannot lock joint %d at %.4f", locked_joint, locked_value)
        return None


def _compute_jacobian_mujoco(
    model_native: mujoco.MjModel,
    data_native: mujoco.MjData,
    site_id: int,
    arm_dof_idx: np.ndarray,
    arm_qpos_idx: np.ndarray,
    q_arm: np.ndarray,
    full_qpos: np.ndarray,
) -> np.ndarray:
    """Compute 6×7 geometric Jacobian using raw MuJoCo (no mink).

    Returns the Jacobian in the world frame, stacked as
    ``[J_translation; J_rotation]`` of shape ``(6, 7)``.
    """
    q_full = full_qpos.copy()
    q_full[arm_qpos_idx] = q_arm

    data_native.qpos[:] = q_full
    mujoco.mj_fwdPosition(model_native, data_native)

    jacp = np.zeros((3, model_native.nv))
    jacr = np.zeros((3, model_native.nv))
    mujoco.mj_jacSite(model_native, data_native, jacp, jacr, site_id)

    # Slice to arm DOFs only → (6, 7)
    return np.vstack([jacp[:, arm_dof_idx], jacr[:, arm_dof_idx]])


def solve_ik_eaik(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    joint_limits_lo: np.ndarray,
    joint_limits_hi: np.ndarray,
    *,
    sweep_joint: int = EAIK_SWEEP_JOINT_INDEX,
    num_sweep: int = EAIK_NUM_SWEEP_VALUES,
    sweep_range: Optional[Tuple[float, float]] = None,
) -> List[np.ndarray]:
    """Solve IK analytically using EAIK with joint sweeping.

    Sweeps the locked joint across its range, collecting all closed-form
    solutions. Solutions outside joint limits are discarded.

    Args:
        target_pos: Desired EEF position in world frame ``(3,)``.
        target_quat: Desired EEF quaternion ``(x, y, z, w)`` in world frame.
        joint_limits_lo: Lower joint limits ``(7,)``.
        joint_limits_hi: Upper joint limits ``(7,)``.
        sweep_joint: EAIK joint index to lock and sweep (default: 3 = Panda joint 4).
        num_sweep: Number of discrete values for the swept joint.
        sweep_range: ``(lo, hi)`` range for the swept joint. If None, uses
            the joint limits of the corresponding robot joint.

    Returns:
        List of valid joint configurations ``(7,)``, unsorted.
    """
    # Build 4×4 homogeneous target
    rot_mat = R.from_quat(target_quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = target_pos

    # Determine sweep range from joint limits
    robot_joint_idx = EAIK_SWEEP_JOINT_ROBOT_INDEX
    if sweep_range is None:
        lo = float(joint_limits_lo[robot_joint_idx])
        hi = float(joint_limits_hi[robot_joint_idx])
    else:
        lo, hi = sweep_range

    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        logger.warning("EAIK: invalid sweep range [%.3f, %.3f]", lo, hi)
        return []

    sweep_values = np.linspace(lo, hi, num_sweep)
    all_solutions: List[np.ndarray] = []

    for val in sweep_values:
        bot = _build_eaik_robot(sweep_joint, round(float(val), 6))
        if bot is None:
            continue

        try:
            result = bot.IK(T)
        except Exception:
            continue

        for q_sol, is_ls in zip(result.Q, result.is_LS):
            # Prefer exact solutions; accept LS only if no exact ones found
            if is_ls:
                continue

            q = np.asarray(q_sol, dtype=float)
            if q.shape[0] != 7:
                continue

            # Enforce joint limits
            if np.any(q < joint_limits_lo - 1e-6) or np.any(q > joint_limits_hi + 1e-6):
                continue

            # Clamp to exact limits
            q = np.clip(q, joint_limits_lo, joint_limits_hi)
            all_solutions.append(q)

    # If no exact solutions found, accept least-squares solutions
    if not all_solutions:
        for val in sweep_values:
            bot = _build_eaik_robot(sweep_joint, round(float(val), 6))
            if bot is None:
                continue
            try:
                result = bot.IK(T)
            except Exception:
                continue
            for q_sol, is_ls in zip(result.Q, result.is_LS):
                q = np.asarray(q_sol, dtype=float)
                if q.shape[0] != 7:
                    continue
                if np.any(q < joint_limits_lo - 1e-6) or np.any(q > joint_limits_hi + 1e-6):
                    continue
                q = np.clip(q, joint_limits_lo, joint_limits_hi)
                all_solutions.append(q)

    return all_solutions


def _filter_and_rank_solutions(
    solutions: List[np.ndarray],
    model_native: mujoco.MjModel,
    data_native: mujoco.MjData,
    site_id: int,
    arm_dof_idx: np.ndarray,
    arm_qpos_idx: np.ndarray,
    full_qpos: np.ndarray,
    *,
    unique_atol: float = constants.IK_UNIQUE_TOLERANCE,
    min_manip: float = constants.IK_MIN_MANIPULABILITY,
    max_solutions: int = constants.IK_MAX_SOLUTIONS,
) -> List[np.ndarray]:
    """De-duplicate, filter by manipulability, and rank solutions.

    Args:
        solutions: Raw solutions from ``solve_ik_eaik``.
        model_native: Native MuJoCo model.
        data_native: Native MuJoCo data (will be modified in-place).
        site_id: MuJoCo site ID for Jacobian computation.
        arm_dof_idx: DOF indices for the arm ``(7,)``.
        arm_qpos_idx: qpos indices for the arm ``(7,)``.
        full_qpos: Full model qpos vector ``(nq,)``.
        unique_atol: Tolerance for de-duplication.
        min_manip: Minimum manipulability threshold.
        max_solutions: Maximum number of solutions to return.

    Returns:
        Ranked list of joint configurations, best manipulability first.
    """
    if not solutions:
        return []

    # De-duplicate
    unique: List[np.ndarray] = []
    for q in solutions:
        if not any(np.allclose(q, u, atol=unique_atol, rtol=0.0) for u in unique):
            unique.append(q)

    # Score by manipulability
    scored: List[Tuple[float, np.ndarray]] = []
    for q in unique:
        J = _compute_jacobian_mujoco(
            model_native, data_native, site_id,
            arm_dof_idx, arm_qpos_idx, q, full_qpos,
        )
        m = compute_manipulability_from_jacobian(J)
        if m < min_manip:
            logger.debug("EAIK: solution rejected, manipulability %.4e < %.2e", m, min_manip)
            continue
        scored.append((m, q))

        if len(scored) >= max_solutions * 2:
            break

    # Sort by manipulability (best first) and trim
    scored.sort(key=lambda x: x[0], reverse=True)
    result = [q for _, q in scored[:max_solutions]]

    if scored:
        logger.debug(
            "EAIK: %d solutions ranked by manipulability (best=%.4e, worst=%.4e)",
            len(result), scored[0][0], scored[min(len(scored) - 1, max_solutions - 1)][0],
        )

    return result


def enumerate_ik_solutions_eaik(
    env: DualArmEnvironment,
    *,
    robot_id: int,
    target_pose: StampedPose,
    tf_manager: TransformManager,
    arm: str = "right",
    site_name: Optional[str] = None,
    env_lock: Optional[threading.RLock] = None,
    num_sweep: int = EAIK_NUM_SWEEP_VALUES,
    max_solutions: int = constants.IK_MAX_SOLUTIONS,
    unique_atol: float = constants.IK_UNIQUE_TOLERANCE,
    q_reference: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Enumerate IK solutions using EAIK analytical solver.

    Drop-in replacement for ``enumerate_ik_solutions_for_arm`` that uses
    closed-form IK instead of iterative QP, giving ~1000× speedup.

    Args:
        env: The dual-arm simulation environment.
        robot_id: Robot index (0 or 1).
        target_pose: Desired EEF pose (any frame).
        tf_manager: For frame conversions.
        arm: Which arm (``"right"`` or ``"left"``).
        site_name: Override for the MuJoCo site name.
        env_lock: Optional threading lock for environment access.
        num_sweep: Number of swept-joint values.
        max_solutions: Maximum solutions to return.
        unique_atol: Tolerance for uniqueness filtering.
        q_reference: Optional reference joint configuration for
            warm-starting.  When provided, the returned solutions are
            re-sorted by L2 distance to *q_reference* (closest first)
            so that consecutive waypoint IK solutions stay continuous.

    Returns:
        List of joint configurations ``(7,)``, ranked by proximity to
        *q_reference* if given, otherwise by manipulability.
    """

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        if len(joint_ids) != constants.FRANKA_ARM_DOF:
            raise RuntimeError(
                f"Expected {constants.FRANKA_ARM_DOF} arm joints for "
                f"robot{robot_id}, found {len(joint_ids)}."
            )

        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        joint_dof_idx = mujoco_utils.get_joint_dof_indices(model, joint_ids)
        joint_lo, joint_hi = mujoco_utils.get_joint_limits(model, joint_ids)

        # Transform target to world frame (MuJoCo sites are in world)
        pose = target_pose
        if pose.frame_id != constants.WORLD_FRAME:
            pose = tf_manager.transform_pose(pose, constants.WORLD_FRAME)

        target_pos = np.asarray(pose.position, dtype=float)
        target_quat = np.asarray(pose.quaternion, dtype=float)

        if site_name is None:
            site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)

        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        # Get site ID for Jacobian computation
        site_id = mujoco.mj_name2id(model_native, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise RuntimeError(f"MuJoCo site '{site_name}' not found in model")

        full_qpos = np.array(data.qpos, copy=True, dtype=float).ravel()

        # Save and restore simulation state
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            # Get all analytical solutions via joint sweep
            raw_solutions = solve_ik_eaik(
                target_pos, target_quat,
                joint_lo, joint_hi,
                num_sweep=num_sweep,
            )

            logger.debug(
                "EAIK: %d raw solutions for robot%d at %s",
                len(raw_solutions), robot_id, target_pos,
            )

            # Filter, de-duplicate, and rank by manipulability
            solutions = _filter_and_rank_solutions(
                raw_solutions,
                model_native, data_native, site_id,
                joint_dof_idx, joint_qpos_idx, full_qpos,
                unique_atol=unique_atol,
                max_solutions=max_solutions,
            )

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_fwdPosition(model_native, data_native)

        # Re-sort by proximity to reference config for trajectory continuity
        if q_reference is not None and len(solutions) > 1:
            solutions.sort(key=lambda q: float(np.linalg.norm(q - q_reference)))

        return solutions


def compute_arm_jacobian(
    env: DualArmEnvironment,
    *,
    robot_id: int,
    arm: str = "right",
    q_config: Optional[np.ndarray] = None,
    site_name: Optional[str] = None,
    env_lock=None,
) -> np.ndarray:
    """Compute the 6×7 geometric Jacobian at the EEF site using raw MuJoCo.

    Drop-in replacement for the mink-based ``compute_arm_jacobian`` that
    avoids creating a ``mink.Configuration`` (saves ~0.6ms per call).

    The Jacobian maps 7-DOF joint velocities to 6-D spatial velocities
    ``[v_x, v_y, v_z, ω_x, ω_y, ω_z]`` in the **world frame**.

    Args:
        env: The dual-arm simulation environment.
        robot_id: Index of the robot (0 or 1).
        arm: Which arm (``"right"`` or ``"left"``).
        q_config: Joint configuration ``(7,)`` to evaluate at.
            If *None*, uses the current simulation state.
        site_name: Override for the MuJoCo site name.
        env_lock: Optional threading lock.

    Returns:
        ``np.ndarray`` of shape ``(6, 7)``.
    """
    from contextlib import nullcontext

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        if len(joint_ids) != constants.FRANKA_ARM_DOF:
            raise RuntimeError(
                f"Expected {constants.FRANKA_ARM_DOF} arm joints for "
                f"robot{robot_id}, found {len(joint_ids)}."
            )

        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        joint_dof_idx = mujoco_utils.get_joint_dof_indices(model, joint_ids)

        if site_name is None:
            site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)

        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        site_id = mujoco.mj_name2id(model_native, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise RuntimeError(f"MuJoCo site '{site_name}' not found in model")

        full_qpos = np.array(data.qpos, copy=True, dtype=float).ravel()

        if q_config is not None:
            q_arm = np.asarray(q_config, dtype=float)
        else:
            q_arm = full_qpos[joint_qpos_idx]

        return _compute_jacobian_mujoco(
            model_native, data_native, site_id,
            joint_dof_idx, joint_qpos_idx, q_arm, full_qpos,
        )
