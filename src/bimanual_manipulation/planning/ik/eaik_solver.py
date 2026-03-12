"""Analytical IK solver using EAIK (Efficient Analytical Inverse Kinematics).

Provides closed-form IK solutions for the 7-DOF Franka Panda by sweeping
all joints in turn: each one is locked at discrete values while the
remaining 6-DOF system is solved analytically.  This explores the full
redundancy manifold and replaces the iterative QP-based ``solve_ik_mink``
with a ~1000× faster analytical approach.

When a reference configuration (*q_reference*) is supplied, each joint's
sweep is centred on the reference value within a configurable window,
biasing solutions toward the previous configuration for trajectory
continuity (warm-starting).
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
from bimanual_manipulation.planning.ik.manipulability import (
    compute_manipulability_from_jacobian,
)
from bimanual_manipulation.planning.utils import mujoco_utils
from bimanual_manipulation.transforms import StampedPose, TransformManager

logger = logging.getLogger(__name__)


def _get_urdf_path() -> str:
    """Return the Franka Panda URDF path from constants."""
    urdf = constants.PANDA_URDF_PATH
    if not urdf.exists():
        raise FileNotFoundError(
            f"Panda URDF not found at {urdf}. "
            "Ensure robosuite is installed with the bullet_data assets."
        )
    return str(urdf)


@lru_cache(maxsize=1024)
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


def _sweep_values_for_joint(
    joint_idx: int,
    joint_limits_lo: np.ndarray,
    joint_limits_hi: np.ndarray,
    num_sweep: int,
    q_reference: Optional[np.ndarray],
) -> np.ndarray:
    """Compute sweep values for a single joint.

    When *q_reference* is provided the sweep is centred on
    ``q_reference[joint_idx]`` within a ``±EAIK_WARM_START_WINDOW``
    window (clipped to joint limits), and the reference value itself
    is always included.  Otherwise a uniform sweep across the full
    joint range is used.
    """
    lo = float(joint_limits_lo[joint_idx])
    hi = float(joint_limits_hi[joint_idx])
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return np.array([])

    if q_reference is not None:
        centre = float(np.clip(q_reference[joint_idx], lo, hi))
        half = constants.EAIK_WARM_START_WINDOW
        w_lo = max(lo, centre - half)
        w_hi = min(hi, centre + half)
        vals = np.linspace(w_lo, w_hi, num_sweep)
        # Ensure the reference value itself is included
        if not np.any(np.isclose(vals, centre, atol=1e-8)):
            vals = np.sort(np.append(vals, centre))
    else:
        vals = np.linspace(lo, hi, num_sweep)

    # Snap to a fixed grid so UrdfRobot objects are reused across
    # different q_reference values (avoids expensive URDF re-parsing).
    step = constants.EAIK_GRID_STEP
    vals = np.unique(np.clip(np.round(vals / step) * step, lo, hi))
    return vals


def solve_ik_eaik(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    joint_limits_lo: np.ndarray,
    joint_limits_hi: np.ndarray,
    *,
    num_sweep: int = constants.EAIK_NUM_SWEEP_VALUES,
    q_reference: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Solve IK analytically using EAIK with joint sweeping.

    All 7 joints are swept in turn (each one locked while the remaining
    6-DOF system is solved analytically).  This explores the robot's
    full redundancy manifold and produces a much richer solution set
    than sweeping a single joint.

    When *q_reference* is provided the sweep for each joint is
    concentrated near the reference value, biasing solutions toward
    the previous configuration and maintaining trajectory continuity.

    Args:
        target_pos: Desired EEF position in robot base frame ``(3,)``.
        target_quat: Desired EEF quaternion ``(x, y, z, w)`` in robot base frame.
        joint_limits_lo: Lower joint limits ``(7,)``.
        joint_limits_hi: Upper joint limits ``(7,)``.
        num_sweep: Number of discrete values per joint.
        q_reference: Optional reference configuration for warm-start.
            When given, each joint's sweep is centred on the reference
            value within a ``±EAIK_WARM_START_WINDOW`` window.

    Returns:
        List of valid joint configurations ``(7,)``, unsorted.
    """
    # Build 4×4 homogeneous target
    rot_mat = R.from_quat(target_quat).as_matrix()
    T_target = np.eye(4)
    T_target[:3, :3] = rot_mat
    T_target[:3, 3] = target_pos

    n_joints = len(joint_limits_lo)
    all_solutions: List[np.ndarray] = []

    for accept_ls in (False, True):
        for jidx in range(n_joints):
            sweep_vals = _sweep_values_for_joint(
                jidx, joint_limits_lo, joint_limits_hi, num_sweep, q_reference,
            )
            for val in sweep_vals:
                bot = _build_eaik_robot(jidx, round(float(val), 6))
                if bot is None:
                    continue

                try:
                    result = bot.IK(T_target)
                except Exception:
                    continue

                for q_sol, is_ls in zip(result.Q, result.is_LS):
                    if is_ls and not accept_ls:
                        continue

                    q = np.asarray(q_sol, dtype=float)
                    if q.shape[0] != 7:
                        continue
                    if np.any(q < joint_limits_lo - 1e-6) or np.any(
                        q > joint_limits_hi + 1e-6
                    ):
                        continue

                    q = np.clip(q, joint_limits_lo, joint_limits_hi)
                    all_solutions.append(q)

        # Use exact solutions if we found any; only fall back to LS if needed
        if all_solutions:
            break

    return all_solutions


def _fk_error(
    data_native: mujoco.MjData,
    site_id: int,
    target_pos_world: np.ndarray,
    target_quat_world: np.ndarray,
) -> Tuple[float, float]:
    """Compute position and orientation FK error after ``mj_fwdPosition``.

    Returns:
        ``(pos_err, rot_err)`` — Euclidean position error (m) and
        geodesic rotation error (rad).
    """
    achieved_pos = np.asarray(data_native.site_xpos[site_id], dtype=float)
    achieved_mat = np.asarray(data_native.site_xmat[site_id], dtype=float).reshape(3, 3)
    pos_err = float(np.linalg.norm(achieved_pos - target_pos_world))
    rot_err = float(
        np.linalg.norm(
            (R.from_quat(target_quat_world) * R.from_matrix(achieved_mat).inv()).as_rotvec()
        )
    )
    return pos_err, rot_err


def _filter_and_rank_solutions(
    solutions: List[np.ndarray],
    model_native: mujoco.MjModel,
    data_native: mujoco.MjData,
    site_id: int,
    arm_dof_idx: np.ndarray,
    arm_qpos_idx: np.ndarray,
    full_qpos: np.ndarray,
    *,
    target_pos_world: np.ndarray,
    target_quat_world: np.ndarray,
    unique_atol: float = constants.IK_UNIQUE_TOLERANCE,
    min_manip: float = constants.IK_MIN_MANIPULABILITY,
    max_solutions: int = constants.IK_MAX_SOLUTIONS,
    fk_pos_tol: float = constants.EAIK_FK_POS_TOLERANCE,
    fk_rot_tol: float = constants.EAIK_FK_ROT_TOLERANCE,
) -> List[np.ndarray]:
    """De-duplicate, FK-verify, filter by manipulability, and rank solutions.

    Each candidate is set into MuJoCo via ``mj_fwdPosition`` (already
    required for the Jacobian) and its achieved EEF pose is compared
    against the world-frame target.  Solutions whose position error
    exceeds *fk_pos_tol* or orientation error exceeds *fk_rot_tol* are
    discarded — this catches unreachable targets and LS approximations
    that are too far off.

    Args:
        solutions: Raw solutions from ``solve_ik_eaik``.
        model_native: Native MuJoCo model.
        data_native: Native MuJoCo data (will be modified in-place).
        site_id: MuJoCo site ID for Jacobian computation.
        arm_dof_idx: DOF indices for the arm ``(7,)``.
        arm_qpos_idx: qpos indices for the arm ``(7,)``.
        full_qpos: Full model qpos vector ``(nq,)``.
        target_pos_world: Target position in world frame ``(3,)``.
        target_quat_world: Target quaternion ``(x,y,z,w)`` in world frame.
        unique_atol: Tolerance for de-duplication.
        min_manip: Minimum manipulability threshold.
        max_solutions: Maximum number of solutions to return.
        fk_pos_tol: Max allowed position error (m).
        fk_rot_tol: Max allowed orientation error (rad).

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

    # Score by manipulability, with FK verification
    scored: List[Tuple[float, np.ndarray]] = []
    for q in unique:
        # _compute_jacobian_mujoco calls mj_fwdPosition internally,
        # so site_xpos / site_xmat are up-to-date afterwards.
        J = _compute_jacobian_mujoco(
            model_native, data_native, site_id,
            arm_dof_idx, arm_qpos_idx, q, full_qpos,
        )

        # FK verification: reject solutions that don't reach the target
        pos_err, rot_err = _fk_error(
            data_native, site_id, target_pos_world, target_quat_world,
        )
        if pos_err > fk_pos_tol or rot_err > fk_rot_tol:
            logger.debug(
                "EAIK: solution rejected by FK check "
                "(pos_err=%.4f m > %.4f, rot_err=%.4f rad > %.4f)",
                pos_err, fk_pos_tol, rot_err, fk_rot_tol,
            )
            continue

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
    else:
        logger.debug("EAIK: no solutions passed FK and manipulability filtering")
        
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
    num_sweep: int = constants.EAIK_NUM_SWEEP_VALUES,
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

        if site_name is None:
            site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)

        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        # Get site ID for Jacobian computation and FK verification
        site_id = mujoco.mj_name2id(model_native, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise RuntimeError(f"MuJoCo site '{site_name}' not found in model")

        # --- World-frame target (for FK verification) ---
        pose_world = target_pose
        if pose_world.frame_id != constants.WORLD_FRAME:
            pose_world = tf_manager.transform_pose(pose_world, constants.WORLD_FRAME)
        target_pos_world = np.asarray(pose_world.position, dtype=float)
        target_quat_world = np.asarray(pose_world.quaternion, dtype=float)

        # --- Base-frame target (for EAIK) ---
        # EAIK solves IK in the URDF base frame (panda_link0 = robot_base).
        base_frame = f"robot{robot_id}_base"
        pose_base = target_pose
        if pose_base.frame_id != base_frame:
            pose_base = tf_manager.transform_pose(pose_base, base_frame)

        target_pos = np.asarray(pose_base.position, dtype=float)
        target_quat = np.asarray(pose_base.quaternion, dtype=float)

        # --- Pre-compensate for EAIK EEF ↔ MuJoCo site mismatch ---
        # EAIK's kinematic chain ends at a different point/orientation
        # than MuJoCo's grip_site (different frame conventions and the
        # fixed joint8 / gripper offset).  Compute the transform from
        # EAIK's EEF to MuJoCo's site using EAIK FK at the current
        # joint config, so frame conventions are consistent.
        q_current = np.array(data.qpos, dtype=float).ravel()[joint_qpos_idx]
        bot_fk = _build_eaik_robot(0, round(float(q_current[0]), 6))
        if bot_fk is None:
            raise RuntimeError("Failed to build EAIK robot for FK offset")
        T_eaik_current = np.array(bot_fk.fwdKin(list(q_current)))

        # MuJoCo site pose in robot base frame
        robot = env.robots[robot_id]
        base_body_name = robot.robot_model.root_body
        base_pos_w = np.array(data.get_body_xpos(base_body_name), dtype=float)
        base_rot_w = np.array(
            data.get_body_xmat(base_body_name), dtype=float,
        ).reshape(3, 3)
        base_rot_inv = base_rot_w.T

        site_pos_w = np.array(data_native.site_xpos[site_id], dtype=float)
        site_mat_w = np.array(data_native.site_xmat[site_id], dtype=float).reshape(3, 3)

        T_site_base = np.eye(4)
        T_site_base[:3, :3] = base_rot_inv @ site_mat_w
        T_site_base[:3, 3] = base_rot_inv @ (site_pos_w - base_pos_w)

        # Offset from EAIK EEF to MuJoCo site (in EAIK's frame)
        T_eaik_to_site = np.linalg.inv(T_eaik_current) @ T_site_base

        # EAIK target = desired_site_base @ inv(offset)
        T_site_desired = np.eye(4)
        T_site_desired[:3, :3] = R.from_quat(target_quat).as_matrix()
        T_site_desired[:3, 3] = target_pos

        T_eaik_target = T_site_desired @ np.linalg.inv(T_eaik_to_site)
        target_pos = T_eaik_target[:3, 3]
        target_quat = R.from_matrix(T_eaik_target[:3, :3]).as_quat()

        full_qpos = np.array(data.qpos, copy=True, dtype=float).ravel()

        # Save and restore simulation state
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            # Get all analytical solutions via multi-joint sweep
            raw_solutions = solve_ik_eaik(
                target_pos, target_quat,
                joint_lo, joint_hi,
                num_sweep=num_sweep,
                q_reference=q_reference,
            )

            logger.debug(
                "EAIK: %d raw solutions for robot%d at %s",
                len(raw_solutions), robot_id, target_pos,
            )

            # Filter, de-duplicate, FK-verify, and rank by manipulability
            solutions = _filter_and_rank_solutions(
                raw_solutions,
                model_native, data_native, site_id,
                joint_dof_idx, joint_qpos_idx, full_qpos,
                target_pos_world=target_pos_world,
                target_quat_world=target_quat_world,
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
