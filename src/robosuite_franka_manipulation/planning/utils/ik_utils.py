"""Inverse Kinematics (IK) utilities.

Provides IK solvers and Jacobian computation for single-arm manipulation.
The solver (``solve_ik_mink``) uses `mink <https://github.com/kevinzakka/mink>`_'s
QP-based differential IK with automatic joint-limit enforcement and
singularity damping.
"""

from contextlib import nullcontext
import threading
from typing import List, Optional, Tuple, Iterable

import mink
from mink.lie import SE3 as MinkSE3, SO3 as MinkSO3
import numpy as np
import mujoco
import logging
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager

from robosuite_franka_manipulation.planning.utils import mujoco_utils

logger = logging.getLogger(__name__)


def _manipulability_from_jac(jac: np.ndarray) -> float:
    """Yoshikawa manipulability: sqrt(det(J @ J^T)).

    A robust formulation using singular values to avoid numerical issues
    with near-singular Jacobians.
    """
    sv = np.linalg.svd(jac, compute_uv=False)
    return float(np.prod(sv))  # == sqrt(det(J @ J^T))


# ---------------------------------------------------------------------------
# Frozen-DOF helper (shared with compute_arm_jacobian / mink IK)
# ---------------------------------------------------------------------------

def _get_frozen_dof_indices(
    model_native: mujoco.MjModel,
    active_dof_idx: np.ndarray,
) -> List[int]:
    """Return DOF indices of everything *except* the given active DOFs.

    Used to construct a ``DofFreezingTask`` so that only the target arm
    moves during IK.
    """
    active_set = set(int(d) for d in active_dof_idx)
    frozen: List[int] = []
    for jid in range(model_native.njnt):
        dof_adr = int(model_native.jnt_dofadr[jid])
        n_dof = {
            mujoco.mjtJoint.mjJNT_FREE: 6,
            mujoco.mjtJoint.mjJNT_BALL: 3,
            mujoco.mjtJoint.mjJNT_SLIDE: 1,
            mujoco.mjtJoint.mjJNT_HINGE: 1,
        }.get(int(model_native.jnt_type[jid]), 1)
        for d in range(dof_adr, dof_adr + n_dof):
            if d not in active_set:
                frozen.append(d)
    return frozen


def _build_target_se3(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> "MinkSE3":
    """Build a ``mink.SE3`` target from position + quaternion (xyzw)."""
    rot_matrix = R.from_quat(target_quat).as_matrix()
    so3 = MinkSO3.from_matrix(rot_matrix)
    return MinkSE3.from_rotation_and_translation(so3, np.asarray(target_pos, dtype=float))


# ---------------------------------------------------------------------------
# Primary IK solver — mink-based iterative QP
# ---------------------------------------------------------------------------

def solve_ik_mink(
    model_native: mujoco.MjModel,
    full_qpos: np.ndarray,
    site_name: str,
    arm_qpos_idx: np.ndarray,
    arm_dof_idx: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_init: np.ndarray,
    *,
    pos_tol: float,
    ori_tol: float,
    max_iters: int = constants.MINK_IK_MAX_ITERS,
    dt: float = constants.MINK_IK_DT,
) -> Optional[np.ndarray]:
    """Solve single-arm IK using mink's iterative QP differential IK.

    Creates a ``mink.Configuration`` seeded with *q_init* for the target
    arm (all other DOFs frozen via ``DofFreezingTask``), then iterates
    ``mink.solve_ik`` until the ``FrameTask`` error drops below the
    requested tolerances.

    Features:

    * Joint limits enforced automatically via ``ConfigurationLimit``.
    * Singularity damping via ``lm_damping`` in ``FrameTask``
      and the ``DampingTask`` regulariser.
    * No manual Jacobian computation.

    Args:
        model_native: **Native** ``mujoco.MjModel`` (not robosuite wrapper).
        full_qpos: Full model qpos vector ``(nq,)`` — other joints' positions
            provide context for the kinematic tree.
        site_name: MuJoCo site name whose pose is driven to the target.
        arm_qpos_idx: qpos indices for the arm being solved (7,).
        arm_dof_idx: DOF (velocity) indices for the arm (7,).
        target_pos: Desired site position in **world** frame ``(3,)``.
        target_quat: Desired site orientation as quaternion ``(x, y, z, w)``.
        q_init: Initial arm joint configuration ``(7,)``.
        pos_tol: Convergence tolerance on position error (metres).
        ori_tol: Convergence tolerance on orientation error (radians).
        max_iters: Maximum QP iterations.
        dt: Integration time-step per QP iteration (seconds).

    Returns:
        Converged arm joint configuration ``(7,)`` or *None* on failure.
    """
    # ── Seed the full qpos with q_init for the target arm ─────────────
    q_full = np.array(full_qpos, dtype=float)
    q_full[arm_qpos_idx] = q_init

    # ── Build mink objects (lightweight, no persistent state needed) ──
    config = mink.Configuration(model_native, q=q_full)

    frame_task = mink.FrameTask(
        frame_name=site_name,
        frame_type="site",
        position_cost=constants.MINK_POSITION_COST,
        orientation_cost=constants.MINK_ORIENTATION_COST,
        gain=constants.MINK_FRAME_GAIN,
        lm_damping=constants.MINK_LM_DAMPING,
    )

    posture_task = mink.PostureTask(model=model_native, cost=constants.MINK_POSTURE_COST)
    posture_task.set_target(q_full)  # bias toward initial posture

    damping_task = mink.DampingTask(model_native, cost=constants.MINK_DAMPING_COST)
    config_limit = mink.ConfigurationLimit(model_native)

    frozen_dofs = _get_frozen_dof_indices(model_native, arm_dof_idx)
    freeze_task = mink.DofFreezingTask(model_native, dof_indices=frozen_dofs) if frozen_dofs else None

    # ── Set target SE3 ────────────────────────────────────────────────
    target_se3 = _build_target_se3(target_pos, target_quat)
    frame_task.set_target(target_se3)

    # ── Build task / limit lists ──────────────────────────────────────
    tasks: list = [frame_task, posture_task, damping_task]
    if freeze_task is not None:
        tasks.append(freeze_task)
    limits = [config_limit]

    # ── Iterative QP solve ────────────────────────────────────────────
    prev_err_norm = float("inf")
    stall_count = 0

    for i in range(max_iters):
        try:
            vel = mink.solve_ik(
                config, tasks, dt,
                solver=constants.MINK_QP_SOLVER,
                limits=limits,
            )
        except Exception:  # noqa: BLE001 — QP infeasible at this step
            logger.debug("mink QP infeasible at iter %d, stopping.", i)
            break

        config.integrate_inplace(vel, dt)

        # ── Convergence check ─────────────────────────────────────────
        current_se3 = config.get_transform_frame_to_world(site_name, "site")
        pos_err = float(np.linalg.norm(current_se3.translation() - target_pos))

        rel_se3 = target_se3.inverse().multiply(current_se3)
        ori_err = float(np.linalg.norm(rel_se3.rotation().log()))

        if pos_err < pos_tol and ori_err < ori_tol:
            return np.array(config.q[arm_qpos_idx], dtype=float)

        # ── Stall detection ───────────────────────────────────────────
        err_norm = pos_err + ori_err
        if abs(prev_err_norm - err_norm) < constants.MINK_IK_STALL_TOL:
            stall_count += 1
            if stall_count >= constants.MINK_IK_STALL_WINDOW:
                logger.debug(
                    "mink IK stalled at iter %d (pos=%.5f, ori=%.5f)",
                    i, pos_err, ori_err,
                )
                break
        else:
            stall_count = 0
        prev_err_norm = err_norm

    # ── Final tolerance check (may have converged on last iter) ───────
    current_se3 = config.get_transform_frame_to_world(site_name, "site")
    pos_err = float(np.linalg.norm(current_se3.translation() - target_pos))
    rel_se3 = target_se3.inverse().multiply(current_se3)
    ori_err = float(np.linalg.norm(rel_se3.rotation().log()))

    if pos_err < pos_tol and ori_err < ori_tol:
        return np.array(config.q[arm_qpos_idx], dtype=float)

    logger.debug(
        "mink IK failed: pos_err=%.5f (tol %.4f), ori_err=%.5f (tol %.4f)",
        pos_err, pos_tol, ori_err, ori_tol,
    )
    return None


def unique_solutions(solutions: Iterable[np.ndarray], atol: float = 1e-3) -> List[np.ndarray]:
    uniq: List[np.ndarray] = []
    for q in solutions:
        if not any(np.allclose(q, u, atol=atol, rtol=0.0) for u in uniq):
            uniq.append(q)
    return uniq


def compute_arm_jacobian(
    env: DualArmEnvironment,
    *,
    robot_id: int,
    arm: str = "right",
    q_config: Optional[np.ndarray] = None,
    site_name: Optional[str] = None,
    env_lock: threading.RLock = None,
) -> np.ndarray:
    """Compute the 6×7 geometric Jacobian at the EEF flange site.

    Uses ``mink.Configuration.get_frame_jacobian`` and slices to the
    arm's DOF columns.  The Jacobian maps 7-DOF joint velocities to
    6-D spatial velocities ``[v_x, v_y, v_z, ω_x, ω_y, ω_z]``
    expressed in the **body (local) frame** of the site (mink convention).

    Args:
        env: The dual-arm simulation environment.
        robot_id: Index of the robot (0 or 1).
        arm: Which arm (``"right"`` or ``"left"``).
        q_config: Joint configuration (7,) at which to evaluate the Jacobian.
            If *None*, the current simulation configuration is used.
        site_name: Override for the MuJoCo site name.  Defaults to
            ``get_eef_site_name(robot_id, arm)``.
        env_lock: Optional reentrant lock guarding the environment.

    Returns:
        ``np.ndarray`` of shape ``(6, 7)`` – the stacked translational and
        rotational Jacobian columns for the arm joints.
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

        # Build full qpos, optionally overriding the arm config
        q_full = np.array(data.qpos, copy=True, dtype=float).ravel()
        if q_config is not None:
            q_full[joint_qpos_idx] = q_config

        config = mink.Configuration(model_native, q=q_full)

        # get_frame_jacobian returns (6, nv) in the site's local (body) frame.
        J_full = config.get_frame_jacobian(site_name, "site")  # (6, nv)
        return J_full[:, joint_dof_idx]  # (6, 7)


def enumerate_ik_solutions_for_arm(
    env: DualArmEnvironment,
    *,
    robot_id: int,
    target_pose: StampedPose,
    tf_manager: TransformManager,
    arm: str = "right",
    site_name: Optional[str] = None,
    q_warm_start: Optional[np.ndarray] = None,
    env_lock: threading.RLock = None,
    num_seed_configs: int = constants.IK_NUM_SEED_CONFIGS,
    elbow_discretization: int = constants.IK_ELBOW_DISCRETIZATION,
    max_solutions: int = constants.IK_MAX_SOLUTIONS,
    pos_tol: float = constants.IK_POS_TOLERANCE,
    ori_tol: float = constants.IK_ORI_TOLERANCE,
    unique_atol: float = constants.IK_UNIQUE_TOLERANCE,
) -> List[np.ndarray]:
    f"""Enumerate IK solutions for a single arm by running IK from multiple seeds.

    Discretization strategy (simple + robust):
    - Sample the last joint ("elbow/wrist" redundancy) across its joint range.
    - Add random seeds around the current configuration.
    - Run mink-based IK from each seed; keep converged solutions; de-duplicate.
    - Rank surviving solutions by manipulability (best-conditioned first).

    Returns:
        List of joint angle arrays of shape (f{constants.FRANKA_ARM_DOF},).
    """

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        if len(joint_ids) != constants.FRANKA_ARM_DOF:
            raise RuntimeError(f"Expected {constants.FRANKA_ARM_DOF} arm joints for robot{robot_id}, found {len(joint_ids)}.")

        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        joint_dof_idx = mujoco_utils.get_joint_dof_indices(model, joint_ids)
        joint_lo, joint_hi = mujoco_utils.get_joint_limits(model, joint_ids)

        # MuJoCo site poses (site_xpos/site_xmat) are expressed in WORLD.
        # Therefore, run IK in world coordinates.
        
        pose_sp = target_pose

        if pose_sp.frame_id != constants.WORLD_FRAME:
            pose_sp = tf_manager.transform_pose(pose_sp, constants.WORLD_FRAME)

        target_pos = np.asarray(pose_sp.position, dtype=float)
        target_quat = np.asarray(pose_sp.quaternion, dtype=float)

        if site_name is None:
            site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)

        q_current = mujoco_utils.get_qpos(data, joint_qpos_idx)

        # Separate seeds into priority groups
        priority_seeds: List[np.ndarray] = []
        fallback_seeds: List[np.ndarray] = []

        rng = np.random.default_rng(0)

        # 1) Warm start and perturbations (Highest Priority)
        if q_warm_start is not None:
            q_warm = np.asarray(q_warm_start, dtype=float).copy()
            if q_warm.shape == q_current.shape:
                priority_seeds.append(q_warm)
                # Add small perturbations around warm start
                for _ in range(num_seed_configs):
                     q_pert = q_warm + rng.normal(scale=0.05, size=q_warm.shape)
                     priority_seeds.append(np.clip(q_pert, joint_lo, joint_hi))
        else:
            # If no warm start, current pose is the best guess
            priority_seeds.append(q_current.copy())

        # 2) Fallback seeds: Discretize and Random (Lower Priority)
        # Discretize last joint
        last_lo, last_hi = float(joint_lo[-1]), float(joint_hi[-1])
        if elbow_discretization > 1 and np.isfinite(last_lo) and np.isfinite(last_hi) and last_hi > last_lo:
            for v in np.linspace(last_lo, last_hi, elbow_discretization):
                q0 = q_current.copy()
                q0[-1] = float(v)
                fallback_seeds.append(q0)

        # Random perturbation seeds from current
        for _ in range(max(0, int(num_seed_configs))):
            q0 = q_current + rng.normal(scale=constants.IK_RANDOM_PERTURBATION_SCALE, size=q_current.shape)
            fallback_seeds.append(np.clip(q0, joint_lo, joint_hi))

        # Uniform random seeds
        for _ in range(max(0, int(num_seed_configs))):
            fallback_seeds.append(rng.uniform(low=joint_lo, high=joint_hi))

        # Save and restore simulation state
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        # Pre-compute mink helpers (native model, frozen DOFs, full qpos)
        model_native = mujoco_utils.get_native_mj_object(model)
        full_qpos = np.array(data.qpos, copy=True, dtype=float).ravel()

        min_manip = constants.IK_MIN_MANIPULABILITY
        scored: List[Tuple[float, np.ndarray]] = []

        def _check_manipulability(q_sol: np.ndarray) -> Optional[float]:
            """Compute manipulability and return the score if above threshold, else None."""
            q_full = full_qpos.copy()
            q_full[joint_qpos_idx] = q_sol
            config = mink.Configuration(model_native, q=q_full)
            J = config.get_frame_jacobian(site_name, "site")
            J_arm = J[:, joint_dof_idx]  # (6, 7)
            m = _manipulability_from_jac(J_arm)
            if m < min_manip:
                logger.debug("IK solution rejected: manipulability %.4e < %.2e", m, min_manip)
                return None
            return m

        def run_ik_on_seeds(seeds_list):
            nonlocal scored
            for q0 in seeds_list:
                q_sol = solve_ik_mink(
                    model_native,
                    full_qpos,
                    site_name,
                    joint_qpos_idx,
                    joint_dof_idx,
                    target_pos,
                    target_quat,
                    q0,
                    pos_tol=pos_tol,
                    ori_tol=ori_tol,
                )
                if q_sol is not None:
                    q_sol = np.asarray(q_sol, dtype=float)
                    # De-duplicate: check BEFORE computing manipulability (O(n) per solution)
                    if any(np.allclose(q_sol, s[1], atol=unique_atol, rtol=0.0) for s in scored):
                        continue
                    # Check manipulability before accepting the solution
                    m = _check_manipulability(q_sol)
                    if m is None:
                        continue
                    scored.append((m, q_sol))
                    if len(scored) >= int(max_solutions):
                        return True
            return False

        try:
            # Try priority seeds first (warm-start + perturbations)
            run_ik_on_seeds(priority_seeds)

            # Try fallback seeds if we haven't filled max_solutions yet
            if len(scored) < max_solutions:
                run_ik_on_seeds(fallback_seeds)

            if not scored:
                logger.debug(f"No IK solutions found for robot {robot_id} at {target_pos}")

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup

            # Unwrap for native call
            mujoco.mj_fwdPosition(mujoco_utils.get_native_mj_object(model), mujoco_utils.get_native_mj_object(data))

        # ------------------------------------------------------------------
        # Sort accepted solutions by manipulability (best first).
        # ------------------------------------------------------------------
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            solutions = [q for _, q in scored]
            logger.debug(
                "IK solutions ranked by manipulability: best=%.4e, worst=%.4e (%d solutions)",
                scored[0][0], scored[-1][0], len(scored),
            )
        else:
            solutions = []

        return solutions


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
