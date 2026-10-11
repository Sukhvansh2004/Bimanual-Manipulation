"""Mink QP-based iterative IK solver.

Uses ``mink.solve_ik`` with a ``FrameTask``, ``PostureTask``,
``DampingTask``, ``DofFreezingTask``, and joint limits to converge on a
target EEF pose via quadratic-programming differential IK steps.

Multiple random seeds are tried to explore the solution manifold.
"""

import logging
import threading
from contextlib import nullcontext
from typing import List, Optional

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.planning.ik.manipulability import (
    compute_manipulability_from_jacobian,
)
from bimanual_manipulation.utils import mujoco_utils
from bimanual_manipulation.utils.mujoco_utils import compute_site_jacobian
from bimanual_manipulation.transforms import StampedPose, TransformManager

logger = logging.getLogger(__name__)


def _build_target_se3(target_pos: np.ndarray, target_quat: np.ndarray):
    """Convert position + quaternion (xyzw) into a ``mink.lie.SE3``.

    Args:
        target_pos: ``(3,)`` position.
        target_quat: ``(4,)`` quaternion ``(x, y, z, w)``.

    Returns:
        The corresponding ``mink.SE3`` transform.
    """
    rot = R.from_quat(target_quat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = target_pos
    return mink.SE3.from_matrix(T)


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
    pos_tol: float = constants.MINK_IK_POS_TOLERANCE,
    ori_tol: float = constants.MINK_IK_ROT_TOLERANCE,
    max_iters: int = constants.MINK_IK_MAX_ITERS,
    dt: float = constants.MINK_IK_DT,
) -> Optional[np.ndarray]:
    """Solve IK iteratively using mink QP differential IK.

    Tracks the target with ``FrameTask``, ``PostureTask`` (toward the
    initial config), ``DampingTask`` and a ``DofFreezingTask`` (as a
    task) for all non-arm DOFs, subject to ``ConfigurationLimit``.
    Stops early when ``pos_err + rot_err`` fails to improve by
    ``MINK_IK_STALL_TOL`` for ``MINK_IK_STALL_WINDOW`` iterations.

    Args:
        model_native: Native MuJoCo model.
        full_qpos: Full simulation qpos vector.
        site_name: EEF site name in the model.
        arm_qpos_idx: qpos indices for the arm joints.
        arm_dof_idx: DOF indices for the arm joints.
        target_pos: Desired EEF position in world frame ``(3,)``.
        target_quat: Desired EEF quaternion ``(x,y,z,w)`` in world frame.
        q_init: Initial joint configuration ``(7,)``.
        pos_tol: Position convergence tolerance (m).
        ori_tol: Orientation convergence tolerance (rad).
        max_iters: Maximum QP iterations.
        dt: Integration timestep per iteration.

    Returns:
        Converged joint configuration ``(7,)``, or ``None`` if not
        converged within *max_iters* or stalled.
    """
    q_full = full_qpos.copy()
    q_full[arm_qpos_idx] = q_init

    config = mink.Configuration(model_native, q=q_full)

    # Build the EEF tracking task
    eef_task = mink.FrameTask(
        frame_name=site_name,
        frame_type="site",
        position_cost=constants.MINK_POSITION_COST,
        orientation_cost=constants.MINK_ORIENTATION_COST,
        gain=constants.MINK_FRAME_GAIN,
        lm_damping=constants.MINK_LM_DAMPING,
    )
    target_se3 = _build_target_se3(target_pos, target_quat)
    eef_task.set_target(target_se3)

    posture_task = mink.PostureTask(model=model_native, cost=constants.MINK_POSTURE_COST)
    posture_task.set_target_from_configuration(config)
    damping_task = mink.DampingTask(model=model_native, cost=constants.MINK_DAMPING_COST)

    # Freeze all DOFs except the arm
    frozen_dofs = mujoco_utils.get_frozen_dof_indices(model_native, arm_dof_idx)
    freeze_task = None
    if frozen_dofs:
        freeze_task = mink.DofFreezingTask(model=model_native, dof_indices=frozen_dofs)

    tasks = [eef_task, posture_task, damping_task]
    if freeze_task is not None:
        tasks.append(freeze_task)

    limits = [
        mink.ConfigurationLimit(model_native),
    ]

    R_target = R.from_quat(target_quat)

    stall_window = constants.MINK_IK_STALL_WINDOW
    best_err = float("inf")
    stall_count = 0

    for _ in range(max_iters):
        vel = mink.solve_ik(
            config, tasks, dt,
            solver=constants.MINK_QP_SOLVER,
            limits=limits,
        )
        config.integrate_inplace(vel, dt)

        # Check convergence
        T_current = config.get_transform_frame_to_world(site_name, "site")
        achieved_pos = T_current.translation()
        achieved_rot = T_current.rotation()
        pos_err = float(np.linalg.norm(achieved_pos - target_pos))
        rot_err = float(np.linalg.norm(
            (R_target * R.from_matrix(achieved_rot.as_matrix()).inv()).as_rotvec()
        ))

        if pos_err <= pos_tol and rot_err <= ori_tol:
            return config.q[arm_qpos_idx].copy()

        total_err = pos_err + rot_err
        if total_err < best_err - constants.MINK_IK_STALL_TOL:
            best_err = total_err
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= stall_window:
                logger.debug("Mink IK stalled at pos_err=%.4f, rot_err=%.4f", pos_err, rot_err)
                return None

    logger.debug("Mink IK did not converge in %d iters", max_iters)
    return None


def _generate_seeds(
    q_reference: Optional[np.ndarray],
    joint_lo: np.ndarray,
    joint_hi: np.ndarray,
    n_seeds: int,
    q_current: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Generate IK seed configurations.

    If *q_reference* is given it is the first seed, followed by
    *q_current* (when provided and different) and Gaussian perturbations
    (σ = 0.25 rad) around *q_reference* for the remaining slots.
    Otherwise *q_current* (when provided) and ``n_seeds // 3``
    perturbations around it (σ = 0.3 rad) come first, and the remaining
    slots use uniform random seeds within the joint limits.  Perturbed
    seeds are clipped to the limits.  An unseeded RNG is used, so seeds
    are non-deterministic.

    Args:
        q_reference: Optional reference configuration.
        joint_lo: Lower joint limits.
        joint_hi: Upper joint limits.
        n_seeds: Target number of seeds.
        q_current: Optional current arm configuration.

    Returns:
        List of seed configurations (at least one when *q_reference* or
        *q_current* is given, even if *n_seeds* is smaller).
    """
    rng = np.random.default_rng()
    seeds: List[np.ndarray] = []
    if q_reference is not None:
        seeds.append(np.asarray(q_reference, dtype=float).copy())
        if q_current is not None and not np.allclose(q_reference, q_current, atol=1e-6):
            seeds.append(np.asarray(q_current, dtype=float).copy())
        for _ in range(n_seeds - len(seeds)):
            perturbed = q_reference + rng.normal(scale=0.25, size=len(joint_lo))
            seeds.append(np.clip(perturbed, joint_lo, joint_hi))
    else:
        if q_current is not None:
            seeds.append(np.asarray(q_current, dtype=float).copy())
            # Add perturbations around current config
            for _ in range(min(n_seeds // 3, n_seeds - len(seeds))):
                perturbed = q_current + rng.normal(scale=0.3, size=len(joint_lo))
                seeds.append(np.clip(perturbed, joint_lo, joint_hi))
        for _ in range(n_seeds - len(seeds)):
            seeds.append(rng.uniform(joint_lo, joint_hi))
    return seeds


def enumerate_ik_solutions_mink(
    env: DualArmEnvironment,
    *,
    robot_id: int,
    target_pose: StampedPose,
    tf_manager: TransformManager,
    arm: str = "right",
    site_name: Optional[str] = None,
    env_lock: Optional[threading.RLock] = None,
    max_solutions: int = constants.IK_MAX_SOLUTIONS,
    unique_atol: float = constants.IK_UNIQUE_TOLERANCE,
    q_reference: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Enumerate IK solutions using mink QP-based iterative IK.

    Same interface as ``enumerate_ik_solutions_eaik``.  Runs
    :func:`solve_ik_mink` from ``MINK_IK_NUM_SEEDS`` seeds (see
    :func:`_generate_seeds`), de-duplicates, drops solutions below
    ``IK_MIN_MANIPULABILITY`` and ranks by manipulability.  Simulation
    qpos/qvel are restored afterwards.

    Args:
        env: The dual-arm simulation environment.
        robot_id: Robot index (0 or 1).
        target_pose: Desired EEF site pose (any frame known to
            *tf_manager*).
        tf_manager: For frame conversions.
        arm: Which arm (``"right"`` or ``"left"``).
        site_name: Override for the MuJoCo site name.
        env_lock: Optional threading lock for environment access.
        max_solutions: Maximum solutions to return.
        unique_atol: Tolerance for uniqueness filtering.
        q_reference: Optional reference configuration used as the first
            seed and for re-sorting by L2 distance (closest first).

    Returns:
        List of joint configurations ``(7,)``, ranked by proximity to
        *q_reference* if given, otherwise by manipulability.

    Raises:
        RuntimeError: If the arm does not have 7 joints or the site is
            not found.
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

        site_id = mujoco.mj_name2id(
            model_native, mujoco.mjtObj.mjOBJ_SITE, site_name,
        )
        if site_id < 0:
            raise RuntimeError(f"MuJoCo site '{site_name}' not found in model")

        # World-frame target
        pose_world = target_pose
        if pose_world.frame_id != constants.WORLD_FRAME:
            pose_world = tf_manager.transform_pose(pose_world, constants.WORLD_FRAME)
        target_pos = np.asarray(pose_world.position, dtype=float)
        target_quat = np.asarray(pose_world.quaternion, dtype=float)

        full_qpos = np.array(data.qpos, copy=True, dtype=float).ravel()
        q_current_arm = full_qpos[joint_qpos_idx].copy()

        # Save and restore simulation state
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            seeds = _generate_seeds(
                q_reference, joint_lo, joint_hi,
                constants.MINK_IK_NUM_SEEDS,
                q_current=q_current_arm,
            )

            raw_solutions: List[np.ndarray] = []
            for seed in seeds:
                result = solve_ik_mink(
                    model_native, full_qpos, site_name,
                    joint_qpos_idx, joint_dof_idx,
                    target_pos, target_quat, seed,
                )
                if result is not None:
                    raw_solutions.append(result)

            logger.debug(
                "Mink IK: %d/%d seeds converged for robot%d at pos %s",
                len(raw_solutions), len(seeds), robot_id,
                np.array2string(target_pos, precision=3),
            )

            # De-duplicate (vectorised: sort then compare neighbours)
            if raw_solutions:
                sol_arr = np.array(raw_solutions)
                order = np.lexsort(sol_arr[:, ::-1].T)
                sorted_sols = sol_arr[order]
                if len(sorted_sols) > 1:
                    diffs = np.abs(sorted_sols[1:] - sorted_sols[:-1])
                    is_dup = np.all(diffs <= unique_atol, axis=1)
                    mask = np.empty(len(sorted_sols), dtype=bool)
                    mask[0] = True
                    mask[1:] = ~is_dup
                    unique = [sorted_sols[i] for i in np.flatnonzero(mask)]
                else:
                    unique = [sorted_sols[0]]
            else:
                unique = []

            # Filter by manipulability and rank
            scored = []
            for q in unique:
                J = compute_site_jacobian(
                    model_native, data_native, site_id,
                    joint_dof_idx, joint_qpos_idx, q, full_qpos,
                )
                m = compute_manipulability_from_jacobian(J)
                if m >= constants.IK_MIN_MANIPULABILITY:
                    scored.append((m, q))

            scored.sort(key=lambda x: x[0], reverse=True)
            solutions = [q for _, q in scored[:max_solutions]]

        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_fwdPosition(model_native, data_native)

        # Re-sort by proximity to reference for trajectory continuity
        if q_reference is not None and len(solutions) > 1:
            solutions.sort(key=lambda q: float(np.linalg.norm(q - q_reference)))

        return solutions



