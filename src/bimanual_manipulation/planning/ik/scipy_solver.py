"""Scipy L-BFGS-B IK solver.

Uses MuJoCo forward kinematics for the cost function and computes IK
via ``scipy.optimize.minimize`` with the L-BFGS-B method.  Multiple
random seeds are tried to explore the solution manifold.
"""

import logging
import threading
from contextlib import nullcontext
from typing import List, Optional

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.planning.ik.manipulability import (
    compute_manipulability_from_jacobian,
)
from bimanual_manipulation.planning.utils import mujoco_utils
from bimanual_manipulation.transforms import StampedPose, TransformManager

logger = logging.getLogger(__name__)


def _fk_objective_and_grad(
    q_arm: np.ndarray,
    model_native: mujoco.MjModel,
    data_native: mujoco.MjData,
    site_id: int,
    arm_qpos_idx: np.ndarray,
    arm_dof_idx: np.ndarray,
    full_qpos: np.ndarray,
    target_pos: np.ndarray,
    R_target: R,
) -> tuple:
    """Return (cost, gradient) for the IK objective.

    Cost is ``||pos_err||^2 + ||rot_err||^2``.  The analytical gradient
    is ``-2 J^T e`` where *J* is the geometric Jacobian and
    *e = [pos_err; rot_err]*.
    """
    q_full = full_qpos.copy()
    q_full[arm_qpos_idx] = q_arm
    data_native.qpos[:] = q_full
    mujoco.mj_fwdPosition(model_native, data_native)

    pos_err, rot_err = mujoco_utils.compute_site_error_fast(
        data_native, site_id, target_pos, R_target,
    )
    err = np.concatenate([pos_err, rot_err])
    cost = float(np.dot(err, err))

    # Geometric Jacobian at the site
    jacp = np.zeros((3, model_native.nv))
    jacr = np.zeros((3, model_native.nv))
    mujoco.mj_jacSite(model_native, data_native, jacp, jacr, site_id)
    J = np.vstack([jacp[:, arm_dof_idx], jacr[:, arm_dof_idx]])

    grad = -2.0 * J.T @ err
    return cost, grad


def solve_ik_scipy(
    model_native: mujoco.MjModel,
    data_native: mujoco.MjData,
    site_id: int,
    arm_qpos_idx: np.ndarray,
    arm_dof_idx: np.ndarray,
    full_qpos: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    joint_lo: np.ndarray,
    joint_hi: np.ndarray,
    *,
    q_seeds: List[np.ndarray],
    pos_tol: float = constants.SCIPY_IK_POS_TOLERANCE,
    rot_tol: float = constants.SCIPY_IK_ROT_TOLERANCE,
    max_iter: int = constants.SCIPY_IK_MAX_ITERS,
) -> List[np.ndarray]:
    """Solve IK using L-BFGS-B from multiple seeds.

    Args:
        model_native: Native MuJoCo model.
        data_native: Native MuJoCo data (modified in-place, caller should backup).
        site_id: MuJoCo site ID for the EEF.
        arm_qpos_idx: qpos indices for arm joints.
        arm_dof_idx: DOF indices for arm joints.
        full_qpos: Full model qpos vector.
        target_pos: Desired position in world frame ``(3,)``.
        target_quat: Desired quaternion ``(x,y,z,w)`` in world frame.
        joint_lo: Lower joint limits ``(7,)``.
        joint_hi: Upper joint limits ``(7,)``.
        q_seeds: Initial guesses for the optimiser.
        pos_tol: Max position error to accept a solution.
        rot_tol: Max orientation error to accept a solution.
        max_iter: Maximum L-BFGS-B iterations per seed.

    Returns:
        List of converged joint configurations.
    """
    bounds = list(zip(joint_lo.tolist(), joint_hi.tolist()))
    solutions: List[np.ndarray] = []
    R_target = R.from_quat(target_quat)

    for q0 in q_seeds:
        result = minimize(
            _fk_objective_and_grad,
            q0,
            args=(
                model_native, data_native, site_id,
                arm_qpos_idx, arm_dof_idx, full_qpos,
                target_pos, R_target,
            ),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-8},
        )
        q_sol = np.clip(result.x, joint_lo, joint_hi)

        # Verify FK accuracy
        data_native.qpos[:] = full_qpos.copy()
        data_native.qpos[arm_qpos_idx] = q_sol
        mujoco.mj_fwdPosition(model_native, data_native)
        pos_err, rot_err = mujoco_utils.compute_site_error_fast(
            data_native, site_id, target_pos, R_target,
        )
        p_err = float(np.linalg.norm(pos_err))
        r_err = float(np.linalg.norm(rot_err))

        if p_err <= pos_tol and r_err <= rot_tol:
            solutions.append(q_sol)
        else:
            logger.debug(
                "Scipy IK seed rejected: pos_err=%.4f m, rot_err=%.4f rad",
                p_err, r_err,
            )

    return solutions


def _generate_seeds(
    q_reference: Optional[np.ndarray],
    joint_lo: np.ndarray,
    joint_hi: np.ndarray,
    n_seeds: int,
    q_current: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Generate IK seed configurations.

    If *q_reference* is given it is always the first seed, followed by
    perturbations around it.  The current arm configuration *q_current*
    is always included when provided.  Remaining slots use uniform
    random seeds for coverage.
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


def enumerate_ik_solutions_scipy(
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
    """Enumerate IK solutions using Scipy L-BFGS-B optimiser.

    Same interface as ``enumerate_ik_solutions_eaik``.
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
                constants.SCIPY_IK_NUM_SEEDS,
                q_current=q_current_arm,
            )

            raw_solutions = solve_ik_scipy(
                model_native, data_native, site_id,
                joint_qpos_idx, joint_dof_idx, full_qpos,
                target_pos, target_quat,
                joint_lo, joint_hi,
                q_seeds=seeds,
            )

            logger.debug(
                "Scipy IK: %d/%d seeds converged for robot%d at pos %s",
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
                J = _compute_jacobian(
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


def _compute_jacobian(
    model_native: mujoco.MjModel,
    data_native: mujoco.MjData,
    site_id: int,
    arm_dof_idx: np.ndarray,
    arm_qpos_idx: np.ndarray,
    q_arm: np.ndarray,
    full_qpos: np.ndarray,
) -> np.ndarray:
    """Compute 6x7 geometric Jacobian using raw MuJoCo."""
    q_full = full_qpos.copy()
    q_full[arm_qpos_idx] = q_arm
    data_native.qpos[:] = q_full
    mujoco.mj_fwdPosition(model_native, data_native)

    jacp = np.zeros((3, model_native.nv))
    jacr = np.zeros((3, model_native.nv))
    mujoco.mj_jacSite(model_native, data_native, jacp, jacr, site_id)
    return np.vstack([jacp[:, arm_dof_idx], jacr[:, arm_dof_idx]])


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

    Returns:
        ``np.ndarray`` of shape ``(6, 7)``.
    """
    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        joint_qpos_idx = mujoco_utils.get_joint_qpos_indices(model, joint_ids)
        joint_dof_idx = mujoco_utils.get_joint_dof_indices(model, joint_ids)

        if site_name is None:
            site_name = mujoco_utils.get_eef_site_name(robot_id, arm=arm)

        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        site_id = mujoco.mj_name2id(
            model_native, mujoco.mjtObj.mjOBJ_SITE, site_name,
        )
        if site_id < 0:
            raise RuntimeError(f"MuJoCo site '{site_name}' not found in model")

        full_qpos = np.array(data.qpos, copy=True, dtype=float).ravel()
        q_arm = np.asarray(q_config, dtype=float) if q_config is not None else full_qpos[joint_qpos_idx]

        return _compute_jacobian(
            model_native, data_native, site_id,
            joint_dof_idx, joint_qpos_idx, q_arm, full_qpos,
        )
