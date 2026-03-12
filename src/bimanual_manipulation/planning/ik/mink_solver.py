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
from bimanual_manipulation.planning.utils import mujoco_utils
from bimanual_manipulation.transforms import StampedPose, TransformManager

logger = logging.getLogger(__name__)


def _build_target_se3(target_pos: np.ndarray, target_quat: np.ndarray):
    """Convert position + quaternion (xyzw) into a ``mink.lie.SE3``."""
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
        Converged joint configuration ``(7,)`` or ``None``.
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
            (R.from_quat(target_quat) * R.from_matrix(achieved_rot.as_matrix()).inv()).as_rotvec()
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
) -> List[np.ndarray]:
    """Generate IK seed configurations."""
    rng = np.random.default_rng()
    seeds: List[np.ndarray] = []
    if q_reference is not None:
        seeds.append(np.asarray(q_reference, dtype=float).copy())
        for _ in range(n_seeds - 1):
            perturbed = q_reference + rng.normal(scale=0.25, size=len(joint_lo))
            seeds.append(np.clip(perturbed, joint_lo, joint_hi))
    else:
        for _ in range(n_seeds):
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

        # Save and restore simulation state
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        try:
            seeds = _generate_seeds(
                q_reference, joint_lo, joint_hi,
                constants.MINK_IK_NUM_SEEDS,
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
                "Mink IK: %d raw solutions for robot%d", len(raw_solutions), robot_id,
            )

            # De-duplicate
            unique: List[np.ndarray] = []
            for q in raw_solutions:
                if not any(np.allclose(q, u, atol=unique_atol, rtol=0.0) for u in unique):
                    unique.append(q)

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
