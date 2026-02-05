"""Inverse Kinematics (IK) utilities."""

from contextlib import nullcontext
import threading
from typing import List, Optional, Sequence, Union, Iterable

import numpy as np
import mujoco

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager
from robosuite.utils.binding_utils import MjModel, MjData

from robosuite_franka_manipulation.planning.utils import mujoco_utils

ArrayLike = Union[Sequence[float], np.ndarray]


def solve_ik_dls(
    model: MjModel,
    data: MjData,
    site_id: int,
    joint_qpos_idx: np.ndarray,
    joint_dof_idx: np.ndarray,
    joint_lo: np.ndarray,
    joint_hi: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    q_init: np.ndarray,
    *,
    max_iters: int,
    pos_tol: float,
    ori_tol: float,
    damping: float,
    step_size: float,
) -> Optional[np.ndarray]:
    """Iterative IK using damped least squares on a MuJoCo site."""
    # model and data are expected to be the public mujoco objects (env.sim.model, env.sim.data)

    q = np.asarray(q_init, dtype=float).copy()
    q = q.reshape(-1)
    q = np.clip(q, joint_lo, joint_hi)
    mujoco_utils.set_qpos(data, joint_qpos_idx, q)
    
    # Unwrap for native call
    mujoco.mj_forward(mujoco_utils.get_native_mj_object(model), mujoco_utils.get_native_mj_object(data))

    nv = int(model.nv)
    jacp = np.zeros((3, nv), dtype=float)
    jacr = np.zeros((3, nv), dtype=float)

    lam2 = float(damping) ** 2

    for _ in range(int(max_iters)):
        pos_err, rot_err = mujoco_utils.compute_site_error(model, data, site_id, target_pos, target_quat)
        if np.linalg.norm(pos_err) <= pos_tol and np.linalg.norm(rot_err) <= ori_tol:
            return q

        # Unwrap for native call
        mujoco.mj_jacSite(mujoco_utils.get_native_mj_object(model), mujoco_utils.get_native_mj_object(data), jacp, jacr, site_id)
        
        J = np.vstack([jacp[:, joint_dof_idx], jacr[:, joint_dof_idx]])  # (6, 7)
        err = np.concatenate([pos_err, rot_err], axis=0)

        # dq = J^T (J J^T + lam^2 I)^-1 * err
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + lam2 * np.eye(6), err)
        dq = step_size * dq

        q = np.clip(q + dq, joint_lo, joint_hi)
        mujoco_utils.set_qpos(data, joint_qpos_idx, q)
        
        # Unwrap for native call
        mujoco.mj_forward(mujoco_utils.get_native_mj_object(model), mujoco_utils.get_native_mj_object(data))

    # Final check
    pos_err, rot_err = mujoco_utils.compute_site_error(model, data, site_id, target_pos, target_quat)
    if np.linalg.norm(pos_err) <= pos_tol and np.linalg.norm(rot_err) <= ori_tol:
        return q
    return None


def unique_solutions(solutions: Iterable[np.ndarray], atol: float = 1e-3) -> List[np.ndarray]:
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
    target_frame: str = constants.WORLD_FRAME,
    arm: str = "right",
    site_name: Optional[str] = None,
    q_warm_start: Optional[np.ndarray] = None,
    env_lock: threading.RLock = None,
    num_seed_configs: int = constants.IK_NUM_SEED_CONFIGS,
    elbow_discretization: int = constants.IK_ELBOW_DISCRETIZATION,
    max_solutions: int = constants.IK_MAX_SOLUTIONS,
    pos_tol: float = constants.IK_POS_TOLERANCE,
    ori_tol: float = constants.IK_ORI_TOLERANCE,
    max_iters: int = constants.IK_MAX_ITERATIONS,
    damping: float = constants.IK_DAMPING,
    step_size: float = constants.IK_STEP_SIZE,
    unique_atol: float = constants.IK_UNIQUE_TOLERANCE,
) -> List[np.ndarray]:
    f"""Enumerate IK solutions for a single arm by running IK from multiple seeds.

    Discretization strategy (simple + robust):
    - Sample the last joint ("elbow/wrist" redundancy) across its joint range.
    - Add random seeds around the current configuration.
    - Run DLS IK from each seed; keep converged solutions; de-duplicate.

    Returns:
        List of joint angle arrays of shape (f{constants.DEFAULT_ROBOT_ACTION_DIM},).
    """

    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        joint_ids = mujoco_utils.get_arm_joint_ids(model, robot_id)
        if len(joint_ids) != constants.DEFAULT_ROBOT_ACTION_DIM:
            raise RuntimeError(f"Expected {constants.DEFAULT_ROBOT_ACTION_DIM} arm joints for robot{robot_id}, found {len(joint_ids)}.")

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
        # robosuite wraps the MjModel; use its name->id helper
        site_id = int(model.site_name2id(site_name))

        q_current = mujoco_utils.get_qpos(data, joint_qpos_idx)

        seeds: List[np.ndarray] = []

        # 0) Warm start if provided
        if q_warm_start is not None:
            q_warm = np.asarray(q_warm_start, dtype=float).copy()
            if q_warm.shape == q_current.shape:
                seeds.append(q_warm)
                # Add small perturbations around warm start
                rng = np.random.default_rng(0)
                for _ in range(num_seed_configs):
                     q_pert = q_warm + rng.normal(scale=0.05, size=q_warm.shape)
                     seeds.append(np.clip(q_pert, joint_lo, joint_hi))

        else:
            # 1) Discretize last joint across its limits
            last_lo, last_hi = float(joint_lo[-1]), float(joint_hi[-1])
            if elbow_discretization > 1 and np.isfinite(last_lo) and np.isfinite(last_hi) and last_hi > last_lo:
                for v in np.linspace(last_lo, last_hi, elbow_discretization):
                    q0 = q_current.copy()
                    q0[-1] = float(v)
                    seeds.append(q0)
            else:
                seeds.append(q_current.copy())

            # 2) Random perturbation seeds
            rng = np.random.default_rng(0)
            for _ in range(max(0, int(num_seed_configs))):
                q0 = q_current + rng.normal(scale=constants.IK_RANDOM_PERTURBATION_SCALE, size=q_current.shape)
                q0 = np.clip(q0, joint_lo, joint_hi)
                seeds.append(q0)

            # 3) A few uniform random seeds in joint space
            for _ in range(max(0, int(num_seed_configs // 3))):
                u = rng.uniform(low=joint_lo, high=joint_hi)
                seeds.append(u)

        # Save and restore simulation state
        qpos_backup = np.array(data.qpos, copy=True)
        qvel_backup = np.array(data.qvel, copy=True)

        solutions: List[np.ndarray] = []
        try:
            for q0 in seeds:
                q_sol = solve_ik_dls(
                    model,
                    data,
                    site_id,
                    joint_qpos_idx,
                    joint_dof_idx,
                    joint_lo,
                    joint_hi,
                    target_pos,
                    target_quat,
                    q0,
                    max_iters=max_iters,
                    pos_tol=pos_tol,
                    ori_tol=ori_tol,
                    damping=damping,
                    step_size=step_size,
                )
                if q_sol is None:
                    continue
                solutions.append(np.asarray(q_sol, dtype=float))
                solutions = unique_solutions(solutions, atol=unique_atol)
                if len(solutions) >= int(max_solutions):
                    break
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            
            # Unwrap for native call
            mujoco.mj_forward(mujoco_utils.get_native_mj_object(model), mujoco_utils.get_native_mj_object(data))

        return solutions
