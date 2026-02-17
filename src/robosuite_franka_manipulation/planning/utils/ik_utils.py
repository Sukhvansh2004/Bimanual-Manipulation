"""Inverse Kinematics (IK) utilities."""

from contextlib import nullcontext
import threading
from typing import List, Optional, Sequence, Union, Iterable

import numpy as np
import mujoco
import logging
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager
from robosuite.utils.binding_utils import MjModel, MjData

from robosuite_franka_manipulation.planning.utils import mujoco_utils

logger = logging.getLogger(__name__)

ArrayLike = Union[Sequence[float], np.ndarray]


def solve_ik_scipy(
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
    pos_tol: float,
    ori_tol: float,
) -> Optional[np.ndarray]:
    """Residual-based IK using scipy.optimize.minimize with SLSQP for higher precision."""
    
    # Unwrap once at the start of the IK solve. These native pointers remain 
    # valid throughout the optimization loop as the environment state is 
    # only modified, not re-allocated.
    model_native = mujoco_utils.get_native_mj_object(model)
    data_native = mujoco_utils.get_native_mj_object(data)

    def compute_state(q):
        # Update joint positions directly in native data
        mujoco_utils.set_qpos(data_native, joint_qpos_idx, q)
        
        # Forward kinematics
        mujoco.mj_fwdPosition(model_native, data_native)
        
        # Compute site error using native data
        pos_err, rot_err = mujoco_utils.compute_site_error(model_native, data_native, site_id, target_pos, target_quat)
        err = np.concatenate([pos_err, rot_err])
        
        # Compute Jacobian
        nv = int(model_native.nv)
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(model_native, data_native, jacp, jacr, site_id)
        
        # Extract Jacobian for the active joints
        jac = np.vstack([jacp[:, joint_dof_idx], jacr[:, joint_dof_idx]])
        
        return err, jac

    def objective(q):
        err, _ = compute_state(q)
        return 0.5 * np.dot(err, err)

    def gradient(q):
        err, jac = compute_state(q)
        # Gradient of 0.5 * ||f(q)||^2 is J(q)^T * f(q). 
        # Since error = target - current, d(error)/dq = -J, so grad = -J^T * error.
        return -jac.T @ err

    # Use SLSQP for precise constrained optimization
    res = minimize(
        objective,
        q_init,
        method='SLSQP',
        jac=gradient,
        bounds=list(zip(joint_lo, joint_hi)),
    )

    q_final = res.x
    
    # Final state update and error check
    mujoco_utils.set_qpos(data_native, joint_qpos_idx, q_final)
    mujoco.mj_fwdPosition(model_native, data_native)
    pos_err, rot_err = mujoco_utils.compute_site_error(model_native, data_native, site_id, target_pos, target_quat)
    pos_dist = np.linalg.norm(pos_err)
    rot_dist = np.linalg.norm(rot_err)

    if pos_dist <= pos_tol and rot_dist <= ori_tol:
        logger.debug(f"IK solution found: pos_err={pos_dist:.6f}, rot_err={rot_dist:.6f}")
        return q_final
    
    if not res.success:
        logger.debug(f"IK solver failed to converge: {res.message} (pos_err={pos_dist:.6f}, rot_err={rot_dist:.6f})")
    else:
        logger.debug(f"IK converged but tolerance not met: pos_err={pos_dist:.6f}, rot_err={rot_dist:.6f}")

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
    - Run Scipy-based IK from each seed; keep converged solutions; de-duplicate.

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
        # robosuite wraps the MjModel; use its name->id helper
        site_id = int(model.site_name2id(site_name))

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

        solutions: List[np.ndarray] = []
        
        def run_ik_on_seeds(seeds_list):
            nonlocal solutions
            for q0 in seeds_list:
                q_sol = solve_ik_scipy(
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
                    pos_tol=pos_tol,
                    ori_tol=ori_tol,
                )
                if q_sol is not None:
                    solutions.append(np.asarray(q_sol, dtype=float))
                    solutions = unique_solutions(solutions, atol=unique_atol)
                    if len(solutions) >= int(max_solutions):
                        return True
            return False

        try:
            # Try priority seeds first
            found_priority = run_ik_on_seeds(priority_seeds)
            
            # If no solutions found from priority seeds, try fallback seeds
            if not solutions:
                run_ik_on_seeds(fallback_seeds)
            
            if not solutions:
                logger.debug(f"No IK solutions found for robot {robot_id} at {target_pos}")
                
        finally:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            
            # Unwrap for native call
            mujoco.mj_fwdPosition(mujoco_utils.get_native_mj_object(model), mujoco_utils.get_native_mj_object(data))

        return solutions


def apply_ik_orientation_correction(pose: StampedPose) -> StampedPose:
    """Apply -90 degree Z rotation correction to align grasp with MuJoCo site frame."""
    corrected = pose.copy()
    r_corr = R.from_euler('z', -90, degrees=True)
    r_pose = R.from_quat(corrected.quaternion)
    corrected.quaternion = (r_pose * r_corr).as_quat()
    return corrected
