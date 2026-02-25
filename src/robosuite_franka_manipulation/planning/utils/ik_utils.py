"""Inverse Kinematics (IK) utilities."""

from contextlib import nullcontext
import threading
from typing import List, Optional, Sequence, Union, Iterable

import numpy as np
import mujoco
import logging
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.spatial.transform import Rotation as R

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.environments.dual_arm import DualArmEnvironment
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager
from robosuite.utils.binding_utils import MjModel, MjData

from robosuite_franka_manipulation.planning.utils import mujoco_utils

logger = logging.getLogger(__name__)

ArrayLike = Union[Sequence[float], np.ndarray]


def _manipulability_from_jac(jac: np.ndarray) -> float:
    """Yoshikawa manipulability: sqrt(det(J @ J^T)).

    A robust formulation using singular values to avoid numerical issues
    with near-singular Jacobians.
    """
    sv = np.linalg.svd(jac, compute_uv=False)
    return float(np.prod(sv))  # == sqrt(det(J @ J^T))


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
    manipulability_weight: float = constants.IK_MANIPULABILITY_WEIGHT,
    joint_limit_weight: float = constants.IK_JOINT_LIMIT_WEIGHT,
    use_analytical_grad: bool = constants.USE_GRAD_SCIPY_IK,
) -> Optional[np.ndarray]:
    """Residual-based IK using scipy.optimize.minimize with SLSQP for higher precision.

    Includes a manipulability regulariser that penalises configurations near
    singularities (small det(J J^T)), and a log-barrier joint-limit
    penalty that pushes the solution away from the edges of the joint
    range, steering the solver toward better-conditioned solutions.
    """
    # Pre-compute joint-range midpoints and half-widths for the barrier.
    joint_range = joint_hi - joint_lo  # full range per joint
    
    # Unwrap once at the start of the IK solve. These native pointers remain 
    # valid throughout the optimization loop as the environment state is 
    # only modified, not re-allocated.
    model_native = mujoco_utils.get_native_mj_object(model)
    data_native = mujoco_utils.get_native_mj_object(data)

    # Cache the last FK evaluation so that constraint functions called at
    # the same q as the objective don't trigger redundant forward-kinematics.
    _cache: dict = {'key': None, 'err': None, 'jac': None}

    def compute_state(q):
        key = q.tobytes()
        if _cache['key'] != key:
            mujoco_utils.set_qpos(data_native, joint_qpos_idx, q)
            mujoco.mj_fwdPosition(model_native, data_native)

            pos_err, rot_err = mujoco_utils.compute_site_error(
                model_native, data_native, site_id, target_pos, target_quat)
            err = np.concatenate([pos_err, rot_err])

            nv = int(model_native.nv)
            jacp = np.zeros((3, nv))
            jacr = np.zeros((3, nv))
            mujoco.mj_jacSite(model_native, data_native, jacp, jacr, site_id)
            jac = np.vstack([jacp[:, joint_dof_idx], jacr[:, joint_dof_idx]])

            _cache['key'] = key
            _cache['err'] = err
            _cache['jac'] = jac

        return _cache['err'], _cache['jac']

    def _joint_limit_barrier(q):
        """Soft log-barrier: -sum[ log(q-lo) + log(hi-q) ].

        Grows to +inf as any joint approaches its limit, gently
        pushing the optimiser toward the centre of the joint range.
        A small epsilon avoids log(0) if the solver probes the boundary.
        """
        eps = 1e-8
        d_lo = np.maximum(q - joint_lo, eps)
        d_hi = np.maximum(joint_hi - q, eps)
        # Normalise by joint range so wide and narrow joints contribute
        # equally, then sum the per-joint penalties.
        return -np.sum(np.log(d_lo / joint_range) + np.log(d_hi / joint_range))

    def _compute_jac_for_q(q_eval):
        """Lightweight FK + Jacobian (no cache) for finite-difference probes."""
        mujoco_utils.set_qpos(data_native, joint_qpos_idx, q_eval)
        mujoco.mj_fwdPosition(model_native, data_native)
        nv = int(model_native.nv)
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(model_native, data_native, jacp, jacr, site_id)
        return np.vstack([jacp[:, joint_dof_idx], jacr[:, joint_dof_idx]])

    def _objective_impl(q, with_grad=False):
        """Unified objective (and optional gradient) computation.

        When *with_grad* is False, returns the scalar cost.
        When *with_grad* is True, returns ``(cost, grad)`` for use
        with ``scipy.optimize.minimize(..., jac=True)``.

        Gradient components (all analytical except manipulability):
          - Task cost:      J^T @ err
          - Joint-limit barrier: -1/(q-lo) + 1/(hi-q)
          - Manipulability: forward finite-difference on the scalar mu
            (7 extra FK calls) since MuJoCo does not expose dJ/dq.
        """
        err, jac = compute_state(q)

        # ---- Cost ----
        cost = 0.5 * np.dot(err, err)
        manip = 0.0
        if manipulability_weight > 0:
            manip = _manipulability_from_jac(jac)
            cost += manipulability_weight * (-np.log(manip + 1e-12))
        if joint_limit_weight > 0:
            cost += joint_limit_weight * _joint_limit_barrier(q)

        if not with_grad:
            return cost

        # ---- Gradient ----
        # 1) Task cost: d/dq [0.5 ||e||^2] = J^T @ e
        grad = - jac.T @ err

        # 2) Manipulability: d/dq [-w * log(mu + eps)]
        #    = -w / (mu + eps) * d_mu/dq
        #    d_mu/dq is approximated via forward finite-difference.
        if manipulability_weight > 0:
            h = 1e-7
            manip_grad = np.zeros(len(q))
            for i in range(len(q)):
                q_p = q.copy()
                q_p[i] += h
                jac_p = _compute_jac_for_q(q_p)
                manip_p = _manipulability_from_jac(jac_p)
                manip_grad[i] = (manip_p - manip) / h
            grad += manipulability_weight * (-1.0 / (manip + 1e-12)) * manip_grad
            # Restore FK state & invalidate cache after FD probes
            _cache['key'] = None
            mujoco_utils.set_qpos(data_native, joint_qpos_idx, q)
            mujoco.mj_fwdPosition(model_native, data_native)

        # 3) Joint-limit barrier (fully analytical):
        #    d/dq_j B = -1/(q_j - lo_j) + 1/(hi_j - q_j)
        if joint_limit_weight > 0:
            eps_b = 1e-8
            d_lo = np.maximum(q - joint_lo, eps_b)
            d_hi = np.maximum(joint_hi - q, eps_b)
            grad += joint_limit_weight * (-1.0 / d_lo + 1.0 / d_hi)

        return cost, grad
    
    # 1. Position Constraint Functions
    def pos_err_norm(q):
        err, _ = compute_state(q)
        return np.linalg.norm(err[:3])

    def pos_err_jac(q):
        """Analytical Jacobian of the position error norm to prevent finite-difference cache misses."""
        err, jac = compute_state(q)
        norm = np.linalg.norm(err[:3])
        if norm < 1e-10:  # Prevent division by zero if perfectly on target
            return np.zeros_like(q)
        return -(err[:3] @ jac[:3, :]) / norm

    # 2. Orientation Constraint Functions
    def ori_err_norm(q):
        err, _ = compute_state(q)
        return np.linalg.norm(err[3:])

    def ori_err_jac(q):
        """Analytical Jacobian of the orientation error norm."""
        err, jac = compute_state(q)
        norm = np.linalg.norm(err[3:])
        if norm < 1e-10:
            return np.zeros_like(q)
        return -(err[3:] @ jac[3:, :]) / norm

    # Define NonlinearConstraints: -infinity <= error_norm <= tolerance
    constraints = [
        NonlinearConstraint(
            fun=pos_err_norm, 
            lb=-np.inf, 
            ub=pos_tol, 
            jac=pos_err_jac if use_analytical_grad else '2-point'
        ),
        NonlinearConstraint(
            fun=ori_err_norm, 
            lb=-np.inf, 
            ub=ori_tol, 
            jac=ori_err_jac if use_analytical_grad else '2-point'
        ),
    ]

    # Use SLSQP for precise, fast constrained optimization.
    res = minimize(
        lambda q: _objective_impl(q, with_grad=use_analytical_grad),
        q_init,
        method='SLSQP',
        jac=use_analytical_grad,
        bounds=Bounds(joint_lo, joint_hi),
        constraints=constraints,
        options={'ftol': 1e-6, 'disp': False}
    )

    q_final = res.x

    mujoco_utils.set_qpos(data_native, joint_qpos_idx, q_final)
    mujoco.mj_fwdPosition(model_native, data_native)
    pos_err, rot_err = mujoco_utils.compute_site_error(model_native, data_native, site_id, target_pos, target_quat)
    pos_dist = np.linalg.norm(pos_err)
    rot_dist = np.linalg.norm(rot_err)

    if not res.success:
        logger.debug(f"IK solver failed to converge: {res.message} (pos_err={pos_dist:.6f}, rot_err={rot_dist:.6f})")
        return None
    
    return q_final


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

    The Jacobian maps 7-DOF joint velocities to 6-D spatial velocities
    ``[v_x, v_y, v_z, ω_x, ω_y, ω_z]`` expressed in the **world frame**,
    evaluated at the ``ft_frame`` site (the flange where the gripper mounts).

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
        site_id = int(model.site_name2id(site_name))

        model_native = mujoco_utils.get_native_mj_object(model)
        data_native = mujoco_utils.get_native_mj_object(data)

        # Optionally set a custom joint configuration
        restore = False
        if q_config is not None:
            qpos_backup = np.array(data.qpos, copy=True)
            qvel_backup = np.array(data.qvel, copy=True)
            mujoco_utils.set_qpos(data_native, joint_qpos_idx, q_config)
            mujoco.mj_fwdPosition(model_native, data_native)
            restore = True

        nv = int(model_native.nv)
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(model_native, data_native, jacp, jacr, site_id)

        J = np.vstack([jacp[:, joint_dof_idx], jacr[:, joint_dof_idx]])  # (6, 7)

        if restore:
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_fwdPosition(model_native, data_native)

        return J


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

        # ------------------------------------------------------------------
        # Rank solutions by manipulability and reject near-singular ones.
        # This ensures the best-conditioned configuration is tried first in
        # downstream collision checks, and prevents warm-starting from
        # singularities that cause trajectory jumps.
        # ------------------------------------------------------------------
        if solutions:
            model_native = mujoco_utils.get_native_mj_object(model)
            data_native = mujoco_utils.get_native_mj_object(data)
            nv = int(model_native.nv)

            scored: List[tuple] = []  # (manipulability, solution)
            for q_sol in solutions:
                mujoco_utils.set_qpos(data_native, joint_qpos_idx, q_sol)
                mujoco.mj_fwdPosition(model_native, data_native)
                jacp = np.zeros((3, nv))
                jacr = np.zeros((3, nv))
                mujoco.mj_jacSite(model_native, data_native, jacp, jacr, site_id)
                jac = np.vstack([jacp[:, joint_dof_idx], jacr[:, joint_dof_idx]])
                m = _manipulability_from_jac(jac)
                scored.append((m, q_sol))

            # Filter out near-singular solutions
            min_manip = constants.IK_MIN_MANIPULABILITY
            scored = [(m, q) for m, q in scored if m >= min_manip]

            # Sort descending by manipulability (best first)
            scored.sort(key=lambda x: x[0], reverse=True)
            solutions = [q for _, q in scored]

            if scored:
                logger.debug(
                    "IK solutions ranked by manipulability: best=%.4e, worst=%.4e (%d solutions)",
                    scored[0][0], scored[-1][0], len(scored),
                )
            else:
                logger.debug("All IK solutions rejected (manipulability < %.2e)", min_manip)

            # Restore simulation state after scoring
            data.qpos[:] = qpos_backup
            data.qvel[:] = qvel_backup
            mujoco.mj_fwdPosition(model_native, data_native)

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
