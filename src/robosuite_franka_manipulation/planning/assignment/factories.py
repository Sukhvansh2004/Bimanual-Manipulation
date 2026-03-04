"""Factory closures for IK, Jacobian, and collision-check callables.

Each factory takes a :class:`DualArmEnvironment` (and optionally a lock)
and returns a closure suitable for passing into the assignment solver.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Callable, Dict, Optional

import numpy as np

from robosuite_franka_manipulation.environments import DualArmEnvironment
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager


def make_ik_solver_fn(
    env: DualArmEnvironment,
    tf_manager: TransformManager,
    env_lock: Optional[threading.RLock] = None,
    relaxed: bool = False,
) -> Callable[[StampedPose, int], Optional[np.ndarray]]:
    """Create an IK-solver closure compatible with :func:`~solver.compute_optimal_robot_assignment`.

    The returned function has signature
    ``(pose: StampedPose, robot_id: int, q_warm_start: Optional[np.ndarray] = None) -> Optional[np.ndarray]``
    and internally delegates to :func:`~planning.utils.ik_utils.enumerate_ik_solutions_for_arm`,
    applying the orientation correction beforehand.

    Args:
        env: The dual-arm simulation environment.
        tf_manager: Transform manager for frame conversions.
        env_lock: Optional reentrant lock guarding the environment.
        relaxed: If True, use relaxed IK tolerances suited for handover
            cost estimation (faster, higher acceptance rate).
    """
    from robosuite_franka_manipulation import constants as C
    from robosuite_franka_manipulation.planning.utils.ik_utils import (
        enumerate_ik_solutions_for_arm,
        apply_ik_orientation_correction,
    )

    pos_tol = C.HANDOVER_IK_POS_TOLERANCE if relaxed else C.IK_POS_TOLERANCE
    ori_tol = C.HANDOVER_IK_ORI_TOLERANCE if relaxed else C.IK_ORI_TOLERANCE

    def _ik_solver(
        pose: StampedPose,
        robot_id: int,
        q_warm_start: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        corrected = apply_ik_orientation_correction(pose)
        solutions = enumerate_ik_solutions_for_arm(
            env,
            robot_id=robot_id,
            target_pose=corrected,
            tf_manager=tf_manager,
            env_lock=env_lock,
            max_solutions=1,
            pos_tol=pos_tol,
            ori_tol=ori_tol,
            q_warm_start=q_warm_start,
        )
        return solutions[0] if solutions else None

    return _ik_solver


def make_jacobian_fns(
    env: DualArmEnvironment,
    env_lock: Optional[threading.RLock] = None,
) -> Dict[int, Callable[[np.ndarray], np.ndarray]]:
    """Create per-robot Jacobian closures for :func:`~solver.compute_optimal_robot_assignment`.

    Returns a dict ``{0: jac_fn_robot0, 1: jac_fn_robot1}`` where each function
    has signature ``(q: np.ndarray) -> np.ndarray`` (6×7 Jacobian) and internally
    delegates to :func:`~planning.utils.ik_utils.compute_arm_jacobian`.

    Args:
        env: The dual-arm simulation environment.
        env_lock: Optional reentrant lock guarding the environment.
    """
    from robosuite_franka_manipulation.planning.utils.ik_utils import compute_arm_jacobian

    def _make_for_robot(robot_id: int) -> Callable[[np.ndarray], np.ndarray]:
        def _jacobian_fn(q: np.ndarray) -> np.ndarray:
            return compute_arm_jacobian(
                env, robot_id=robot_id, q_config=q, env_lock=env_lock,
            )
        return _jacobian_fn

    return {0: _make_for_robot(0), 1: _make_for_robot(1)}


def make_collision_check_fn(
    env: DualArmEnvironment,
    env_lock: Optional[threading.RLock] = None,
    manipulated_object: Optional[str] = None,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Create a collision-check closure for handover evaluation.

    The returned function has signature ``(q_arm0, q_arm1) -> collision_free``.
    It temporarily sets both arms' joint configurations in MuJoCo, runs a
    forward pass, checks for inter-arm **and** arm-environment contacts, and
    restores the original simulation state.

    Contacts involving the *manipulated_object* are ignored so that the
    grasped object does not trigger false positives.

    Args:
        env: The dual-arm simulation environment.
        env_lock: Optional reentrant lock guarding the environment.
        manipulated_object: Geom-name prefix of the manipulated object
            (e.g. ``"plate_0"``, ``"mug_0"``).  Contacts involving geoms
            whose names start with this prefix are ignored.
    """
    import mujoco
    from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
        get_arm_joint_ids,
        get_joint_qpos_indices,
        get_native_mj_object,
    )
    from robosuite_franka_manipulation.planning.utils.collision_utils import (
        check_inter_arm_collision,
        in_collision,
    )

    model = env.sim.model
    data = env.sim.data
    qpos_idx_0 = get_joint_qpos_indices(model, get_arm_joint_ids(model, 0))
    qpos_idx_1 = get_joint_qpos_indices(model, get_arm_joint_ids(model, 1))

    # Pre-compute ignore lists for arm-environment collision checking.
    # Inter-arm contacts are already detected by check_inter_arm_collision,
    # so we exclude them from the in_collision call to avoid double-counting.
    _inter_arm_ignore_pairs = [
        ("robot0_", "robot1_"),
        ("robot0_", "gripper1_"),
        ("gripper0_", "robot1_"),
        ("gripper0_", "gripper1_"),
    ]
    _ignore_names = [manipulated_object] if manipulated_object else None

    def _check(q_arm0: np.ndarray, q_arm1: np.ndarray) -> bool:
        """Return True if the dual-arm configuration is collision-free."""
        ctx = env_lock if env_lock is not None else nullcontext()
        with ctx:
            qpos_backup = np.array(data.qpos, copy=True)
            qvel_backup = np.array(data.qvel, copy=True)
            try:
                data.qpos[qpos_idx_0] = q_arm0
                data.qpos[qpos_idx_1] = q_arm1
                mujoco.mj_forward(
                    get_native_mj_object(model),
                    get_native_mj_object(data),
                )
                # 1. Inter-arm collision (arm0 ↔ arm1)
                if check_inter_arm_collision(env):
                    return False
                # 2. Arm-environment collision (ignore manipulated object
                #    and inter-arm pairs already handled above)
                if in_collision(
                    env,
                    ignore_names=_ignore_names,
                    ignore_pairs=_inter_arm_ignore_pairs,
                ):
                    return False
                return True
            finally:
                data.qpos[:] = qpos_backup
                data.qvel[:] = qvel_backup
                mujoco.mj_forward(
                    get_native_mj_object(model),
                    get_native_mj_object(data),
                )

    return _check


def make_handover_clearance_fn(
    env: DualArmEnvironment,
    clearance_radius: float,
    manipulated_object: Optional[str] = None,
    env_lock: Optional[threading.RLock] = None,
) -> Callable[[StampedPose], bool]:
    """Create an environment-clearance closure for handover evaluation.

    The returned function has signature ``(obj_pose: StampedPose) -> bool``
    and returns ``True`` when the object position is at least
    *clearance_radius* away from every static environment geom centre.
    This is a fast heuristic used to skip handovers at locations where
    environment obstacles would prevent a second arm from reaching in
    (e.g. inside a shelf or rack).

    Args:
        env: The dual-arm simulation environment.
        clearance_radius: Minimum distance (metres) from object centre to
            each environment geom centre for the location to be considered
            *clear* for handover.
        manipulated_object: Name of the manipulated object (excluded from
            the environment geom list).
        env_lock: Optional reentrant lock guarding the environment.
    """
    import mujoco
    from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
        get_environment_collision_geoms,
        get_native_mj_object,
    )

    native_model = get_native_mj_object(env.sim.model)
    native_data = get_native_mj_object(env.sim.data)

    env_geom_names = get_environment_collision_geoms(
        native_model, robot_ids=[0, 1], object_name=manipulated_object,
    )

    # Pre-compute geom ids
    env_geom_ids = [native_model.geom(name).id for name in env_geom_names]

    # Snapshot static environment geom world positions (they never move)
    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        mujoco.mj_forward(native_model, native_data)
        env_geom_positions = (
            np.array([native_data.geom_xpos[gid].copy() for gid in env_geom_ids])
            if env_geom_ids
            else np.empty((0, 3))
        )

    def _clearance_ok(obj_pose: StampedPose) -> bool:
        """Return ``True`` when the object has sufficient clearance."""
        if env_geom_positions.shape[0] == 0:
            return True
        dists = np.linalg.norm(env_geom_positions - obj_pose.position, axis=1)
        return float(dists.min()) > clearance_radius

    return _clearance_ok
