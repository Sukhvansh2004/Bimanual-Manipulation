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
) -> Callable[[StampedPose, int], Optional[np.ndarray]]:
    """Create an IK-solver closure compatible with :func:`~solver.compute_optimal_robot_assignment`.

    The returned function has signature ``(pose: StampedPose, robot_id: int) -> Optional[np.ndarray]``
    and internally delegates to :func:`~planning.utils.ik_utils.enumerate_ik_solutions_for_arm`,
    applying the orientation correction beforehand.

    Args:
        env: The dual-arm simulation environment.
        tf_manager: Transform manager for frame conversions.
        env_lock: Optional reentrant lock guarding the environment.
    """
    from robosuite_franka_manipulation.planning.utils.ik_utils import (
        enumerate_ik_solutions_for_arm,
        apply_ik_orientation_correction,
    )

    def _ik_solver(pose: StampedPose, robot_id: int) -> Optional[np.ndarray]:
        corrected = apply_ik_orientation_correction(pose)
        solutions = enumerate_ik_solutions_for_arm(
            env,
            robot_id=robot_id,
            target_pose=corrected,
            tf_manager=tf_manager,
            env_lock=env_lock,
            max_solutions=1,
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
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Create an inter-arm collision-check closure for handover evaluation.

    The returned function has signature ``(q_arm0, q_arm1) -> collision_free``.
    It temporarily sets both arms' joint configurations in MuJoCo, runs a
    forward pass, checks for inter-arm contacts, and restores the original
    simulation state.

    Args:
        env: The dual-arm simulation environment.
        env_lock: Optional reentrant lock guarding the environment.
    """
    import mujoco
    from robosuite_franka_manipulation.planning.utils.mujoco_utils import (
        get_arm_joint_ids,
        get_joint_qpos_indices,
        get_native_mj_object,
    )
    from robosuite_franka_manipulation.planning.utils.collision_utils import (
        check_inter_arm_collision,
    )

    model = env.sim.model
    data = env.sim.data
    qpos_idx_0 = get_joint_qpos_indices(model, get_arm_joint_ids(model, 0))
    qpos_idx_1 = get_joint_qpos_indices(model, get_arm_joint_ids(model, 1))

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
                return not check_inter_arm_collision(env)
            finally:
                data.qpos[:] = qpos_backup
                data.qvel[:] = qvel_backup
                mujoco.mj_forward(
                    get_native_mj_object(model),
                    get_native_mj_object(data),
                )

    return _check
