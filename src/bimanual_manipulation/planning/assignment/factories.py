"""Factory closures for IK, Jacobian, and collision-check callables.

Each factory takes a :class:`DualArmEnvironment` (and optionally a lock)
and returns a closure suitable for passing into the assignment solver.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import Callable, Dict, Optional

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T

from bimanual_manipulation.environments import DualArmEnvironment
from bimanual_manipulation.utils.collision_utils import (
    check_inter_arm_collision,
    in_collision,
)
from bimanual_manipulation.planning.ik.ik_utils import (
    apply_ik_orientation_correction,
)
from bimanual_manipulation.planning.ik.ik_utils import (
    enumerate_ik_solutions_for_arm,
)
from bimanual_manipulation.planning.utils.mujoco_utils import (
    get_arm_joint_ids,
    get_eef_site_name,
    get_environment_collision_geoms,
    get_joint_dof_indices,
    get_joint_qpos_indices,
    get_native_mj_object,
)
from bimanual_manipulation.transforms import StampedPose, TransformManager


def make_ik_solver_fn(
    env: DualArmEnvironment,
    tf_manager: TransformManager,
    env_lock: Optional[threading.RLock] = None,
) -> Callable[[StampedPose, int], Optional[np.ndarray]]:
    """Create an IK-solver closure compatible with :func:`~solver.compute_optimal_robot_assignment`.

    The returned function has signature
    ``(pose: StampedPose, robot_id: int, q_warm_start: Optional[np.ndarray] = None) -> Optional[np.ndarray]``
    and internally delegates to EAIK analytical IK, applying the orientation
    correction beforehand.

    Args:
        env: The dual-arm simulation environment.
        tf_manager: Transform manager for frame conversions.
        env_lock: Optional reentrant lock guarding the environment.
    """
    def _ik_solver(
        pose: StampedPose,
        robot_id: int,
        q_warm_start: Optional[np.ndarray] = None,
    ) -> tuple:
        """Return ``(solutions_list, residual)``.

        Uses EAIK analytical IK for fast offline solving.
        Returns all valid solutions (ranked by proximity to
        *q_warm_start* when given) so the caller can evaluate
        multiple candidates.  *residual* is 0.0 when at least one
        solution is found, ``inf`` otherwise.
        """
        corrected = apply_ik_orientation_correction(pose)
        solutions = enumerate_ik_solutions_for_arm(
            env,
            robot_id=robot_id,
            target_pose=corrected,
            tf_manager=tf_manager,
            env_lock=env_lock,
            q_reference=q_warm_start,
        )
        if solutions:
            return solutions
        return None

    return _ik_solver


def make_jacobian_fns(
    env: DualArmEnvironment,
    env_lock: Optional[threading.RLock] = None,
) -> Dict[int, Callable[[np.ndarray], np.ndarray]]:
    """Create per-robot Jacobian closures for :func:`~solver.compute_optimal_robot_assignment`.

    Returns a dict ``{0: jac_fn_robot0, 1: jac_fn_robot1}`` where each function
    has signature ``(q: np.ndarray) -> np.ndarray`` (6×7 Jacobian).

    Each closure lazily creates a ``mink.Configuration`` on first call and
    reuses it via ``config.update(q=...)`` to avoid repeated ``MjData``
    allocation (~0.55 ms → 0.01 ms per call).

    Args:
        env: The dual-arm simulation environment.
        env_lock: Optional reentrant lock guarding the environment.
    """
    def _make_for_robot(robot_id: int) -> Callable[[np.ndarray], np.ndarray]:
        _cached: dict = {}  # mutable container for lazy init

        def _jacobian_fn(q: np.ndarray) -> np.ndarray:
            ctx = env_lock if env_lock is not None else nullcontext()
            with ctx:
                if not _cached:
                    model = env.sim.model
                    joint_ids = get_arm_joint_ids(model, robot_id)
                    _cached["qpos_idx"] = get_joint_qpos_indices(model, joint_ids)
                    _cached["dof_idx"] = get_joint_dof_indices(model, joint_ids)
                    _cached["site"] = get_eef_site_name(robot_id, arm="right")
                    model_native = get_native_mj_object(model)
                    _cached["model_native"] = model_native
                    _cached["site_id"] = mujoco.mj_name2id(
                        model_native, mujoco.mjtObj.mjOBJ_SITE, _cached["site"],
                    )

                model_native = _cached["model_native"]
                data_native = get_native_mj_object(env.sim.data)
                q_full = np.array(env.sim.data.qpos, copy=True, dtype=float).ravel()
                q_full[_cached["qpos_idx"]] = q

                data_native.qpos[:] = q_full
                mujoco.mj_fwdPosition(model_native, data_native)

                jacp = np.zeros((3, model_native.nv))
                jacr = np.zeros((3, model_native.nv))
                mujoco.mj_jacSite(
                    model_native, data_native, jacp, jacr, _cached["site_id"],
                )
                return np.vstack([
                    jacp[:, _cached["dof_idx"]],
                    jacr[:, _cached["dof_idx"]],
                ])
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
    and returns ``True`` when every environment-obstacle *surface* is at
    least *clearance_radius* away from the manipulated-object *surface*.

    Unlike a simple centre-to-centre Euclidean check, this temporarily
    repositions the object in MuJoCo, inflates its geom margins to
    *clearance_radius*, runs collision detection, and inspects the
    resulting contact array for mesh-to-mesh proximity — the same
    mechanism used by the suction-weld contact logic.

    Falls back to a centre-distance heuristic when the manipulated object
    has no free joint (cannot be repositioned).

    Args:
        env: The dual-arm simulation environment.
        clearance_radius: Minimum distance (metres) from object surface to
            each environment geom surface for the location to be considered
            *clear* for handover.
        manipulated_object: Name of the manipulated object (excluded from
            the environment geom list).
        env_lock: Optional reentrant lock guarding the environment.
    """
    native_model = get_native_mj_object(env.sim.model)
    native_data = get_native_mj_object(env.sim.data)

    # --- Environment geom IDs (static obstacles) ---
    env_geom_names = get_environment_collision_geoms(
        native_model, robot_ids=[0, 1], object_name=manipulated_object,
    )
    env_geom_id_set = frozenset(
        native_model.geom(name).id for name in env_geom_names
    )

    if not env_geom_id_set or manipulated_object is None:
        # Nothing to collide with — clearance always passes.
        def _always_clear(obj_pose: StampedPose) -> bool:
            return True
        return _always_clear

    # --- Manipulated-object body & geom IDs ---
    obj_body_id = mujoco.mj_name2id(
        native_model, mujoco.mjtObj.mjOBJ_BODY, manipulated_object,
    )
    obj_geom_id_set: set[int] = set()
    for gid in range(native_model.ngeom):
        bid = int(native_model.geom_bodyid[gid])
        # Walk up the kinematic tree to see if this geom belongs to
        # the manipulated object (or any of its child bodies).
        while bid > 0:
            if bid == obj_body_id:
                obj_geom_id_set.add(gid)
                break
            bid = int(native_model.body_parentid[bid])
    obj_geom_id_set = frozenset(obj_geom_id_set)

    # --- Find the object's free-joint qpos address ---
    obj_jnt_qposadr: Optional[int] = None
    for jid in range(native_model.njnt):
        if (int(native_model.jnt_bodyid[jid]) == obj_body_id
                and int(native_model.jnt_type[jid]) == 0):  # mjJNT_FREE
            obj_jnt_qposadr = int(native_model.jnt_qposadr[jid])
            break

    if obj_jnt_qposadr is None or not obj_geom_id_set:
        # Cannot reposition this body — fall back to centre-distance.
        ctx_init = env_lock if env_lock is not None else nullcontext()
        with ctx_init:
            mujoco.mj_forward(native_model, native_data)
            env_geom_positions = (
                np.array([native_data.geom_xpos[gid].copy()
                          for gid in env_geom_id_set])
                if env_geom_id_set else np.empty((0, 3))
            )

        def _clearance_ok_fallback(obj_pose: StampedPose) -> bool:
            if env_geom_positions.shape[0] == 0:
                return True
            dists = np.linalg.norm(
                env_geom_positions - obj_pose.position, axis=1,
            )
            return float(dists.min()) > clearance_radius

        return _clearance_ok_fallback

    # Save original geom margins so we can restore after each probe.
    _orig_margins = {
        gid: float(native_model.geom_margin[gid]) for gid in obj_geom_id_set
    }

    def _clearance_ok(obj_pose: StampedPose) -> bool:
        """Return ``True`` when the object has sufficient mesh-to-mesh clearance."""
        ctx = env_lock if env_lock is not None else nullcontext()
        with ctx:
            # Snapshot simulation state
            qpos_bak = native_data.qpos.copy()
            qvel_bak = native_data.qvel.copy()
            try:
                # Move object to the candidate pose via its free joint.
                # Free-joint qpos layout: [x, y, z, qw, qx, qy, qz]
                native_data.qpos[obj_jnt_qposadr:obj_jnt_qposadr + 3] = (
                    obj_pose.position
                )
                # StampedPose quaternion is xyzw (scipy); MuJoCo wants wxyz.
                native_data.qpos[obj_jnt_qposadr + 3:obj_jnt_qposadr + 7] = (
                    T.convert_quat(obj_pose.quaternion, to="wxyz")
                )

                # Inflate object-geom margins so contacts are generated
                # when surfaces are within clearance_radius.
                for gid in obj_geom_id_set:
                    native_model.geom_margin[gid] = clearance_radius

                # Position kinematics + collision detection (cheaper than
                # full mj_forward — skips velocity / actuation / constraints).
                mujoco.mj_kinematics(native_model, native_data)
                mujoco.mj_comPos(native_model, native_data)
                mujoco.mj_collision(native_model, native_data)

                # Scan contact array for object ↔ environment pairs whose
                # surface distance is less than clearance_radius.
                ncon = int(native_data.ncon)
                for ci in range(ncon):
                    c = native_data.contact[ci]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    is_obj_env = (g1 in obj_geom_id_set and g2 in env_geom_id_set)
                    is_env_obj = (g1 in env_geom_id_set and g2 in obj_geom_id_set)
                    if (is_obj_env or is_env_obj) and float(c.dist) < clearance_radius:
                        return False
                return True
            finally:
                # Restore geom margins
                for gid, m in _orig_margins.items():
                    native_model.geom_margin[gid] = m
                # Restore full simulation state
                native_data.qpos[:] = qpos_bak
                native_data.qvel[:] = qvel_bak
                mujoco.mj_forward(native_model, native_data)

    return _clearance_ok
