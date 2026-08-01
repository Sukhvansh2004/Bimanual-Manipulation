"""MuJoCo utility functions for manipulating the simulation state."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Sequence, Tuple, Optional, Union
import logging
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from robosuite.utils.binding_utils import MjModel, MjData

from bimanual_manipulation import constants

if TYPE_CHECKING:
    from bimanual_manipulation.environments.dual_arm import DualArmEnvironment

logger = logging.getLogger(__name__)

def get_native_mj_object(obj):
    """Helper to unwrap Robosuite MjModel/MjData wrappers to native MuJoCo objects."""
    if hasattr(obj, '_data'):
        return obj._data
    if hasattr(obj, '_model'):
        return obj._model
    logger.debug(f"Object {obj} is not a Robosuite MjModel/MjData wrapper, returning as-is.")
    return obj


def get_arm_joint_ids(model, robot_id: int) -> List[int]:
    """Get the joint IDs for a specific robot arm."""
    prefix = f"robot{robot_id}_joint"
    joint_ids = [j for j in range(model.njnt) if (model.joint(j).name or "").startswith(prefix)]
    # Ensure ordering joint1..joint7
    joint_ids.sort(key=lambda j: int((model.joint(j).name or "").replace(prefix, "") or "0"))
    return joint_ids


def get_joint_qpos_indices(model, joint_ids: Sequence[int]) -> np.ndarray:
    """Get the qpos indices for the given joint IDs."""
    idx: List[int] = []
    for j in joint_ids:
        qposadr = model.joint(j).qposadr
        # robosuite binding_utils returns a length-1 ndarray for *_adr fields
        if isinstance(qposadr, np.ndarray):
            idx.append(int(qposadr.reshape(-1)[0]))
        else:
            idx.append(int(qposadr))
    return np.asarray(idx, dtype=int)


def get_joint_dof_indices(model, joint_ids: Sequence[int]) -> np.ndarray:
    """Get the DOF indices for the given joint IDs."""
    idx: List[int] = []
    for j in joint_ids:
        dofadr = model.joint(j).dofadr
        if isinstance(dofadr, np.ndarray):
            idx.append(int(dofadr.reshape(-1)[0]))
        else:
            idx.append(int(dofadr))
    return np.asarray(idx, dtype=int)


def get_joint_limits(model, joint_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Get the lower and upper joint limits."""
    # model.jnt_range is (njnt, 2)
    jnt_range = np.asarray(model.jnt_range, dtype=float)
    lo = np.asarray([jnt_range[j, 0] for j in joint_ids], dtype=float)
    hi = np.asarray([jnt_range[j, 1] for j in joint_ids], dtype=float)
    return lo, hi


def get_eef_site_name(robot_id: int, arm: str = "right") -> str:
    """Get the end-effector flange site name (where the gripper mounts to the arm)."""
    # ft_frame is the force-torque frame at the robot-gripper flange interface.
    return f"gripper{robot_id}_{arm}_ft_frame"


def get_qpos(data, qpos_idx: np.ndarray) -> np.ndarray:
    """Get the joint positions for the given indices."""
    qpos = np.asarray(data.qpos)
    if qpos.ndim == 2 and qpos.shape[1] == 1:
        return np.asarray(qpos[qpos_idx, 0], dtype=float)
    return np.asarray(qpos[qpos_idx], dtype=float)


def set_qpos(data, qpos_idx: np.ndarray, q: np.ndarray) -> None:
    """Set the joint positions for the given indices."""
    q = np.asarray(q, dtype=float).reshape(-1)
    qpos = data.qpos
    if np.asarray(qpos).ndim == 2:
        qpos[qpos_idx, 0] = q
    else:
        qpos[qpos_idx] = q


def compute_site_error(model, data, site_id: int, target_pos: np.ndarray, target_quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute position and orientation error for a site."""
    cur_pos = np.asarray(data.site_xpos[site_id], dtype=float)
    cur_mat = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    target_rot = R.from_quat(np.asarray(target_quat, dtype=float))
    cur_rot = R.from_matrix(cur_mat)
    # orientation error as rotation vector taking current -> target
    rot_err = (target_rot * cur_rot.inv()).as_rotvec()
    pos_err = np.asarray(target_pos, dtype=float) - cur_pos
    return pos_err, rot_err


def compute_site_error_fast(
    data, site_id: int, target_pos: np.ndarray, R_target: R,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute site error with a pre-constructed ``R_target``.

    Same as ``compute_site_error`` but avoids repeated ``R.from_quat``
    when called in a loop with the same target orientation.
    """
    cur_pos = np.asarray(data.site_xpos[site_id], dtype=float)
    cur_mat = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    cur_rot = R.from_matrix(cur_mat)
    rot_err = (R_target * cur_rot.inv()).as_rotvec()
    pos_err = np.asarray(target_pos, dtype=float) - cur_pos
    return pos_err, rot_err


def geom_belongs_to_robot(geom_name: str) -> Optional[int]:
    """Return robot_id if geom is part of a robot/gripper, else None.
    
    Observed prefixes: "robot0_...", "robot1_...", "gripper0_...", "gripper1_..."
    """
    if geom_name.startswith("robot0_") or geom_name.startswith("gripper0_"):
        return 0
    if geom_name.startswith("robot1_") or geom_name.startswith("gripper1_"):
        return 1
    return None


# ---------------------------------------------------------------------------
# Collision-geom introspection helpers
# ---------------------------------------------------------------------------

def get_robot_collision_geoms(model, robot_id: int) -> List[str]:
    """Return collision geom names for a robot (arm + gripper).

    Excludes visual-only geoms (names containing ``"vis"``).

    Args:
        model: Native MuJoCo model.
        robot_id: Robot index (0 or 1).

    Returns:
        List of geom name strings.
    """
    prefixes = (f"robot{robot_id}_", f"gripper{robot_id}_")
    return [
        model.geom(i).name
        for i in range(model.ngeom)
        if model.geom(i).name.startswith(prefixes)
        and "vis" not in model.geom(i).name
    ]


def get_all_dof_indices_except(model, active_robot_id: int) -> List[int]:
    """Return DOF indices of everything *except* the given robot's arm joints.

    These DOFs are typically frozen in a QP so that only the active
    robot's arm moves.

    This is a convenience wrapper around :func:`get_frozen_dof_indices`
    that resolves the robot's arm DOF indices automatically.

    Args:
        model: Native MuJoCo model.
        active_robot_id: Robot whose joints should be *excluded* from the result.

    Returns:
        List of DOF indices to freeze.
    """
    active_jids = get_arm_joint_ids(model, active_robot_id)
    active_dof_idx = get_joint_dof_indices(model, active_jids)
    return get_frozen_dof_indices(model, active_dof_idx)


def get_object_collision_geoms(model, object_name: Optional[str]) -> List[str]:
    """Return collision geom names whose name contains *object_name*.

    Handles the robosuite naming convention where object geoms are
    prefixed with the object's instance name (e.g. ``mug_0_g0``).

    Args:
        model: Native MuJoCo model.
        object_name: Name substring to search for, or ``None``.

    Returns:
        List of matching geom name strings.
    """
    if not object_name:
        return []
    return [
        model.geom(i).name
        for i in range(model.ngeom)
        if object_name in model.geom(i).name
        and "vis" not in model.geom(i).name
    ]


def get_object_free_joint_qpos_indices(
    model, object_name: Optional[str],
) -> Optional[np.ndarray]:
    """Return the 7 qpos indices of *object_name*'s free joint.

    Robosuite names an object's root body ``<name>_main`` (falling back
    to ``<name>``).  A movable object owns a single free joint whose
    qpos layout is ``[x, y, z, qw, qx, qy, qz]``.

    Args:
        model: Native MuJoCo model.
        object_name: Object instance name (e.g. ``"mug_0"``), or ``None``.

    Returns:
        ``(7,)`` array of qpos indices, or ``None`` when the object is
        unknown or static (no free joint).
    """
    if not object_name:
        return None

    body_id = -1
    for candidate in (f"{object_name}_main", object_name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
        if bid >= 0:
            body_id = bid
            break
    if body_id < 0:
        return None

    for jid in range(model.njnt):
        if (int(model.jnt_bodyid[jid]) == body_id
                and int(model.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_FREE):
            adr = int(model.jnt_qposadr[jid])
            return np.arange(adr, adr + 7)
    return None


def get_environment_collision_geoms(
    model,
    robot_ids: Sequence[int],
    object_name: Optional[str] = None,
) -> List[str]:
    """Return collision geom names for static environment obstacles.

    These are geoms that belong to neither robot nor the manipulated
    object — typically tables, shelves, walls, etc.  Visual-only geoms
    (containing ``"vis"``) and the floor plane are excluded.

    Args:
        model: Native MuJoCo model.
        robot_ids: Robot indices whose geoms should be excluded.
        object_name: Name of the manipulated object to exclude.

    Returns:
        List of environment geom name strings.
    """
    robot_prefixes = tuple(
        p
        for rid in robot_ids
        for p in (f"robot{rid}_", f"gripper{rid}_")
    )
    env_geoms: List[str] = []
    for i in range(model.ngeom):
        name = model.geom(i).name
        if not name:
            continue
        if "vis" in name:
            continue
        if name == "floor":
            continue
        if name.startswith(robot_prefixes):
            continue
        if object_name and object_name in name:
            continue
        env_geoms.append(name)
    return env_geoms


# ---------------------------------------------------------------------------
# Frozen-DOF helper
# ---------------------------------------------------------------------------

def get_frozen_dof_indices(
    model_native,
    active_dof_idx: np.ndarray,
) -> List[int]:
    """Return DOF indices of everything *except* the given active DOFs.

    Used to construct a ``DofFreezingTask`` so that only the target arm
    moves during IK.

    Args:
        model_native: **Native** ``mujoco.MjModel``.
        active_dof_idx: DOF indices for the arm being solved.

    Returns:
        List of DOF indices to freeze.
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


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def compute_site_jacobian(
    model_native: mujoco.MjModel,
    data_native: mujoco.MjData,
    site_id: int,
    arm_dof_idx: np.ndarray,
    arm_qpos_idx: np.ndarray,
    q_arm: np.ndarray,
    full_qpos: np.ndarray,
) -> np.ndarray:
    """Compute the 6×N geometric Jacobian for an arm at a MuJoCo site.

    This is the single canonical implementation used by all IK solvers.
    Sets the arm joints to *q_arm*, runs ``mj_fwdPosition``, then
    extracts the positional and rotational Jacobian columns for the arm
    DOFs.

    Args:
        model_native: Native ``mujoco.MjModel``.
        data_native: Native ``mujoco.MjData``.
        site_id: MuJoCo site ID for the end-effector.
        arm_dof_idx: DOF indices for the arm joints.
        arm_qpos_idx: Qpos indices for the arm joints.
        q_arm: ``(DOF,)`` joint configuration for the arm.
        full_qpos: ``(nq,)`` full qpos vector (will be copied, not mutated).

    Returns:
        ``(6, DOF)`` Jacobian — rows 0-2 are linear, rows 3-5 are angular,
        expressed in the world frame.
    """
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
    """Compute the 6×7 geometric Jacobian at the EEF site for a Franka arm.

    High-level convenience wrapper around :func:`compute_site_jacobian`
    that resolves joint IDs, site IDs, and qpos from a
    :class:`DualArmEnvironment`.

    Args:
        env: The dual-arm simulation environment.
        robot_id: Robot index (0 or 1).
        arm: Arm name (default ``"right"``).
        q_config: Optional ``(7,)`` joint config.  If *None*, uses the
            current simulation state.
        site_name: MuJoCo site name.  If *None*, uses the default EEF site.
        env_lock: Optional threading lock for simulation access.

    Returns:
        ``(6, 7)`` world-frame geometric Jacobian.
    """
    ctx = env_lock if env_lock is not None else nullcontext()
    with ctx:
        model = env.sim.model
        data = env.sim.data

        joint_ids = get_arm_joint_ids(model, robot_id)
        joint_qpos_idx = get_joint_qpos_indices(model, joint_ids)
        joint_dof_idx = get_joint_dof_indices(model, joint_ids)

        if site_name is None:
            site_name = get_eef_site_name(robot_id, arm=arm)

        model_native = get_native_mj_object(model)
        data_native = get_native_mj_object(data)

        site_id = mujoco.mj_name2id(
            model_native, mujoco.mjtObj.mjOBJ_SITE, site_name,
        )
        if site_id < 0:
            raise RuntimeError(f"MuJoCo site '{site_name}' not found in model")

        full_qpos = np.array(data.qpos, copy=True, dtype=float).ravel()
        q_arm = (
            np.asarray(q_config, dtype=float)
            if q_config is not None
            else full_qpos[joint_qpos_idx]
        )

        return compute_site_jacobian(
            model_native, data_native, site_id,
            joint_dof_idx, joint_qpos_idx, q_arm, full_qpos,
        )
