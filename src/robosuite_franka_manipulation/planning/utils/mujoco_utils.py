"""MuJoCo utility functions for manipulating the simulation state."""

from typing import List, Sequence, Tuple, Optional, Union
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from robosuite.utils.binding_utils import MjModel, MjData

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
    """Get the end-effector site name."""
    # In this repo's robosuite configs we use the single "right" arm.
    # Site naming convention from env inspection: gripper{robot_id}_right_grip_site
    return f"gripper{robot_id}_{arm}_grip_site"


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


def geom_belongs_to_robot(geom_name: str) -> Optional[int]:
    """Return robot_id if geom is part of a robot/gripper, else None.
    
    Observed prefixes: "robot0_...", "robot1_...", "gripper0_...", "gripper1_..."
    """
    if geom_name.startswith("robot0_") or geom_name.startswith("gripper0_"):
        return 0
    if geom_name.startswith("robot1_") or geom_name.startswith("gripper1_"):
        return 1
    return None
