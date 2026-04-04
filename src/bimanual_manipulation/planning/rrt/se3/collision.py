"""
Object collision checking for the SE(3) RRT planner.

Provides single-pose and pose-setting collision detection for an
object against environment geometry (ignoring robot/gripper contacts).
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.utils.mujoco_utils import get_native_mj_object
from robosuite.utils import transform_utils as T

logger = logging.getLogger(__name__)


def object_in_collision(
    model,
    data,
    object_geom_ids: np.ndarray,
    robot_mask: np.ndarray,
    contact_threshold: float = constants.COLLISION_CONTACT_THRESHOLD,
) -> bool:
    """Check if the object collides with any non-robot geometry.

    Contacts between the object and robot arms/grippers are ignored.

    Args:
        model: MuJoCo model (native or wrapped).
        data: MuJoCo data (native or wrapped).
        object_geom_ids: Array of geom IDs belonging to the object.
        robot_mask: Boolean mask ``(ngeom,)`` — True for robot/gripper geoms
            (obtain via ``mujoco_collision.get_robot_geom_mask``).
        contact_threshold: Contact distance threshold.

    Returns:
        True if the object collides with any environment geometry.
    """
    ncon = int(data.ncon)
    if ncon <= 0:
        return False

    g1 = np.asarray(data.contact.geom1[:ncon])
    g2 = np.asarray(data.contact.geom2[:ncon])
    dist = np.asarray(data.contact.dist[:ncon])

    obj_mask = np.zeros(model.ngeom, dtype=bool)
    obj_mask[object_geom_ids] = True

    active = dist <= contact_threshold
    involves_obj = obj_mask[g1] | obj_mask[g2]
    active &= involves_obj

    other_is_robot = np.where(obj_mask[g1], robot_mask[g2], robot_mask[g1])
    active &= ~other_is_robot

    both_obj = obj_mask[g1] & obj_mask[g2]
    active &= ~both_obj

    return bool(np.any(active))


def check_object_pose_collision(
    env: DualArmEnvironment,
    object_name: str,
    position: np.ndarray,
    quaternion: np.ndarray,
    object_geom_ids: np.ndarray,
    robot_mask: np.ndarray,
    obj_joint_addr: int,
    env_lock=None,
) -> bool:
    """Set the object to a given pose and check for environment collisions.

    Args:
        env: The simulation environment.
        object_name: Name of the object.
        position: 3D position.
        quaternion: Quaternion [x, y, z, w] (scipy convention).
        object_geom_ids: Pre-computed object geom IDs.
        robot_mask: Boolean mask ``(ngeom,)`` for robot/gripper geoms.
        obj_joint_addr: qpos address for the object's free joint.
        env_lock: Optional threading lock.

    Returns:
        True if the object is in collision at this pose.
    """
    model = env.sim.model
    data = env.sim.data

    native_model = get_native_mj_object(model)
    native_data = get_native_mj_object(data)

    # Inflate geom margins so MuJoCo generates contacts for near-miss pairs
    clearance = constants.RRT_SAFETY_CLEARANCE
    orig_margins = native_model.geom_margin[object_geom_ids].copy()
    if clearance > 0:
        native_model.geom_margin[object_geom_ids] = np.maximum(orig_margins, clearance)

    quat_wxyz = T.convert_quat(quaternion, to="wxyz")
    data.qpos[obj_joint_addr:obj_joint_addr + 3] = position
    data.qpos[obj_joint_addr + 3:obj_joint_addr + 7] = quat_wxyz

    try:
        mujoco.mj_fwdPosition(native_model, native_data)

        return object_in_collision(
            model, data, object_geom_ids, robot_mask,
            contact_threshold=max(constants.COLLISION_CONTACT_THRESHOLD, clearance),
        )
    finally:
        native_model.geom_margin[object_geom_ids] = orig_margins
