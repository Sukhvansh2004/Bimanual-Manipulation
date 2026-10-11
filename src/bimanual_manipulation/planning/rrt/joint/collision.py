"""
Joint-space collision checking for the arm RRT planner.

Provides single-configuration and segment-level collision detection
for one arm (arm B) while the other arm (arm A) is fixed.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.utils.mujoco_collision import (
    ensure_geom_caches,
)
from bimanual_manipulation.utils.mujoco_utils import set_qpos

logger = logging.getLogger(__name__)


def check_arm_config_collision(
    env: DualArmEnvironment,
    q_b: np.ndarray,
    qpos_idx_b: np.ndarray,
    q_a: np.ndarray,
    qpos_idx_a: np.ndarray,
    native_model,
    native_data,
    arm_b_geom_ids: np.ndarray,
    obj_qpos_idx: np.ndarray = None,
    obj_qpos: np.ndarray = None,
    *,
    contact_threshold: float = constants.COLLISION_CONTACT_THRESHOLD,
) -> bool:
    """Check whether arm B at *q_b* collides with anything.

    Sets both arms' configurations (and optionally the object pose) in
    MuJoCo, runs ``mj_forward``, and scans the contact buffer for
    contacts involving arm B.

    Contacts between arm A and the environment / object are **not**
    flagged — only contacts where at least one geom belongs to arm B
    (including arm B self-contacts).

    The caller is responsible for state backup/restore.

    Args:
        env: Simulation environment.
        q_b: Candidate joint configuration for arm B ``(7,)``.
        qpos_idx_b: qpos indices of arm B's joints.
        q_a: Fixed joint configuration of arm A ``(7,)``.
        qpos_idx_a: qpos indices of arm A's joints.
        native_model: Low-level ``mujoco.MjModel``.
        native_data: Low-level ``mujoco.MjData``.
        arm_b_geom_ids: Array of geom IDs belonging to arm B.
        obj_qpos_idx: Optional qpos indices of the manipulated object's
            free joint ``(7,)``.
        obj_qpos: Optional free-joint qpos of the object carried by
            arm A at this instant.  Teleporting arm A does not drag a
            friction-grasped object along, so the pose has to be
            imposed explicitly for arm B to see it as an obstacle.
        contact_threshold: A contact counts when its ``dist`` is at most
            this value. Defaults to :data:`constants.COLLISION_CONTACT_THRESHOLD`.

    Returns:
        ``True`` if arm B has any collision contact.
    """
    data = env.sim.data
    model = env.sim.model

    set_qpos(data, qpos_idx_a, q_a)
    set_qpos(data, qpos_idx_b, q_b)
    if obj_qpos_idx is not None and obj_qpos is not None:
        set_qpos(data, obj_qpos_idx, obj_qpos)

    # Use mj_forward instead of mj_fwdPosition to resolve equality 
    # constraints (welds). This is crucial for checking collisions 
    # of the manipulated object while it is being held.
    mujoco.mj_forward(native_model, native_data)

    ncon = int(data.ncon)
    if ncon <= 0:
        return False

    ensure_geom_caches(model)

    g1 = np.asarray(data.contact.geom1[:ncon])
    g2 = np.asarray(data.contact.geom2[:ncon])
    dist = np.asarray(data.contact.dist[:ncon])

    # Distance filter
    mask = dist <= contact_threshold
    if not np.any(mask):
        return False

    # Only contacts involving arm B
    b_mask = np.zeros(model.ngeom, dtype=bool)
    b_mask[arm_b_geom_ids] = True
    involves_b = b_mask[g1] | b_mask[g2]
    mask &= involves_b

    return bool(np.any(mask))


def collision_free_joint_segment(
    env: DualArmEnvironment,
    q_start: np.ndarray,
    q_end: np.ndarray,
    q_a: np.ndarray,
    qpos_idx_a: np.ndarray,
    qpos_idx_b: np.ndarray,
    native_model,
    native_data,
    arm_b_geom_ids: np.ndarray,
    resolution: float = None,
    skip_endpoints: bool = False,
    obj_qpos_idx: np.ndarray = None,
    obj_qpos: np.ndarray = None,
) -> bool:
    """Check if a straight-line joint-space segment is collision-free.

    Linearly interpolates between *q_start* and *q_end* in
    ``n = max(2, ceil(||q_end - q_start|| / resolution))`` intervals and
    checks the configurations at ``t = i / n`` for ``i = 0..n``
    (``i = 1..n-1`` when *skip_endpoints* is set). The caller is
    responsible for state backup/restore.

    Args:
        env: Simulation environment.
        q_start: Start joint configuration ``(7,)``.
        q_end: End joint configuration ``(7,)``.
        q_a: Fixed arm A configuration ``(7,)``.
        qpos_idx_a: qpos indices of arm A's joints.
        qpos_idx_b: qpos indices of arm B's joints.
        native_model: ``mujoco.MjModel``.
        native_data: ``mujoco.MjData``.
        arm_b_geom_ids: Geom IDs belonging to arm B.
        resolution: Maximum joint-space L2 distance (radians) between
            checks. Defaults to :data:`constants.ARM_RRT_COLLISION_RESOLUTION`
            when ``None``.
        skip_endpoints: If ``True``, skip checks at exact start/end.
        obj_qpos_idx: Optional qpos indices of the manipulated object's
            free joint ``(7,)``.
        obj_qpos: Optional free-joint qpos of the object carried by
            arm A at this instant.

    Returns:
        ``True`` if the segment is collision-free.
    """
    if resolution is None:
        resolution = constants.ARM_RRT_COLLISION_RESOLUTION

    diff = q_end - q_start
    dist = np.linalg.norm(diff)
    n_checks = max(2, int(np.ceil(dist / resolution)))

    start_idx = 1 if skip_endpoints else 0
    end_idx = n_checks - 1 if skip_endpoints else n_checks

    for i in range(start_idx, end_idx + 1):
        t = i / float(n_checks)
        q_interp = q_start + t * diff
        if check_arm_config_collision(
            env, q_interp, qpos_idx_b, q_a, qpos_idx_a,
            native_model, native_data, arm_b_geom_ids,
            obj_qpos_idx, obj_qpos,
        ):
            return False
    return True
