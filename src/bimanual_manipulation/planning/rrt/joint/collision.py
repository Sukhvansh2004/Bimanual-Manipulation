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
from bimanual_manipulation.planning.utils.collision_utils import (
    _ensure_geom_caches,
)
from bimanual_manipulation.planning.utils.mujoco_utils import set_qpos

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
    *,
    contact_threshold: float = constants.COLLISION_CONTACT_THRESHOLD,
) -> bool:
    """Check whether arm B at *q_b* collides with anything.

    Sets both arms' configurations in MuJoCo, runs forward kinematics,
    and scans the contact buffer for contacts involving arm B.

    Contacts between arm A and the environment / object are **not**
    flagged — only contacts where at least one geom belongs to arm B.

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
        contact_threshold: Contact distance threshold.

    Returns:
        ``True`` if arm B has any collision contact.
    """
    data = env.sim.data
    model = env.sim.model

    set_qpos(data, qpos_idx_a, q_a)
    set_qpos(data, qpos_idx_b, q_b)

    # Inflate geom margins so MuJoCo generates contacts for near-miss pairs
    clearance = constants.ARM_RRT_SAFETY_CLEARANCE
    orig_margins = native_model.geom_margin[arm_b_geom_ids].copy()
    if clearance > 0:
        native_model.geom_margin[arm_b_geom_ids] = np.maximum(orig_margins, clearance)
    effective_threshold = max(contact_threshold, clearance)

    try:
        # Use mj_forward instead of mj_fwdPosition to resolve equality 
        # constraints (welds). This is crucial for checking collisions 
        # of the manipulated object while it is being held.
        mujoco.mj_forward(native_model, native_data)

        ncon = int(data.ncon)
        if ncon <= 0:
            return False

        _ensure_geom_caches(model)

        g1 = np.asarray(data.contact.geom1[:ncon])
        g2 = np.asarray(data.contact.geom2[:ncon])
        dist = np.asarray(data.contact.dist[:ncon])

        # Distance filter
        mask = dist <= effective_threshold
        if not np.any(mask):
            return False

        # Only contacts involving arm B
        b_mask = np.zeros(model.ngeom, dtype=bool)
        b_mask[arm_b_geom_ids] = True
        involves_b = b_mask[g1] | b_mask[g2]
        mask &= involves_b

        return bool(np.any(mask))
    finally:
        native_model.geom_margin[arm_b_geom_ids] = orig_margins


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
) -> bool:
    """Check if a straight-line joint-space segment is collision-free.

    Linearly interpolates between *q_start* and *q_end* at *resolution*
    angular increments and checks each for collisions.

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
        resolution: Angular resolution (radians) between checks.
        skip_endpoints: If ``True``, skip checks at exact start/end.

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
        ):
            return False
    return True
