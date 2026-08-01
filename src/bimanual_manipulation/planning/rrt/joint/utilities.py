"""
Joint-space utilities for the arm RRT planner.

Provides distance, steering, random sampling, temporal-sample
construction, and FK-based end-effector position mapping for Meshcat
visualisation.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import mujoco
import numpy as np

from bimanual_manipulation.utils.mujoco_utils import (
    get_eef_site_name,
    set_qpos,
)

logger = logging.getLogger(__name__)


def build_temporal_samples(
    q_other_arm: np.ndarray,
    q_other_arm_trajectory: Optional[np.ndarray],
    max_temporal_samples: int,
    object_qpos_trajectory: Optional[np.ndarray] = None,
) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Build the ``(q_a, object_qpos)`` snapshots for temporal checking.

    When *q_other_arm_trajectory* is given, arm A is not a static
    obstacle — it sweeps through the workspace.  Rather than checking
    arm B against a single arm A configuration, the planner checks it
    against a fixed number of configurations sampled uniformly along
    the trajectory.

    *object_qpos_trajectory* carries the pose of the object **arm A is
    holding** at those same instants (free-joint qpos, ``[x, y, z, qw,
    qx, qy, qz]``).  Without it the object would stay frozen at its
    current pose while arm A teleports along its trajectory, so arm B
    could happily plan straight through the carried object.

    Args:
        q_other_arm: Static fallback configuration of arm A ``(7,)``.
        q_other_arm_trajectory: Optional ``(N, 7)`` arm A trajectory.
        max_temporal_samples: Upper bound on the number of snapshots.
        object_qpos_trajectory: Optional ``(N, 7)`` object free-joint
            qpos trajectory, index-aligned with
            *q_other_arm_trajectory*.

    Returns:
        List of ``(q_a, object_qpos)`` pairs; ``object_qpos`` is
        ``None`` when no object trajectory was supplied.
    """
    if q_other_arm_trajectory is not None and len(q_other_arm_trajectory) > 1:
        n_traj = len(q_other_arm_trajectory)
        if n_traj <= max_temporal_samples:
            indices = np.arange(n_traj)
        else:
            indices = np.unique(
                np.linspace(0, n_traj - 1, max_temporal_samples, dtype=int)
            )
        q_a_samples = [np.asarray(q_other_arm_trajectory[i]) for i in indices]
    else:
        indices = None
        q_a_samples = [np.asarray(q_other_arm)]

    if object_qpos_trajectory is None or len(object_qpos_trajectory) == 0:
        return [(q_a, None) for q_a in q_a_samples]

    obj_traj = np.asarray(object_qpos_trajectory)
    if indices is None:
        # Arm A is a static obstacle at the *end* of its motion, so the
        # object sits at its final carried pose.
        obj_samples = [obj_traj[-1]]
    elif len(obj_traj) == len(q_other_arm_trajectory):
        obj_samples = [obj_traj[i] for i in indices]
    else:
        # Different sampling rates — map by normalised time.
        src = np.linspace(0, len(obj_traj) - 1, len(q_other_arm_trajectory))
        obj_samples = [obj_traj[int(round(src[i]))] for i in indices]

    return list(zip(q_a_samples, obj_samples))


def joint_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """L2 distance in joint space."""
    return float(np.linalg.norm(q1 - q2))


def steer(
    from_q: np.ndarray,
    to_q: np.ndarray,
    step_size: float,
) -> np.ndarray:
    """Steer from *from_q* toward *to_q*, limited by *step_size*.

    Returns *to_q* directly if the distance is within *step_size*.
    """
    diff = to_q - from_q
    dist = np.linalg.norm(diff)
    if dist <= step_size:
        return to_q.copy()
    return from_q + (step_size / dist) * diff


def sample_random_config(
    joint_lo: np.ndarray,
    joint_hi: np.ndarray,
) -> np.ndarray:
    """Uniformly sample a random joint configuration within limits."""
    return np.random.uniform(joint_lo, joint_hi)


def make_ee_pos_fn(data, qpos_idx_b, native_model, native_data, robot_id_b):
    """Return a callable ``q -> pos`` mapping joint configs to 3-D EE positions.

    Uses the end-effector *site* registered for the given robot.  The
    returned function sets ``qpos``, runs :func:`mujoco.mj_fwdPosition`,
    and reads back the site position.  **Must** be called within a sim
    state backup / restore context.

    Returns ``None`` if the site cannot be resolved.
    """
    site_name = get_eef_site_name(robot_id_b)
    try:
        # Validate that the site exists
        native_model.site(site_name)
    except (KeyError, ValueError):
        logger.warning(
            "EE site '%s' not found for robot%d — "
            "visualisation will be limited.",
            site_name, robot_id_b,
        )
        return None

    def _ee_pos(q: np.ndarray) -> np.ndarray:
        set_qpos(data, qpos_idx_b, q)
        mujoco.mj_fwdPosition(native_model, native_data)
        return np.array(native_data.site(site_name).xpos, copy=True)

    return _ee_pos
