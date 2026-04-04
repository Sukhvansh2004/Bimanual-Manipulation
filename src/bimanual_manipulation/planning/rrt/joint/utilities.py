"""
Joint-space utilities for the arm RRT planner.

Provides distance, steering, random sampling, and FK-based
end-effector position mapping for Meshcat visualisation.
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from bimanual_manipulation.utils.mujoco_utils import (
    get_eef_site_name,
    set_qpos,
)

logger = logging.getLogger(__name__)


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
