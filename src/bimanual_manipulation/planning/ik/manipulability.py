"""
Manipulability computation utilities for robot configurations.

Manipulability measures how well-conditioned the robot is at a given
configuration.  Higher values indicate better dexterity and greater
distance from singularities.

Provides:
    * ``compute_manipulability_simple`` — fast proxy (no robot model needed).
    * ``compute_manipulability_from_jacobian`` — Yoshikawa measure
      :math:`w = \\sqrt{\\det(J J^T)}`.
    * ``compute_trajectory_manipulability`` / ``compute_average_manipulability``
      — batch wrappers.
"""

import numpy as np
from typing import Callable, Optional

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single-configuration measures
# ---------------------------------------------------------------------------

def compute_manipulability_simple(joint_config: np.ndarray) -> float:
    """Compute a fast proxy manipulability for a joint configuration.

    Combines (a) distance from joint limits, (b) joint angle variance,
    and (c) penalty for extreme configurations.  Does **not** require the
    full robot model.

    Args:
        joint_config: Joint configuration array of shape ``(n_joints,)``.

    Returns:
        Manipulability value (higher is better, always > 0).
    """
    normalized = np.arctan2(np.sin(joint_config), np.cos(joint_config))

    # Distance from ±π limits
    limit_margin = np.minimum(
        np.abs(normalized + np.pi),
        np.abs(normalized - np.pi),
    )
    limit_score = np.mean(limit_margin) / np.pi

    # Joint distribution (variance) – sigmoid-normalized
    variance_score = 1.0 / (1.0 + np.exp(-np.var(normalized)))

    # Penalty for near-limit configurations
    extreme_score = 1.0 - np.mean(np.abs(normalized) > 0.9 * np.pi)

    return max(0.4 * limit_score + 0.3 * variance_score + 0.3 * extreme_score, 1e-6)


def compute_manipulability_from_jacobian(jacobian: np.ndarray) -> float:
    """Compute Yoshikawa manipulability from a Jacobian matrix.

    :math:`w = \\prod(\\sigma_i)` where :math:`\\sigma_i` are the
    singular values of *J*.  This is equivalent to
    :math:`\\sqrt{\\det(J\\,J^T)}` but avoids the matrix multiply and
    determinant by using SVD directly.

    Args:
        jacobian: Jacobian matrix of shape ``(m, n)``.

    Returns:
        Manipulability value (≥ 0).
    """
    s = np.linalg.svd(jacobian, compute_uv=False)
    return max(float(np.prod(s)), 1e-6)


# ---------------------------------------------------------------------------
# Trajectory-level wrappers
# ---------------------------------------------------------------------------

def compute_trajectory_manipulability(
    joint_configs: np.ndarray,
    use_jacobian: bool = False,
    jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Compute per-waypoint manipulability for an entire trajectory.

    Args:
        joint_configs: Array of shape ``(n_waypoints, n_joints)``.
        use_jacobian: If *True*, use ``jacobian_fn`` for the Yoshikawa measure.
        jacobian_fn: ``(q) -> J`` — maps a joint config to its Jacobian.

    Returns:
        Array of manipulability values, shape ``(n_waypoints,)``.
    """
    n_waypoints = joint_configs.shape[0]
    values = np.zeros(n_waypoints)

    for i in range(n_waypoints):
        if use_jacobian and jacobian_fn is not None:
            values[i] = compute_manipulability_from_jacobian(jacobian_fn(joint_configs[i]))
        else:
            values[i] = compute_manipulability_simple(joint_configs[i])

    return values


def compute_average_manipulability(
    joint_configs: np.ndarray,
    use_jacobian: bool = False,
    jacobian_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> float:
    """Compute the mean manipulability across a trajectory.

    Args:
        joint_configs: Array of shape ``(n_waypoints, n_joints)``.
        use_jacobian: If *True*, use ``jacobian_fn`` for the Yoshikawa measure.
        jacobian_fn: ``(q) -> J`` — maps a joint config to its Jacobian.

    Returns:
        Average manipulability value.
    """
    return float(np.mean(
        compute_trajectory_manipulability(joint_configs, use_jacobian, jacobian_fn)
    ))
