"""Assignment result data container.

Provides :class:`AssignmentResult` — the output of
:func:`~solver.compute_optimal_robot_assignment`.

See :mod:`~bimanual_manipulation.planning.utils.assignment_utils` for
NPZ persistence helpers (``save_assignment_result`` /
``load_assignment_result``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AssignmentResult:
    """Outcome of :func:`~solver.compute_optimal_robot_assignment`."""

    sequence: List[int]               # robot id (0 or 1) per segment
    trajectory_paths: List[str]       # NPZ filename per segment (e.g. "robot0_trajectory_6.npz")
    total_cost: float
    task_costs: np.ndarray            # per-segment task cost
    switching_costs: np.ndarray       # per-segment switching cost
    handover_points: List[int]        # segment indices where handovers occur
    cost_matrix: np.ndarray           # (n_segments, 2) – best-per-robot summary
    trajectory_matrix: np.ndarray     # (n_segments, 2) – best trajectory idx per robot
    normalized_configs: Optional[np.ndarray] = field(default=None, repr=False)
    """``(n_waypoints, DOF)`` joint configs for the active robot at each
    waypoint, with angles normalised to minimise per-joint steps and
    validated against joint limits.  ``None`` when not computed."""
