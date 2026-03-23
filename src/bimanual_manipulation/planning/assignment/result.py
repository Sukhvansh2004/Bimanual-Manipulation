"""Assignment result data container and NPZ persistence.

Provides :class:`AssignmentResult` — the output of
:func:`~solver.compute_optimal_robot_assignment` — together with
:func:`save_assignment_result` / :func:`load_assignment_result` for
round-trip NPZ serialisation.
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


def save_assignment_result(result: AssignmentResult, filepath: str) -> None:
    """Persist an :class:`AssignmentResult` to an ``.npz`` file."""
    kw = dict(
        sequence=np.array(result.sequence),
        trajectory_paths=np.array(result.trajectory_paths, dtype=str),
        total_cost=result.total_cost,
        task_costs=result.task_costs,
        switching_costs=result.switching_costs,
        handover_points=np.array(result.handover_points),
        cost_matrix=result.cost_matrix,
        trajectory_matrix=result.trajectory_matrix,
    )
    if result.normalized_configs is not None:
        kw["normalized_configs"] = result.normalized_configs
    np.savez(filepath, **kw)
    logger.info("Assignment result saved to %s", filepath)


def load_assignment_result(filepath: str) -> AssignmentResult:
    """Load an :class:`AssignmentResult` from an ``.npz`` file."""
    data = np.load(filepath, allow_pickle=True)
    norm = data["normalized_configs"] if "normalized_configs" in data else None
    return AssignmentResult(
        sequence=data["sequence"].tolist(),
        trajectory_paths=data["trajectory_paths"].tolist(),
        total_cost=float(data["total_cost"]),
        task_costs=data["task_costs"],
        switching_costs=data["switching_costs"],
        handover_points=data["handover_points"].tolist(),
        cost_matrix=data["cost_matrix"],
        trajectory_matrix=data["trajectory_matrix"],
        normalized_configs=norm,
    )
