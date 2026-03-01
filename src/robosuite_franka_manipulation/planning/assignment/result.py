"""Assignment result data container and NPZ persistence.

Provides :class:`AssignmentResult` — the output of
:func:`~solver.compute_optimal_robot_assignment` — together with
:func:`save_assignment_result` / :func:`load_assignment_result` for
round-trip NPZ serialisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AssignmentResult:
    """Outcome of :func:`~solver.compute_optimal_robot_assignment`."""

    sequence: List[int]               # robot id (0 or 1) per segment
    trajectory_indices: List[int]     # grasp-trajectory index per segment
    total_cost: float
    task_costs: np.ndarray            # per-segment task cost
    switching_costs: np.ndarray       # per-segment switching cost
    handover_points: List[int]        # segment indices where handovers occur
    cost_matrix: np.ndarray           # (n_segments, 2) – best-per-robot summary
    trajectory_matrix: np.ndarray     # (n_segments, 2) – best trajectory idx per robot


def save_assignment_result(result: AssignmentResult, filepath: str) -> None:
    """Persist an :class:`AssignmentResult` to an ``.npz`` file."""
    np.savez(
        filepath,
        sequence=np.array(result.sequence),
        trajectory_indices=np.array(result.trajectory_indices),
        total_cost=result.total_cost,
        task_costs=result.task_costs,
        switching_costs=result.switching_costs,
        handover_points=np.array(result.handover_points),
        cost_matrix=result.cost_matrix,
        trajectory_matrix=result.trajectory_matrix,
    )
    logger.info("Assignment result saved to %s", filepath)


def load_assignment_result(filepath: str) -> AssignmentResult:
    """Load an :class:`AssignmentResult` from an ``.npz`` file."""
    data = np.load(filepath, allow_pickle=True)
    return AssignmentResult(
        sequence=data["sequence"].tolist(),
        trajectory_indices=data["trajectory_indices"].tolist(),
        total_cost=float(data["total_cost"]),
        task_costs=data["task_costs"],
        switching_costs=data["switching_costs"],
        handover_points=data["handover_points"].tolist(),
        cost_matrix=data["cost_matrix"],
        trajectory_matrix=data["trajectory_matrix"],
    )
