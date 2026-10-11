"""Assignment result data container.

Provides :class:`AssignmentResult` — the output of
:func:`~solver.compute_optimal_robot_assignment`.

See :mod:`~bimanual_manipulation.planning.assignment.utils` for
NPZ persistence helpers (``save_assignment_result`` /
``load_assignment_result``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AssignmentResult:
    """Outcome of :func:`~solver.compute_optimal_robot_assignment`.

    The execution fields store per-waypoint trajectory data for the
    assigned path so that execution can proceed without loading the
    individual trajectory NPZ files.

    Attributes:
        sequence: Robot id (0 or 1) assigned to each segment.
        trajectory_paths: NPZ filename of the assigned trajectory per
            segment (e.g. ``"robot0_trajectory_6.npz"``).
        total_cost: Optimal DP cost (task + switching).
        task_costs: ``(n_segments,)`` task cost per segment.
        switching_costs: ``(n_segments,)`` switching cost incurred on
            entering each segment (``0`` for segment 0).
        handover_points: Segment indices ``i`` where
            ``sequence[i] != sequence[i - 1]``.
        cost_matrix: ``(n_segments, 2)`` best task cost per robot over all
            of its trajectories (``inf`` when none is feasible).
        trajectory_matrix: ``(n_segments, 2)`` trajectory index achieving
            :attr:`cost_matrix` (``-1`` when none is feasible).

    The remaining fields are documented inline below; joint-config arrays
    contain ``NaN`` rows where no normalised config was available.
    """

    # --- Core assignment outputs ---
    sequence: List[int]               # robot id (0 or 1) per segment
    trajectory_paths: List[str]       # NPZ filename per segment (e.g. "robot0_trajectory_6.npz")
    total_cost: float
    task_costs: np.ndarray            # per-segment task cost
    switching_costs: np.ndarray       # per-segment switching cost
    handover_points: List[int]        # segment indices where handovers occur
    cost_matrix: np.ndarray           # (n_segments, 2) – best-per-robot summary
    trajectory_matrix: np.ndarray     # (n_segments, 2) – best trajectory idx per robot

    # --- Normalised joint configs (assigned path) ---
    normalized_configs: np.ndarray = field(repr=False)
    """``(n_waypoints, DOF)`` joint configs for the active robot at each
    waypoint, with angles normalised to minimise per-joint steps and
    validated against joint limits."""

    # --- Per-waypoint execution data (assigned path) ---
    eef_positions: np.ndarray = field(repr=False)
    """``(n_waypoints, 3)`` end-effector positions along the assigned path."""
    eef_quaternions: np.ndarray = field(repr=False)
    """``(n_waypoints, 4)`` end-effector quaternions along the assigned path."""
    eef_frame_ids: List[str] = field(repr=False)
    """``(n_waypoints,)`` end-effector frame IDs."""
    object_positions: np.ndarray = field(repr=False)
    """``(n_waypoints, 3)`` object positions along the assigned path."""
    object_quaternions: np.ndarray = field(repr=False)
    """``(n_waypoints, 4)`` object quaternions along the assigned path."""
    object_frame_ids: List[str] = field(repr=False)
    """``(n_waypoints,)`` object frame IDs."""
    is_feasible: np.ndarray = field(repr=False)
    """``(n_waypoints,)`` boolean feasibility flag per waypoint."""

    # --- Per-unique-trajectory grasp transforms ---
    grasp_paths: List[str] = field(repr=False)
    """Unique trajectory paths for which grasp transforms are stored."""
    grasp_positions: np.ndarray = field(repr=False)
    """``(n_grasps, 3)`` grasp transform positions (object-in-EE frame)."""
    grasp_quaternions: np.ndarray = field(repr=False)
    """``(n_grasps, 4)`` grasp transform quaternions."""
    grasp_frame_ids: List[str] = field(repr=False)
    """``(n_grasps,)`` grasp transform frame IDs."""

    # --- Handover transfer configs ---
    handover_transfer_configs: np.ndarray = field(repr=False)
    """``(n_handovers, DOF)`` joint configs for the *transferring* arm at
    each handover waypoint.  Indexed parallel to :attr:`handover_points`."""
