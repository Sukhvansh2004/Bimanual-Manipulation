"""Metrics extraction for planning and execution results.

Planning metrics are computed offline from trajectory data + assignment results
(no simulation required).  Execution metrics require running the handover script.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from bimanual_manipulation.planning.ik.manipulability import (
    compute_trajectory_manipulability,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlanningMetrics:
    """Metrics extracted from the assignment planner (offline)."""

    experiment: str = ""
    config_label: str = ""

    # Assignment solver output
    total_cost: float = float("inf")
    n_segments: int = 0
    n_handovers: int = 0
    handover_points: List[int] = field(default_factory=list)
    mean_task_cost: float = 0.0
    total_switching_cost: float = 0.0

    # Manipulability (computed from joint configs along the assigned path)
    mean_manipulability: float = 0.0
    min_manipulability: float = 0.0

    # Feasibility
    feasibility_rate_robot0: float = 0.0  # fraction of fully-feasible R0 trajectories
    feasibility_rate_robot1: float = 0.0
    assigned_feasibility_rate: float = 0.0  # fraction of assigned segments that are feasible

    # Planning time
    planning_time_s: float = 0.0

    # Gripper / handover direction info (populated from assignment sequence)
    grasp_gripper: str = ""            # "finger" or "suction"
    handover_directions: List[str] = field(default_factory=list)  # e.g. ["finger→suction"]
    sequence: List[int] = field(default_factory=list)  # robot assignment per segment

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecutionMetrics:
    """Metrics from running the dual-arm handover execution."""

    experiment: str = ""
    config_label: str = ""

    task_success: bool = False
    handover_success_rate: float = 0.0  # successful handovers / attempted
    n_handovers_attempted: int = 0
    n_handovers_succeeded: int = 0
    n_handovers_lost_object: int = 0  # handovers where object was lost before attempt
    n_collision_incidents: int = 0
    execution_time_s: float = 0.0

    # Per-phase breakdown
    grasp_success: bool = False
    transport_success: bool = False
    place_success: bool = False

    # Additional metrics from ExecutionTracker
    n_switches: int = 0                     # total arm switches (from assignment)
    mean_manipulability: float = 0.0        # mean manipulability along executed path
    object_mean_tracking_error: float = 0.0
    object_max_tracking_error: float = 0.0
    placement_position_error: float = 0.0   # metres
    placement_orientation_error: float = 0.0  # degrees

    # Per-handover results: list of dicts with keys:
    #   waypoint (int), giver (int), receiver (int), giver_gripper (str),
    #   receiver_gripper (str), direction (str e.g. "finger→suction"),
    #   success (bool), object_held_pre (bool), receiver_holds_post (bool),
    #   object_fallen (bool)
    handover_results: List[Dict] = field(default_factory=list)

    # Per-grasp result: dict with keys:
    #   robot (int), gripper (str e.g. "finger"/"suction"), success (bool)
    grasp_result: Optional[Dict] = None

    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Combined planning + execution metrics for one experiment × config."""

    experiment: str = ""
    config_label: str = ""
    config_id: str = ""
    planning: Optional[PlanningMetrics] = None
    execution: Optional[ExecutionMetrics] = None

    def to_dict(self) -> dict:
        d = {
            "experiment": self.experiment,
            "config_label": self.config_label,
            "config_id": self.config_id,
        }
        if self.planning is not None:
            d["planning"] = self.planning.to_dict()
        if self.execution is not None:
            d["execution"] = self.execution.to_dict()
        return d


# ═══════════════════════════════════════════════════════════════════════════
# Planning metrics extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_planning_metrics(
    data_dir: str,
    assignment_file: str,
    experiment_name: str,
    config_label: str,
    planning_time_s: float = 0.0,
    jacobian_fns: Optional[Dict[int, Callable]] = None,
) -> PlanningMetrics:
    """Extract planning metrics from pre-computed data files.

    Args:
        data_dir: Path to the experiment data directory (with trajectory NPZs).
        assignment_file: Path to the assignment_results.npz file.
        experiment_name: Name of the experiment.
        config_label: Label of the ablation config.
        planning_time_s: Wall-clock time for the assignment solver.
        jacobian_fns: Optional ``{robot_id: (q) -> J}`` for Yoshikawa
            manipulability.  When *None*, falls back to the simple proxy.

    Returns:
        Populated PlanningMetrics.
    """
    metrics = PlanningMetrics(
        experiment=experiment_name,
        config_label=config_label,
        planning_time_s=planning_time_s,
    )

    data_path = Path(data_dir)

    # --- Load assignment result ---
    asgn_path = Path(assignment_file)
    if not asgn_path.exists():
        logger.warning("Assignment file not found: %s", assignment_file)
        return metrics

    asgn = np.load(asgn_path, allow_pickle=True)

    sequence = asgn["sequence"]
    metrics.n_segments = len(sequence)
    metrics.total_cost = float(asgn["total_cost"])
    metrics.handover_points = asgn["handover_points"].tolist()
    metrics.n_handovers = len(metrics.handover_points)
    metrics.mean_task_cost = float(asgn["task_costs"].mean())
    metrics.total_switching_cost = float(asgn["switching_costs"].sum())

    # --- Derive gripper / handover direction info from sequence ---
    _ROBOT_GRIPPER = {0: "finger", 1: "suction"}
    metrics.sequence = sequence.tolist()
    if len(sequence) > 0:
        metrics.grasp_gripper = _ROBOT_GRIPPER.get(int(sequence[0]), "")
    for hp in metrics.handover_points:
        if 0 < hp < len(sequence):
            src = _ROBOT_GRIPPER.get(int(sequence[hp - 1]), "?")
            dst = _ROBOT_GRIPPER.get(int(sequence[hp]), "?")
            metrics.handover_directions.append(f"{src}→{dst}")

    # --- Trajectory paths resolution ---
    if "trajectory_paths" not in asgn.files:
        logger.warning("No trajectory_paths in %s", assignment_file)
        return metrics
    traj_paths = asgn["trajectory_paths"]

    # --- Compute manipulability along the assigned path ---
    manip_values = []
    feasible_count = 0

    for seg_idx in range(len(sequence)):
        tpath = str(traj_paths[seg_idx])
        fpath = data_path / tpath
        if not fpath.exists():
            continue

        traj_data = np.load(fpath)
        joint_configs = traj_data["joint_configs"]
        is_feasible = traj_data["is_feasible"]

        if seg_idx < len(joint_configs) and is_feasible[seg_idx]:
            q = joint_configs[seg_idx].reshape(1, -1)
            robot_id = int(sequence[seg_idx])
            jac_fn = jacobian_fns.get(robot_id) if jacobian_fns else None
            m = compute_trajectory_manipulability(
                q,
                use_jacobian=(jac_fn is not None),
                jacobian_fn=jac_fn,
            )
            manip_values.append(float(m[0]))
            feasible_count += 1

    if manip_values:
        metrics.mean_manipulability = float(np.mean(manip_values))
        metrics.min_manipulability = float(np.min(manip_values))

    metrics.assigned_feasibility_rate = feasible_count / max(metrics.n_segments, 1)

    # --- Per-robot feasibility rates ---
    for robot_id, prefix in [(0, "robot0"), (1, "robot1")]:
        traj_files = sorted(data_path.glob(f"{prefix}_trajectory_*.npz"))
        n_total = len(traj_files)
        n_fully_feasible = 0
        for tf in traj_files:
            td = np.load(tf)
            if np.all(td["is_feasible"]):
                n_fully_feasible += 1
        rate = n_fully_feasible / max(n_total, 1)
        if robot_id == 0:
            metrics.feasibility_rate_robot0 = rate
        else:
            metrics.feasibility_rate_robot1 = rate

    return metrics


def extract_trajectory_generation_metrics(data_dir: str, experiment_name: str) -> dict:
    """Extract trajectory generation statistics (independent of assignment).

    Returns dict with trajectory-level stats useful for the report.
    """
    data_path = Path(data_dir)

    stats = {
        "experiment": experiment_name,
        "n_robot0_trajs": 0,
        "n_robot1_trajs": 0,
        "robot0_fully_feasible": 0,
        "robot1_fully_feasible": 0,
        "robot0_feasibility_rates": [],
        "robot1_feasibility_rates": [],
        "n_object_waypoints": 0,
    }

    # Object trajectory
    obj_file = data_path / "object_trajectory.npz"
    if obj_file.exists():
        obj_data = np.load(obj_file)
        stats["n_object_waypoints"] = len(obj_data["object_positions"])

    for robot_id, prefix in [(0, "robot0"), (1, "robot1")]:
        traj_files = sorted(data_path.glob(f"{prefix}_trajectory_*.npz"))
        n_trajs = len(traj_files)
        n_fully_feasible = 0
        feas_rates = []

        for tf in traj_files:
            td = np.load(tf)
            is_feas = td["is_feasible"]
            rate = float(np.mean(is_feas))
            feas_rates.append(rate)
            if np.all(is_feas):
                n_fully_feasible += 1

        stats[f"n_{prefix}_trajs"] = n_trajs
        stats[f"{prefix}_fully_feasible"] = n_fully_feasible
        stats[f"{prefix}_feasibility_rates"] = feas_rates

    return stats
