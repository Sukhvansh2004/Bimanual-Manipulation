"""Disk I/O for trajectory NPZ files.

Loads the object trajectory and per-robot grasp trajectories produced
by the trajectory-generation scripts into a dict suitable for
:func:`~solver.compute_optimal_robot_assignment`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def load_all_trajectories(data_dir: str) -> Dict[str, Any]:
    """Load all trajectory NPZ files produced by trajectory-generation scripts.

    Expects the directory to contain:
    * ``object_trajectory.npz``
    * ``robot0_trajectory_*.npz``
    * ``robot1_trajectory_*.npz``

    Returns a dict compatible with :func:`~solver.compute_optimal_robot_assignment`.
    """
    data_path = Path(data_dir)

    # Object trajectory
    obj_file = data_path / "object_trajectory.npz"
    if not obj_file.exists():
        raise FileNotFoundError(f"Object trajectory not found: {obj_file}")
    obj_data = np.load(obj_file)
    object_trajectory = {
        "positions": obj_data["object_positions"],
        "quaternions": obj_data["object_quaternions"],
        "frame_ids": obj_data["object_frame_ids"],
    }
    logger.info("Loaded object trajectory: %d waypoints", len(object_trajectory["positions"]))

    def _load_robot_trajectories(prefix: str) -> List[Dict[str, Any]]:
        files = sorted(data_path.glob(f"{prefix}_trajectory_*.npz"))
        trajs: List[Dict[str, Any]] = []
        for f in files:
            d = np.load(f)
            trajs.append({
                "eef_positions": d["eef_positions"],
                "eef_quaternions": d["eef_quaternions"],
                "eef_frame_ids": d["eef_frame_ids"],
                "object_positions": d["object_positions"],
                "object_quaternions": d["object_quaternions"],
                "object_frame_ids": d["object_frame_ids"],
                "is_feasible": d["is_feasible"],
                "ik_success": d["ik_success"],
                "collision_free": d["collision_free"],
                "joint_configs": d["joint_configs"],
            })
        logger.info("Loaded %d %s trajectories", len(trajs), prefix)
        return trajs

    return {
        "object_trajectory": object_trajectory,
        "robot0_trajectories": _load_robot_trajectories("robot0"),
        "robot1_trajectories": _load_robot_trajectories("robot1"),
    }
