"""Disk I/O for trajectory NPZ files.

Loads the object trajectory and per-robot grasp trajectories produced
by the trajectory-generation scripts into typed dataclasses suitable
for :func:`~solver.compute_optimal_robot_assignment`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import numpy as np

from bimanual_manipulation.planning.assignment.models import (
    ObjectTrajectory,
    RobotTrajectory,
    TrajectoryBundle,
)

logger = logging.getLogger(__name__)


def load_all_trajectories(
    data_dir: str,
    max_robot0: int | None = None,
    max_robot1: int | None = None,
) -> TrajectoryBundle:
    """Load all trajectory NPZ files produced by trajectory-generation scripts.

    Expects the directory to contain:
    * ``object_trajectory.npz``
    * ``robot0_trajectory_*.npz``
    * ``robot1_trajectory_*.npz``

    Args:
        data_dir: Path to the directory containing trajectory NPZ files.
        max_robot0: If set, load only the first *max_robot0* robot-0
            trajectories (sorted numerically by filename suffix).
        max_robot1: If set, load only the first *max_robot1* robot-1
            trajectories.

    Returns:
        :class:`TrajectoryBundle` with object and per-robot trajectories.

    Raises:
        FileNotFoundError: If ``object_trajectory.npz`` is missing.
    """
    data_path = Path(data_dir)

    # Object trajectory
    obj_file = data_path / "object_trajectory.npz"
    if not obj_file.exists():
        raise FileNotFoundError(f"Object trajectory not found: {obj_file}")
    obj_data = np.load(obj_file)
    object_trajectory = ObjectTrajectory(
        positions=obj_data["object_positions"],
        quaternions=obj_data["object_quaternions"],
        frame_ids=obj_data["object_frame_ids"],
    )
    logger.info(
        "Loaded object trajectory: %d waypoints, shape %s",
        object_trajectory.n_waypoints,
        object_trajectory.positions.shape,
    )

    def _load_robot_trajectories(
        prefix: str, max_count: int | None = None,
    ) -> List[RobotTrajectory]:
        def _numeric_key(p: Path) -> int:
            """Extract the numeric suffix for natural sort order."""
            m = re.search(r"_(\d+)\.npz$", p.name)
            return int(m.group(1)) if m else 0

        files = sorted(data_path.glob(f"{prefix}_trajectory_*.npz"), key=_numeric_key)
        if max_count is not None:
            files = files[:max_count]
        trajs: List[RobotTrajectory] = []
        for f in files:
            d = np.load(f)
            traj = RobotTrajectory(
                eef_positions=d["eef_positions"],
                eef_quaternions=d["eef_quaternions"],
                eef_frame_ids=d["eef_frame_ids"],
                object_positions=d["object_positions"],
                object_quaternions=d["object_quaternions"],
                object_frame_ids=d["object_frame_ids"],
                is_feasible=d["is_feasible"],
                ik_success=d["ik_success"],
                collision_free=d["collision_free"],
                joint_configs=d["joint_configs"],
                source_path=f.name,
            )
            trajs.append(traj)
        n_feasible = sum(1 for t in trajs if np.all(t.is_feasible))
        logger.info(
            "Loaded %d %s trajectories (%d fully feasible)",
            len(trajs), prefix, n_feasible,
        )
        return trajs

    bundle = TrajectoryBundle(
        object_trajectory=object_trajectory,
        robot0_trajectories=_load_robot_trajectories("robot0", max_robot0),
        robot1_trajectories=_load_robot_trajectories("robot1", max_robot1),
    )
    logger.info(
        "TrajectoryBundle ready: %d segments, %d robot0 + %d robot1 trajectories",
        bundle.n_segments,
        len(bundle.robot0_trajectories),
        len(bundle.robot1_trajectories),
    )
    return bundle
