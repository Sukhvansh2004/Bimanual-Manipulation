"""Handover attempt visualizer for the assignment solver.

Renders each evaluated handover in Meshcat under a structured hierarchy::

    handover_robot{X}_to_robot{Y}/
        seg_{N}/
            traj_{A}_to_{B}/
                transfer/  (robot meshes at transfer configs)
                receiver/  (robot meshes at receiver configs)

Colours encode the outcome:

* **Green** — feasible handover (collision-free IK).
* **Red**   — IK succeeded but inter-arm collision detected.
* **Orange** — IK failed for one or both arms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bimanual_manipulation import constants
from bimanual_manipulation.planning.visualization.robot_mesh import (
    RobotMeshVisualizer,
)

if TYPE_CHECKING:
    from bimanual_manipulation.planning.assignment.cost import HandoverResult

logger = logging.getLogger(__name__)


class HandoverVisualizer:
    """Callable visualizer that plugs into ``handover_callback``.

    Usage::

        hv = HandoverVisualizer(vis, tf_manager)
        result = compute_optimal_robot_assignment(
            ..., handover_callback=hv,
        )
    """

    def __init__(
        self,
        vis,
        tf_manager,
    ) -> None:
        self._mesh_vis = RobotMeshVisualizer(
            vis=vis,
            tf_manager=tf_manager,
            base_path="handover_debug",
            mesh_sample_ratio=constants.VIS_HANDOVER_MESH_SAMPLE_RATIO,
        )
        self._vis = vis

    # ------------------------------------------------------------------
    # Callable interface  (used as ``handover_callback``)
    # ------------------------------------------------------------------

    def __call__(self, result: HandoverResult) -> None:
        """Visualize a single handover attempt."""
        t_rid = result.transfer_robot_id
        r_rid = result.receiver_robot_id
        seg = result.segment_idx
        pt = result.prev_traj_idx
        ct = result.curr_traj_idx

        parent = f"handover_robot{t_rid}_to_robot{r_rid}"
        seg_tag = f"seg_{seg}"
        traj_tag = f"traj_{pt}_to_{ct}"
        base = f"{parent}/{seg_tag}/{traj_tag}"

        color, label = self._outcome_color(result)

        logger.debug(
            "Handover vis: %s  outcome=%s  cost=%.3f",
            base, label, result.cost,
        )

        # Render transfer arm configs (if available)
        if result.transfer_configs:
            for i, q in enumerate(result.transfer_configs):
                path = f"{base}/transfer/step_{i:03d}"
                self._mesh_vis.visualize_robot_at_config(
                    robot_id=t_rid,
                    joint_config=q,
                    path=path,
                    color=color,
                    opacity=constants.VIS_MESH_OPACITY,
                )

        # Render receiver arm configs (if available)
        if result.receiver_configs:
            for i, q in enumerate(result.receiver_configs):
                path = f"{base}/receiver/step_{i:03d}"
                self._mesh_vis.visualize_robot_at_config(
                    robot_id=r_rid,
                    joint_config=q,
                    path=path,
                    color=color,
                    opacity=constants.VIS_MESH_OPACITY,
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _outcome_color(result: HandoverResult):
        """Return ``(hex_color, label)`` based on outcome."""
        if not result.feasible:
            if "ik_fail" in result.rejection_reason:
                return constants.VIS_COLOR_IK_FAILED, "ik_failed"
            return constants.VIS_COLOR_INFEASIBLE, "infeasible"
        if result.n_collision_steps > 0:
            return constants.VIS_COLOR_INFEASIBLE, "collision"
        return constants.VIS_COLOR_FEASIBLE, "feasible"
