"""Handover attempt visualizer for the assignment solver.

Renders each evaluated handover in Meshcat under a structured hierarchy::

    handover_robot{X}_to_robot{Y}/
        seg_{N}/
            traj_{A}_to_{B}/
                transfer/  (robot meshes at transfer configs)
                receiver/  (robot meshes at receiver configs)
                object/    (object mesh at handover pose)

Colours encode the outcome:

* **Green** — feasible handover (collision-free IK).
* **Red**   — IK succeeded but inter-arm collision detected.
* **Orange** — IK failed for one or both arms.

The full MuJoCo scene (tables, shelves, etc.) is also streamed once so
the handover context is visible.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from bimanual_manipulation import constants
from bimanual_manipulation.planning.visualization.robot_mesh import (
    RobotMeshVisualizer,
)
from bimanual_manipulation.planning.visualization.object_mesh import (
    ObjectMeshVisualizer,
)
from bimanual_manipulation.utils.meshcat_mj_utils import MeshcatMjVisualizer

if TYPE_CHECKING:
    from bimanual_manipulation.planning.assignment.cost import HandoverResult

logger = logging.getLogger(__name__)


class HandoverVisualizer:
    """Callable visualizer that plugs into ``handover_callback``.

    Usage::

        hv = HandoverVisualizer(vis, tf_manager, object_name="flatPlate")
        result = compute_optimal_robot_assignment(
            ..., handover_callback=hv,
        )
    """

    def __init__(
        self,
        vis,
        tf_manager,
        object_name: Optional[str] = None,
    ) -> None:
        self._mesh_vis = RobotMeshVisualizer(
            vis=vis,
            tf_manager=tf_manager,
            base_path="handover_debug",
            mesh_sample_ratio=constants.VIS_HANDOVER_MESH_SAMPLE_RATIO,
        )
        self._obj_vis = ObjectMeshVisualizer(
            vis=vis,
            tf_manager=tf_manager,
            base_path="handover_debug",
            mesh_sample_ratio=constants.VIS_HANDOVER_MESH_SAMPLE_RATIO,
        )
        self._vis = vis
        self._object_name = object_name
        self._scene_published = False
        self._tf_manager = tf_manager

    def _publish_scene_once(self) -> None:
        """Stream the full MuJoCo scene to Meshcat (tables, shelves, etc.)."""
        if self._scene_published:
            return
        mj_vis = MeshcatMjVisualizer(self._vis)
        mj_vis.publish_environment(self._tf_manager.env.sim)
        self._scene_published = True
        logger.info("Published MuJoCo scene to Meshcat.")

    # ------------------------------------------------------------------
    # Callable interface  (used as ``handover_callback``)
    # ------------------------------------------------------------------

    def __call__(self, result: HandoverResult) -> None:
        """Visualize a single handover attempt."""
        self._publish_scene_once()

        t_rid = result.transfer_robot_id
        r_rid = result.receiver_robot_id
        seg = result.segment_idx
        pt = result.prev_traj_idx
        ct = result.curr_traj_idx

        parent = f"handover_robot{t_rid}_to_robot{r_rid}"
        seg_tag = f"seg_{seg}"
        traj_tag = f"traj_{pt}_to_{ct}"
        base = f"{parent}/{seg_tag}/{traj_tag}"

        logger.debug(
            "Handover vis: %s  cost=%.3f  feasible=%s  collisions=%d",
            base, result.cost, result.feasible, result.n_collision_steps,
        )

        if not result.feasible and "ik_fail" in result.rejection_reason:
            ik_failed_color = constants.VIS_COLOR_IK_FAILED
        else:
            ik_failed_color = None

        # Render transfer arm configs (if available)
        if result.transfer_configs:
            t_mask = result.transfer_collision_mask
            for i, q in enumerate(result.transfer_configs):
                path = f"{base}/transfer/step_{i:03d}"
                if ik_failed_color is not None:
                    step_color = ik_failed_color
                elif t_mask and i < len(t_mask) and t_mask[i]:
                    step_color = constants.VIS_COLOR_INFEASIBLE
                else:
                    step_color = constants.VIS_COLOR_FEASIBLE
                self._mesh_vis.visualize_robot_at_config(
                    robot_id=t_rid,
                    joint_config=q,
                    path=path,
                    color=step_color,
                    opacity=constants.VIS_MESH_OPACITY,
                )

        # Render receiver arm configs (if available)
        if result.receiver_configs:
            r_mask = result.receiver_collision_mask
            for i, q in enumerate(result.receiver_configs):
                path = f"{base}/receiver/step_{i:03d}"
                if ik_failed_color is not None:
                    step_color = ik_failed_color
                elif r_mask and i < len(r_mask) and r_mask[i]:
                    step_color = constants.VIS_COLOR_INFEASIBLE
                else:
                    step_color = constants.VIS_COLOR_FEASIBLE
                self._mesh_vis.visualize_robot_at_config(
                    robot_id=r_rid,
                    joint_config=q,
                    path=path,
                    color=step_color,
                    opacity=constants.VIS_MESH_OPACITY,
                )

        # Render object mesh at handover pose
        if self._object_name and result.object_pose is not None:
            obj_color = (
                ik_failed_color if ik_failed_color is not None
                else constants.VIS_COLOR_FEASIBLE
            )
            obj_path = f"{base}/object"
            self._obj_vis.visualize_object_at_pose(
                object_name=self._object_name,
                pose=result.object_pose,
                path=obj_path,
                color=obj_color,
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
