"""MeshCat visualization for arm trajectories.

Provides trajectory frame visualization using MeshCat for debugging
and monitoring SE(3) trajectory execution.
"""

import logging
from typing import Optional

import meshcat
import meshcat.geometry as g
import numpy as np
from scipy.spatial.transform import Rotation as R

from bimanual_manipulation import constants
from bimanual_manipulation.utils.se3 import pose_to_SE3
from bimanual_manipulation.transforms import TransformManager

try:
    from grasp_gen.utils.meshcat_utils import make_frame
except ImportError:
    make_frame = None

logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Manages MeshCat visualization for a single arm's trajectories.

    Draws under the MeshCat path ``trajectory_robot<robot_id>_<arm_id>``.
    """

    def __init__(
        self,
        robot_id: int,
        arm_id: str,
        tf_manager: TransformManager,
        base_frame: str,
    ) -> None:
        """Initialize the visualizer; the MeshCat connection is created lazily.

        Args:
            robot_id: Robot index.
            arm_id: Arm name (e.g. ``"right"``).
            tf_manager: Used to map ``base_frame`` poses into the world frame.
            base_frame: Frame the trajectory waypoints are expressed in.
        """
        self.robot_id = robot_id
        self.arm_id = arm_id
        self.tf_manager = tf_manager
        self.base_frame = base_frame
        self._meshcat_vis: Optional[meshcat.Visualizer] = None

    def _ensure_visualizer(self) -> meshcat.Visualizer:
        """Connect to MeshCat at ``constants.MESHCAT_ZMQ_URL`` on first use.

        Returns:
            The cached ``meshcat.Visualizer``.
        """
        if self._meshcat_vis is None:
            self._meshcat_vis = meshcat.Visualizer(
                zmq_url=str(constants.MESHCAT_ZMQ_URL)
            )
        return self._meshcat_vis

    def clear(self) -> None:
        """Clear all trajectory visualizations for this arm."""
        vis = self._ensure_visualizer()
        try:
            vis[f"trajectory_robot{self.robot_id}_{self.arm_id}"].delete()
        except Exception:
            pass

    def visualize_se3_trajectory(
        self,
        positions: np.ndarray,
        quaternions: np.ndarray,
        axis_length: float = 0.05,
        axis_radius: float = 0.003,
    ) -> None:
        """Visualize SE(3) trajectory frames in MeshCat.

        Shows coordinate frames along the trajectory path to visualize the
        coupled rotation-translation motion in SE(3) space, plus a polyline
        colored green → blue → red from start to end. Replaces any previous
        trajectory for this arm. No-op for fewer than 2 waypoints or if
        ``make_frame`` is unavailable; falls back to identity (base frame
        drawn as world) if the base-to-world transform cannot be resolved.

        Args:
            positions: Waypoint positions ``(N, 3)`` in robot base frame.
            quaternions: Waypoint quaternions ``(N, 4)`` xyzw in base frame.
            axis_length: Length of coordinate frame axes.
            axis_radius: Radius of coordinate frame axes.
        """
        if len(positions) < 2:
            return

        if make_frame is None:
            logger.debug("make_frame not available, skipping visualization")
            return

        vis = self._ensure_visualizer()
        traj_path = f"trajectory_robot{self.robot_id}_{self.arm_id}"
        root = vis[traj_path]

        try:
            root.delete()
        except Exception:
            pass

        # Transform from robot base to world frame
        try:
            base_to_world = self.tf_manager.get_transform(
                constants.WORLD_FRAME, self.base_frame
            )
            T_base_to_world = base_to_world.to_matrix()
        except Exception as e:
            logger.warning(
                "Could not get base-to-world transform, "
                "visualizing in base frame: %s", e,
            )
            T_base_to_world = np.eye(4)

        n_waypoints = len(positions)
        world_positions = []

        for idx in range(n_waypoints):
            T_pose_in_base = pose_to_SE3(positions[idx], quaternions[idx])
            T_pose_in_world = T_base_to_world @ T_pose_in_base
            world_positions.append(T_pose_in_world[:3, 3])

            if idx == 0:
                frame_name = "start"
            elif idx == n_waypoints - 1:
                frame_name = "end"
            else:
                frame_name = f"waypoint_{idx}"

            frame_path = f"{traj_path}/{frame_name}"
            make_frame(
                vis, name=frame_path,
                h=axis_length, radius=axis_radius, o=0.9,
            )
            vis[frame_path].set_transform(T_pose_in_world)

        # Draw trajectory line
        world_positions = np.array(world_positions)
        line_points = world_positions.T  # (3, N)
        colors = np.zeros((3, n_waypoints))

        for i in range(n_waypoints):
            t = i / max(1, n_waypoints - 1)
            if t < 0.5:
                colors[0, i] = 0
                colors[1, i] = 1 - 2 * t
                colors[2, i] = 2 * t
            else:
                colors[0, i] = 2 * (t - 0.5)
                colors[1, i] = 0
                colors[2, i] = 1 - 2 * (t - 0.5)

        root["path"].set_object(
            g.Line(
                g.PointsGeometry(line_points, colors),
                g.LineBasicMaterial(vertexColors=True, linewidth=2),
            )
        )
