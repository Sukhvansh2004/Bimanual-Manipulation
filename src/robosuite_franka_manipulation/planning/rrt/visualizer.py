"""
Meshcat visualization helper for RRT tree growth and path debugging.

All drawing is done under ``vis["rrt_debug/"]`` so it can be cleared
easily between runs.

Colours
-------
* **cyan** / **orange** — tree_a / tree_b edges
* **red** — collision-rejected samples
* **green** — final (raw) path
* **yellow** — smoothed path
* **white sphere** — start,  **magenta sphere** — goal
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import meshcat.geometry as mg

logger = logging.getLogger(__name__)

# Default position extractor: SE(3) tuple -> position component.
_DEFAULT_POS_FN: Callable[[Any], np.ndarray] = lambda elem: elem[0]


class RRTVisualizer:
    """Draws RRT debug information into a Meshcat scene."""

    def __init__(self, vis, delay: float = 0.0):
        """
        Args:
            vis: A ``meshcat.Visualizer`` instance.
            delay: Seconds to sleep after each draw call (useful for
                   watching the search in real-time).  0 = no delay.
        """
        self._vis = vis
        self._delay = delay
        self._root = vis["rrt_debug"]
        self._edge_counter = 0
        self._collision_counter = 0
        self._root.delete()

    # -- helpers ----------------------------------------------------------

    def _sleep(self):
        if self._delay > 0:
            time.sleep(self._delay)

    @staticmethod
    def _color_int(r: int, g: int, b: int) -> int:
        return (r << 16) | (g << 8) | b

    def _draw_sphere(self, name: str, pos: np.ndarray, radius: float, color: int):
        self._root[name].set_object(
            mg.Sphere(radius),
            mg.MeshPhongMaterial(color=color, opacity=0.9),
        )
        tf = np.eye(4)
        tf[:3, 3] = pos
        self._root[name].set_transform(tf)

    def _draw_line(self, name: str, p1: np.ndarray, p2: np.ndarray, color: int):
        pts = np.column_stack([p1, p2]).astype(np.float32)
        self._root[name].set_object(
            mg.LineSegments(
                mg.PointsGeometry(pts),
                mg.LineBasicMaterial(color=color, linewidth=1),
            )
        )

    # -- public API -------------------------------------------------------

    def draw_start_goal(self, start_pos: np.ndarray, goal_pos: np.ndarray):
        """Draw start (white) and goal (magenta) spheres."""
        self._draw_sphere("start", start_pos, 0.015, 0xffffff)
        self._draw_sphere("goal", goal_pos, 0.015, 0xff00ff)
        self._sleep()

    def draw_tree_edge(
        self, parent_pos: np.ndarray, child_pos: np.ndarray, is_tree_a: bool,
    ):
        """Draw a tree edge (cyan for tree_a, orange for tree_b)."""
        color = self._color_int(0, 200, 255) if is_tree_a else self._color_int(255, 165, 0)
        name = f"edges/e{self._edge_counter}"
        self._edge_counter += 1
        self._draw_line(name, parent_pos, child_pos, color)
        self._sleep()

    def draw_collision_sample(self, pos: np.ndarray):
        """Draw a small red dot for a rejected (collision) sample."""
        name = f"collisions/c{self._collision_counter}"
        self._collision_counter += 1
        self._draw_sphere(name, pos, 0.005, 0xff0000)

    def draw_path(
        self,
        path,
        pos_fn: Optional[Callable[[Any], np.ndarray]] = None,
    ):
        """Draw the final raw path as thick green line segments.

        Args:
            path: List of states (any type supported by *pos_fn*).
            pos_fn: Callable that extracts a 3-D position from a path
                element.  Defaults to ``lambda elem: elem[0]`` (works
                for SE(3) ``(pos, quat)`` tuples).
        """
        if len(path) < 2:
            return
        if pos_fn is None:
            pos_fn = _DEFAULT_POS_FN
        for i in range(len(path) - 1):
            pts = np.column_stack(
                [pos_fn(path[i]), pos_fn(path[i + 1])]
            ).astype(np.float32)
            self._root[f"path/seg{i}"].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(pts),
                    mg.LineBasicMaterial(color=0x00ff00, linewidth=4),
                )
            )
        self._sleep()

    def draw_smoothed_path(
        self,
        path,
        pos_fn: Optional[Callable[[Any], np.ndarray]] = None,
    ):
        """Draw the smoothed path as thick yellow line segments.

        Args:
            path: List of states.
            pos_fn: Position extractor (see :meth:`draw_path`).
        """
        if len(path) < 2:
            return
        if pos_fn is None:
            pos_fn = _DEFAULT_POS_FN
        for i in range(len(path) - 1):
            pts = np.column_stack(
                [pos_fn(path[i]), pos_fn(path[i + 1])]
            ).astype(np.float32)
            self._root[f"smoothed_path/seg{i}"].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(pts),
                    mg.LineBasicMaterial(color=0xffff00, linewidth=4),
                )
            )
        self._sleep()

    def draw_shortcut_path(
        self,
        path,
        pos_fn: Optional[Callable[[Any], np.ndarray]] = None,
    ):
        """Draw the shortcut (Phase 1) path as thin orange line segments.

        Args:
            path: List of states.
            pos_fn: Position extractor (see :meth:`draw_path`).
        """
        if len(path) < 2:
            return
        if pos_fn is None:
            pos_fn = _DEFAULT_POS_FN
        for i in range(len(path) - 1):
            pts = np.column_stack(
                [pos_fn(path[i]), pos_fn(path[i + 1])]
            ).astype(np.float32)
            self._root[f"shortcut_path/seg{i}"].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(pts),
                    mg.LineBasicMaterial(color=self._color_int(255, 165, 0), linewidth=2),
                )
            )
        for i, elem in enumerate(path):
            self._draw_sphere(
                f"shortcut_path/wp{i}", pos_fn(elem), 0.008,
                self._color_int(255, 165, 0),
            )
        self._sleep()

    def draw_bezier_control_polygon(
        self,
        segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ):
        """Draw Bézier control polygons: grey lines with blue/red handles."""
        for seg_idx, (C0, C1, C2, C3) in enumerate(segments):
            prefix = f"bezier_ctrl/seg{seg_idx}"
            for j, (pa, pb) in enumerate([(C0, C1), (C1, C2), (C2, C3)]):
                pts = np.column_stack([pa, pb]).astype(np.float32)
                self._root[f"{prefix}/line{j}"].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(pts),
                        mg.LineBasicMaterial(color=0x888888, linewidth=1),
                    )
                )
            self._draw_sphere(f"{prefix}/c1", C1, 0.006, self._color_int(80, 180, 255))
            self._draw_sphere(f"{prefix}/c2", C2, 0.006, self._color_int(255, 100, 100))
        self._sleep()

    def draw_bezier_curve(
        self,
        path: List[Tuple[np.ndarray, np.ndarray]],
        collision_free: bool = True,
    ):
        """Draw the Bézier-smoothed curve (cyan if clean, red if collisions)."""
        if len(path) < 2:
            return
        color = self._color_int(0, 255, 220) if collision_free else 0xff0000
        for i in range(len(path) - 1):
            pts = np.column_stack([path[i][0], path[i + 1][0]]).astype(np.float32)
            self._root[f"bezier_curve/seg{i}"].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(pts),
                    mg.LineBasicMaterial(color=color, linewidth=3),
                )
            )
        self._sleep()

    def draw_bounds(self, pos_bounds: np.ndarray):
        """Draw the workspace bounding box as wireframe."""
        lo, hi = pos_bounds[0], pos_bounds[1]
        corners = np.array([
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ])
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for idx, (a, b) in enumerate(edges):
            pts = np.column_stack([corners[a], corners[b]]).astype(np.float32)
            self._root[f"bounds/e{idx}"].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(pts),
                    mg.LineBasicMaterial(color=0x888888, linewidth=1),
                )
            )

    def log_iteration(self, iteration: int, tree_a_size: int, tree_b_size: int):
        """Periodically log progress (every 100 iterations)."""
        if iteration % 100 == 0 and iteration > 0:
            logger.info(
                "  RRT iteration %d: tree_a=%d, tree_b=%d",
                iteration, tree_a_size, tree_b_size,
            )
