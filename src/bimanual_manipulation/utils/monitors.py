"""Execution monitors for collision detection and object tracking.

Provides :class:`CollisionMonitor` and :class:`ObjectTrajectoryMonitor`
which are called by :func:`~bimanual_manipulation.utils.env_utils.env_step`
after each simulation step to passively track collisions and object poses.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from bimanual_manipulation import constants
from bimanual_manipulation.environments.dual_arm import DualArmEnvironment
from bimanual_manipulation.utils.mujoco_collision import in_collision

logger = logging.getLogger(__name__)


# =============================================================================
# Collision Monitor
# =============================================================================


class CollisionMonitor:
    """Counts physics-based collision incidents during execution.

    When enabled, after every ``env_step`` the monitor checks MuJoCo contacts
    for inter-arm collisions AND arm-environment collisions (ignoring the
    manipulated object and any configured ``ignore_pairs``).

    Contacts with penetration depth shallower than
    ``min_penetration_depth`` are also excluded — this filters out light
    touches (e.g. grippers barely touching during handover) that do not
    represent dangerous collisions.

    Usage::

        start_collision_monitor("plate_0",
                                ignore_pairs=[("gripper0", "gripper1")])
        # ... execute trajectory ...
        n = stop_collision_monitor()
    """

    def __init__(self):
        self._enabled = False
        self._collision_count = 0
        self._manipulated_object: Optional[str] = None
        self._ignore_pairs: Optional[List[Tuple[str, str]]] = None
        self._min_penetration_depth: float = constants.COLLISION_MONITOR_MIN_DEPTH

    # ── public API ──────────────────────────────────────────────────────

    def start(
        self,
        manipulated_object: Optional[str] = None,
        ignore_pairs: Optional[List[Tuple[str, str]]] = None,
        min_penetration_depth: Optional[float] = None,
    ):
        """Enable monitoring and reset the counter.

        Args:
            manipulated_object: Geom-name prefix to ignore entirely.
            ignore_pairs: List of ``(prefix_a, prefix_b)`` geom-prefix
                pairs whose mutual contacts are ignored.
            min_penetration_depth: Minimum MuJoCo penetration depth
                (negative ``contact.dist``) to count.  ``None`` keeps
                the default from ``constants``.
        """
        self._enabled = True
        self._collision_count = 0
        self._manipulated_object = manipulated_object
        self._ignore_pairs = ignore_pairs
        if min_penetration_depth is not None:
            self._min_penetration_depth = min_penetration_depth
        else:
            self._min_penetration_depth = constants.COLLISION_MONITOR_MIN_DEPTH

    def stop(self) -> int:
        """Disable monitoring and return the accumulated count."""
        self._enabled = False
        count = self._collision_count
        return count

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def collision_count(self) -> int:
        return self._collision_count

    # ── internal ────────────────────────────────────────────────────────

    def check(self, env: DualArmEnvironment):
        """Called by ``env_step`` after each simulation step."""
        if not self._enabled:
            return
        ignore_names = [self._manipulated_object] if self._manipulated_object else None
        if in_collision(
            env,
            ignore_names=ignore_names,
            ignore_pairs=self._ignore_pairs,
            contact_dist_threshold=-self._min_penetration_depth,
        ):
            self._collision_count += 1


# Global collision monitor instance
_collision_monitor = CollisionMonitor()


def start_collision_monitor(
    manipulated_object: Optional[str] = None,
    ignore_pairs: Optional[List[Tuple[str, str]]] = None,
    min_penetration_depth: Optional[float] = None,
):
    """Enable collision monitoring in ``env_step``.

    Args:
        manipulated_object: Geom-name prefix of the grasped object to ignore.
        ignore_pairs: List of ``(prefix_a, prefix_b)`` pairs whose mutual
            contacts are ignored.
        min_penetration_depth: Minimum penetration depth (m) to count.
    """
    _collision_monitor.start(
        manipulated_object,
        ignore_pairs=ignore_pairs,
        min_penetration_depth=min_penetration_depth,
    )


def stop_collision_monitor() -> int:
    """Disable monitoring and return the number of steps that had collisions."""
    return _collision_monitor.stop()


def get_collision_count() -> int:
    """Return the current collision count without stopping the monitor."""
    return _collision_monitor.collision_count


# =============================================================================
# Object Trajectory Monitor
# =============================================================================


class ObjectTrajectoryMonitor:
    """Tracks the manipulated object's actual trajectory vs the planned one.

    When enabled (via :func:`start_object_monitor`), ``env_step`` calls
    :meth:`check` after every simulation step.  Every
    ``constants.OBJECT_MONITOR_LOG_INTERVAL`` steps the monitor emits a
    DEBUG log with the object's current position and the error relative
    to the nearest planned waypoint.

    On :meth:`stop`, returns the full recorded actual trajectory, the
    planned trajectory, and aggregate error statistics.
    """

    def __init__(self):
        self._enabled = False
        self._object_name: Optional[str] = None
        self._step_count = 0
        self._planned_positions: Optional[np.ndarray] = None
        self._actual_positions: List[np.ndarray] = []

    def start(
        self,
        object_name: str,
        planned_positions: Optional[np.ndarray] = None,
    ):
        """Enable monitoring for *object_name*.

        Args:
            object_name: MuJoCo object name (e.g. ``"flatPlate"``).
            planned_positions: Optional ``(N, 3)`` array of planned
                object xyz positions.
        """
        self._enabled = True
        self._object_name = object_name
        self._step_count = 0
        self._planned_positions = planned_positions
        self._actual_positions = []

    def stop(self) -> dict:
        """Disable monitoring and return a summary dict.

        Returns a dict with:
            actual_trajectory: (M, 3) recorded positions.
            planned_trajectory: (N, 3) planned positions (or None).
            mean_tracking_error: mean nearest-waypoint error (m).
            max_tracking_error: max nearest-waypoint error (m).
            total_steps: number of env steps recorded.
        """
        self._enabled = False

        actual = np.array(self._actual_positions) if self._actual_positions else np.empty((0, 3))
        planned = self._planned_positions

        mean_err = 0.0
        max_err = 0.0
        if planned is not None and len(planned) > 0 and len(actual) > 0:
            errors = []
            for pos in actual:
                dists = np.linalg.norm(planned - pos, axis=1)
                errors.append(float(np.min(dists)))
            mean_err = float(np.mean(errors))
            max_err = float(np.max(errors))

        summary = {
            "actual_trajectory": actual,
            "planned_trajectory": planned,
            "mean_tracking_error": mean_err,
            "max_tracking_error": max_err,
            "total_steps": self._step_count,
        }

        if planned is not None and len(planned) > 0 and len(actual) > 0:
            logger.info(
                "Object monitor summary: mean_err=%.4f m, max_err=%.4f m "
                "over %d steps.",
                mean_err, max_err, self._step_count,
            )
        else:
            logger.info(
                "Object monitor summary: %d steps recorded (no planned trajectory for comparison).",
                self._step_count,
            )
        return summary

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check(self, env: DualArmEnvironment):
        """Called by ``env_step`` after each simulation step."""
        if not self._enabled or self._object_name is None:
            return
        self._step_count += 1

        pose = env.get_object_pose(self._object_name)
        if pose is None:
            return
        pos = pose[:3, 3].copy()
        self._actual_positions.append(pos)

        if self._step_count % constants.OBJECT_MONITOR_LOG_INTERVAL == 0:
            msg = "Object '%s' step %d: pos=[%.4f, %.4f, %.4f]"
            args: list = [
                self._object_name, self._step_count,
                pos[0], pos[1], pos[2],
            ]
            if self._planned_positions is not None and len(self._planned_positions) > 0:
                dists = np.linalg.norm(self._planned_positions - pos, axis=1)
                nearest_idx = int(np.argmin(dists))
                nearest_err = float(dists[nearest_idx])
                msg += ", nearest_planned_wp=%d (err=%.4f m)"
                args.extend([nearest_idx, nearest_err])
            logger.debug(msg, *args)


# Global object trajectory monitor instance
_object_monitor = ObjectTrajectoryMonitor()


def start_object_monitor(
    object_name: str,
    planned_positions: Optional[np.ndarray] = None,
):
    """Enable object trajectory monitoring in ``env_step``.

    Args:
        object_name: Name of the manipulated object.
        planned_positions: Optional ``(N, 3)`` planned xyz trajectory.
    """
    _object_monitor.start(object_name, planned_positions)


def stop_object_monitor() -> dict:
    """Disable monitoring and return a summary dict."""
    return _object_monitor.stop()
