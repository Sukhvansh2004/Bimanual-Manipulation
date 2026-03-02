"""
Generic (domain-agnostic) path post-processing utilities.

Provides three reusable algorithms that are parameterised by callbacks
so that domain-specific modules (SE(3), joint-space, …) can plug in
their own collision checks, distance metrics, and interpolation
strategies:

1. **Shortcutting** (:func:`shortcut_generic`) — iteratively removes
   redundant waypoints by trying to connect non-adjacent ones directly.
2. **Bézier smoothing** (:func:`bezier_smooth_generic`) — fits composite
   cubic Bézier splines per segment with a collision-checked fallback
   to linear interpolation.
3. **Densification** (:func:`densify_generic`) — resamples a sparse path
   to a fixed number of evenly-spaced waypoints using arc-length
   parameterisation.

Domain-specific wrappers:
- SE(3):        :mod:`se3.smoothing`
- Joint-space:  :mod:`joint.smoothing`
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Tuple

import numpy as np

from robosuite_franka_manipulation import constants

logger = logging.getLogger(__name__)


# ============================================================================
# Generic (domain-agnostic) post-processing
# ============================================================================

def shortcut_generic(
    path: list,
    *,
    segment_free_fn: Callable[[Any, Any], bool],
    max_iterations: int,
) -> list:
    """Domain-agnostic random shortcutting.

    Randomly selects two non-adjacent waypoints and removes the
    intermediaries when the direct connection is collision-free.
    This function contains the *loop only*; all domain-specific
    collision checking is delegated to *segment_free_fn*.

    Args:
        path: Ordered list of states from start to goal.
        segment_free_fn: ``(s1, s2) -> bool`` — ``True`` when the
            segment between *s1* and *s2* is collision-free.
        max_iterations: Number of random shortcut attempts.

    Returns:
        A (possibly shorter) list of states.
    """
    smoothed = list(path)
    for _ in range(max_iterations):
        if len(smoothed) <= 2:
            break
        i = np.random.randint(0, len(smoothed) - 1)
        j = np.random.randint(i + 1, len(smoothed))
        if j - i <= 1:
            continue
        if segment_free_fn(smoothed[i], smoothed[j]):
            smoothed = smoothed[: i + 1] + smoothed[j:]
    return smoothed


def densify_generic(
    path: list,
    num_steps: int,
    *,
    distance_fn: Callable[[Any, Any], float],
    interpolate_fn: Callable[[Any, Any, float], Any],
) -> list:
    """Domain-agnostic arc-length densification.

    Resamples a sparse path to *num_steps* evenly-spaced waypoints
    using cumulative arc-length parameterisation.  The metric and
    interpolation method are supplied by the caller.

    Args:
        path: Sparse list of states.
        num_steps: Desired number of output waypoints.
        distance_fn: ``(s1, s2) -> float`` — distance between states.
        interpolate_fn: ``(s1, s2, t) -> state`` — interpolate between
            two states at parameter *t* ∈ [0, 1].

    Returns:
        List of *num_steps* states.

    Raises:
        ValueError: If *num_steps* ≤ 0 or *path* is empty.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if len(path) == 0:
        raise ValueError("path is empty")
    if len(path) == 1:
        return [path[0]] * num_steps

    # Compute cumulative arc-length
    seg_lengths = [
        distance_fn(path[i], path[i + 1]) for i in range(len(path) - 1)
    ]
    total = sum(seg_lengths)
    if total < 1e-12:
        return [path[0]] * num_steps

    cum = np.zeros(len(path))
    for i in range(1, len(path)):
        cum[i] = cum[i - 1] + seg_lengths[i - 1]
    cum /= total

    dense: list = []
    for k in range(num_steps):
        s = k / float(num_steps - 1) if num_steps > 1 else 0.0
        seg_idx = int(
            np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(path) - 2)
        )
        seg_start = cum[seg_idx]
        seg_end = cum[seg_idx + 1]
        t_local = (
            0.0
            if seg_end - seg_start < 1e-10
            else (s - seg_start) / (seg_end - seg_start)
        )
        dense.append(
            interpolate_fn(path[seg_idx], path[seg_idx + 1], t_local)
        )
    return dense


# ============================================================================
# Bézier-Curve Smoothing Helpers
# ============================================================================

def compute_bezier_control_points(
    waypoints: List[np.ndarray],
    tension: float = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Compute cubic Bézier control points for a polyline.

    For each segment ``[P_i, P_{i+1}]`` the four control points are::

        C0 = P_i
        C1 = P_i + tension * tangent_i
        C2 = P_{i+1} - tension * tangent_{i+1}
        C3 = P_{i+1}

    where ``tangent_i`` is the Catmull-Rom tangent at waypoint *i*.

    Args:
        waypoints: List of R^n vectors (positions or any vector space).
        tension: Controls how far out the handles extend.

    Returns:
        List of ``(C0, C1, C2, C3)`` tuples, one per segment.
    """
    n = len(waypoints)
    if n < 2:
        return []

    if tension is None:
        tension = constants.BEZIER_TENSION

    tangents = []
    for i in range(n):
        if i == 0:
            tangents.append(waypoints[1] - waypoints[0])
        elif i == n - 1:
            tangents.append(waypoints[-1] - waypoints[-2])
        else:
            tangents.append(waypoints[i + 1] - waypoints[i - 1])

    segments = []
    for i in range(n - 1):
        C0 = waypoints[i]
        C3 = waypoints[i + 1]

        chord = np.linalg.norm(C3 - C0)
        t_i_norm = np.linalg.norm(tangents[i])
        t_i1_norm = np.linalg.norm(tangents[i + 1])

        C1 = C0 + tension * (chord / t_i_norm) * tangents[i] if t_i_norm > 1e-12 else C0.copy()
        C2 = C3 - tension * (chord / t_i1_norm) * tangents[i + 1] if t_i1_norm > 1e-12 else C3.copy()

        segments.append((C0, C1, C2, C3))
    return segments


def eval_cubic_bezier(C0, C1, C2, C3, t: float):
    """De Casteljau evaluation of a cubic Bézier at parameter *t* in [0, 1]."""
    s = 1.0 - t
    return s**3 * C0 + 3 * s**2 * t * C1 + 3 * s * t**2 * C2 + t**3 * C3


def bezier_smooth_generic(
    path: list,
    *,
    collision_fn: Callable[[Any], bool],
    eval_segment_fn: Callable[[list, int, int], list],
    linear_fallback_fn: Callable[[Any, Any, int], list],
    samples_per_segment: int = None,
    tension: float = None,
) -> list:
    """Domain-agnostic per-segment Bézier smoothing with collision fallback.

    For each segment of the (already shortcut-smoothed) path:

    1. Evaluate a dense Bézier curve via *eval_segment_fn*.
    2. Collision-check every sample via *collision_fn*.
    3. If collision-free → use the smooth segment.
       Otherwise → fall back to *linear_fallback_fn* for that segment.

    Both the SE(3) and joint-space smoothers delegate to this function.

    Args:
        path: Ordered list of waypoints (any domain).
        collision_fn: ``state -> bool`` — ``True`` if the state is in
            collision.
        eval_segment_fn: ``(path, seg_idx, samples_per_segment) -> list``
            — returns ``samples_per_segment + 1`` dense samples for a
            single segment, using the full *path* for tangent context.
        linear_fallback_fn: ``(start, end, num_samples) -> list`` —
            returns ``num_samples + 1`` linearly-interpolated samples
            between two waypoints.
        samples_per_segment: Number of dense samples per segment.
            Defaults to :data:`constants.BEZIER_SAMPLES_PER_SEGMENT`.
        tension: Bézier tension (unused here — forwarded by callers).

    Returns:
        Densely sampled smoothed path.
    """
    if samples_per_segment is None:
        samples_per_segment = constants.BEZIER_SAMPLES_PER_SEGMENT

    if len(path) <= 1:
        return list(path)
    if len(path) == 2:
        return linear_fallback_fn(path[0], path[1], samples_per_segment)

    num_segs = len(path) - 1
    n_ok = 0
    result: list = []

    for seg_idx in range(num_segs):
        bez_pts = eval_segment_fn(path, seg_idx, samples_per_segment)

        seg_collides = False
        for pt in bez_pts:
            if collision_fn(pt):
                seg_collides = True
                break

        if not seg_collides:
            n_ok += 1
            start = 1 if seg_idx > 0 else 0
            result.extend(bez_pts[start:])
        else:
            lin_pts = linear_fallback_fn(
                path[seg_idx], path[seg_idx + 1], samples_per_segment,
            )
            start = 1 if seg_idx > 0 else 0
            result.extend(lin_pts[start:])

    logger.info(
        "Bézier smoothing: %d/%d segments smoothed, "
        "%d fell back to linear (%d total dense waypoints)",
        n_ok, num_segs, num_segs - n_ok, len(result),
    )
    return result
