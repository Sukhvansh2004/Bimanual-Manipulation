"""
RRT-Connect planner — generic core.

Provides a **domain-agnostic** bi-directional RRT-Connect implementation
(``rrt_connect_generic``) parameterised by callback functions.

Domain-specific specialisations live in their own modules:

* **SE(3)** — :mod:`planning.rrt.se3_planner` (``rrt_connect_se3``)
* **Joint-space** — :mod:`planning.rrt.joint_planner` (``rrt_connect_joint_space``)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Generic RRT-Connect Core
# ============================================================================

@dataclass
class _TreeNode:
    """Internal tree node used by :func:`rrt_connect_generic`."""

    state: Any
    parent: int = -1


def _generic_nearest(
    tree: List[_TreeNode],
    state: Any,
    distance_fn: Callable[[Any, Any], float],
) -> int:
    """Return index of the nearest node in *tree* to *state*."""
    best_idx = 0
    best_dist = float("inf")
    for i, node in enumerate(tree):
        d = distance_fn(node.state, state)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _generic_extract_path(
    tree_a: List[_TreeNode],
    idx_a: int,
    tree_b: List[_TreeNode],
    idx_b: int,
    a_is_start: bool,
    copy_fn: Callable[[Any], Any],
) -> list:
    """Extract the path from start to goal through two connected trees."""
    path_a: list = []
    idx = idx_a
    while idx >= 0:
        path_a.append(copy_fn(tree_a[idx].state))
        idx = tree_a[idx].parent
    path_a.reverse()

    path_b: list = []
    idx = idx_b
    while idx >= 0:
        path_b.append(copy_fn(tree_b[idx].state))
        idx = tree_b[idx].parent

    if a_is_start:
        return path_a + path_b
    else:
        path_b.reverse()
        return path_b + path_a[::-1]


def rrt_connect_generic(
    start_state: Any,
    goal_state: Any,
    *,
    distance_fn: Callable[[Any, Any], float],
    steer_fn: Callable[[Any, Any, float], Any],
    sample_fn: Callable[[], Any],
    collision_fn: Callable[[Any], bool],
    segment_free_fn: Callable[[Any, Any], bool],
    copy_fn: Callable[[Any], Any],
    step_size: float,
    goal_bias: float,
    max_iterations: int,
    connect_max_steps: int,
    on_extend: Optional[Callable] = None,
    on_collision: Optional[Callable] = None,
    on_iteration: Optional[Callable] = None,
) -> Tuple[bool, list, int, int, int]:
    """Domain-agnostic bi-directional RRT-Connect.

    The algorithm is parameterised entirely through callback functions so
    it can operate on any state representation (SE(3) poses, joint
    configurations, etc.).

    **The caller is responsible for the fast-path direct-connection check.**
    Call ``segment_free_fn(start, goal)`` before invoking this function and
    return early if the segment is clear.

    Args:
        start_state: Start state.
        goal_state: Goal state.
        distance_fn: ``(s1, s2) -> float`` — metric between two states.
        steer_fn: ``(from_s, to_s, step) -> new_s`` — move from one state
            toward another by at most *step*.
        sample_fn: ``() -> state`` — generate a random state.
        collision_fn: ``(state) -> bool`` — ``True`` if *state* is in
            collision.
        segment_free_fn: ``(s1, s2) -> bool`` — ``True`` if the segment
            between *s1* and *s2* is collision-free.
        copy_fn: ``(state) -> state`` — return a deep copy of *state*.
        step_size: Maximum step size for tree extension.
        goal_bias: Probability of sampling the goal instead of a random
            state each iteration.
        max_iterations: Maximum number of RRT iterations.
        connect_max_steps: Maximum greedy-connect steps per iteration.
        on_extend: Optional ``(parent_state, new_state, a_is_start)``
            callback invoked when a new node is added to a tree.
        on_collision: Optional ``(state,)`` callback invoked when a sample
            is rejected due to collision.
        on_iteration: Optional ``(iteration, tree_a_size, tree_b_size)``
            callback invoked at the start of each iteration.

    Returns:
        ``(success, path, iterations, tree_a_size, tree_b_size)`` where
        *path* is a list of states from *start_state* to *goal_state*.
    """
    tree_a: List[_TreeNode] = [_TreeNode(state=copy_fn(start_state))]
    tree_b: List[_TreeNode] = [_TreeNode(state=copy_fn(goal_state))]
    a_is_start = True

    for iteration in range(max_iterations):
        if on_iteration is not None:
            on_iteration(iteration, len(tree_a), len(tree_b))

        # ── Sample ────────────────────────────────────────────────────
        if np.random.random() < goal_bias:
            sample = copy_fn(goal_state if a_is_start else start_state)
        else:
            sample = sample_fn()

        # ── EXTEND tree_a toward sample ───────────────────────────────
        nearest_idx = _generic_nearest(tree_a, sample, distance_fn)
        nearest = tree_a[nearest_idx]
        new_state = steer_fn(nearest.state, sample, step_size)

        if collision_fn(new_state):
            if on_collision is not None:
                on_collision(new_state)
            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start
            continue

        if not segment_free_fn(nearest.state, new_state):
            if on_collision is not None:
                on_collision(new_state)
            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start
            continue

        new_idx_a = len(tree_a)
        tree_a.append(_TreeNode(state=new_state, parent=nearest_idx))
        if on_extend is not None:
            on_extend(nearest.state, new_state, a_is_start)

        # ── CONNECT tree_b toward new node ────────────────────────────
        connected = False
        connect_idx_b = -1

        nearest_b_idx = _generic_nearest(tree_b, new_state, distance_fn)
        current_state = tree_b[nearest_b_idx].state
        parent_b_idx = nearest_b_idx

        for _ in range(connect_max_steps):
            steered = steer_fn(current_state, new_state, step_size)

            if collision_fn(steered):
                break

            if not segment_free_fn(current_state, steered):
                break

            connect_idx_b = len(tree_b)
            tree_b.append(_TreeNode(state=steered, parent=parent_b_idx))
            if on_extend is not None:
                on_extend(current_state, steered, not a_is_start)
            parent_b_idx = connect_idx_b

            if distance_fn(steered, new_state) < step_size * 0.5:
                connected = True
                break

            current_state = steered

        if connected:
            path = _generic_extract_path(
                tree_a, new_idx_a, tree_b, connect_idx_b,
                a_is_start, copy_fn,
            )
            return (True, path, iteration + 1, len(tree_a), len(tree_b))

        tree_a, tree_b = tree_b, tree_a
        a_is_start = not a_is_start

    return (False, [], max_iterations, len(tree_a), len(tree_b))
