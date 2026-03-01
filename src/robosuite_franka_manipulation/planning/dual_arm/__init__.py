"""Dual-arm parallel execution subpackage.

Provides :class:`DualArmParallelPlanner` — a mink-based collision-aware
concurrent trajectory executor for two arms — and
:class:`ParallelExecutionResult`.

Submodules
----------
result
    :class:`ParallelExecutionResult` dataclass.
trajectory_helpers
    Trajectory planning / conversion utilities used by the planner.
mink_setup
    :class:`MinkComponents` dataclass and :func:`build_mink_components`
    builder for constructing mink tasks, limits, and constraints.
planner
    :class:`DualArmParallelPlanner` orchestration class.
"""

from robosuite_franka_manipulation.planning.dual_arm.result import (
    ParallelExecutionResult,
)
from robosuite_franka_manipulation.planning.dual_arm.planner import (
    DualArmParallelPlanner,
)

__all__ = [
    "DualArmParallelPlanner",
    "ParallelExecutionResult",
]
