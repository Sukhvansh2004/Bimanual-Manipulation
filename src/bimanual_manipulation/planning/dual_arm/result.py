"""Data container for dual-arm parallel execution results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParallelExecutionResult:
    """Outcome of a parallel dual-arm execution.

    Attributes:
        success_a: Arm A's final max joint error is below the convergence
            threshold.
        success_b: Arm B's final max joint error (vs. its reference
            trajectory's last waypoint) is below the threshold.
        steps_taken: Number of simulation steps executed.
        duration_s: Wall-clock duration (s).
        final_pos_error_a: Arm A's final max absolute joint error (rad).
        final_pos_error_b: Arm B's final max absolute joint error (rad).
        collision_free: ``True`` when no mink QP solve failed during
            execution (not a direct contact check); ``False`` on the
            arm-A-only fallback.
    """

    success_a: bool = False
    success_b: bool = False
    steps_taken: int = 0
    duration_s: float = 0.0
    final_pos_error_a: float = float("inf")
    final_pos_error_b: float = float("inf")
    collision_free: bool = True

    @property
    def success(self) -> bool:
        """``True`` when both arms succeeded."""
        return self.success_a and self.success_b
