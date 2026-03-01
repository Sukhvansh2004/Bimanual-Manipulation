"""Data container for dual-arm parallel execution results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParallelExecutionResult:
    """Outcome of a parallel dual-arm execution."""

    success_a: bool = False
    success_b: bool = False
    steps_taken: int = 0
    duration_s: float = 0.0
    final_pos_error_a: float = float("inf")
    final_pos_error_b: float = float("inf")
    collision_free: bool = True

    @property
    def success(self) -> bool:
        return self.success_a and self.success_b
