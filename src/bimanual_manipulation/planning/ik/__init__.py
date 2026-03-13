"""Inverse kinematics subpackage.

Provides IK solution enumeration, Jacobian computation, manipulability
measures, and orientation-correction helpers for single-arm manipulation.

Backends
--------
The solver backend is chosen via :data:`constants.IK_SOLVER_BACKEND`:

* ``"eaik"`` — Analytical closed-form via EAIK (fastest, default).
* ``"scipy"`` — Numerical optimisation via L-BFGS-B.
* ``"mink"`` — QP-based iterative differential IK.
"""

from bimanual_manipulation.planning.ik.ik_utils import (
    enumerate_ik_solutions_for_arm,
    compute_arm_jacobian,
    apply_ik_orientation_correction,
    reverse_ik_orientation_correction,
    unique_solutions,
)
from bimanual_manipulation.planning.ik.eaik_solver import (
    enumerate_ik_solutions_eaik,
    solve_ik_eaik,
)
from bimanual_manipulation.planning.ik.manipulability import (
    compute_manipulability_simple,
    compute_manipulability_from_jacobian,
    compute_trajectory_manipulability,
    compute_average_manipulability,
)

__all__ = [
    # IK utilities (backend-agnostic)
    "enumerate_ik_solutions_for_arm",
    "compute_arm_jacobian",
    "apply_ik_orientation_correction",
    "reverse_ik_orientation_correction",
    "unique_solutions",
    # EAIK direct exports
    "enumerate_ik_solutions_eaik",
    "solve_ik_eaik",
    # Manipulability
    "compute_manipulability_simple",
    "compute_manipulability_from_jacobian",
    "compute_trajectory_manipulability",
    "compute_average_manipulability",
]
