"""
High-level trajectory planning subpackage.

Provides pipeline functions that orchestrate the lower-level RRT
planners, smoothing, and densification:

* :func:`plan_object_trajectory` — collision-free SE(3) object path.
* :func:`plan_arm_trajectory` — collision-free joint-space arm path.
* :func:`generate_object_trajectory` — RRT or direct SE(3) interpolation.
* :func:`generate_end_effector_trajectory` — object → EEF via grasp transform.

Modules
-------
object_planner
    SE(3) object trajectory planning via RRT-Connect + Bézier smoothing.
arm_planner
    Joint-space arm trajectory planning via RRT-Connect + smoothing.
generator
    Object/EEF trajectory generation (collision-aware or interpolation).
"""

from bimanual_manipulation.planning.trajectory.object_planner import (
    plan_object_trajectory,
)
from bimanual_manipulation.planning.trajectory.arm_planner import (
    plan_arm_trajectory,
)
from bimanual_manipulation.planning.trajectory.generator import (
    generate_object_trajectory,
    generate_end_effector_trajectory,
)

__all__ = [
    "plan_object_trajectory",
    "plan_arm_trajectory",
    "generate_object_trajectory",
    "generate_end_effector_trajectory",
]
