"""
High-level trajectory planning subpackage.

Provides two top-level pipeline functions that orchestrate the
lower-level RRT planners, smoothing, and densification:

* :func:`plan_object_trajectory` — collision-free SE(3) object path.
* :func:`plan_arm_trajectory` — collision-free joint-space arm path.

Modules
-------
object_planner
    SE(3) object trajectory planning via RRT-Connect + Bézier smoothing.
arm_planner
    Joint-space arm trajectory planning via RRT-Connect + smoothing.
"""

from robosuite_franka_manipulation.planning.trajectory.object_planner import (
    plan_object_trajectory,
)
from robosuite_franka_manipulation.planning.trajectory.arm_planner import (
    plan_arm_trajectory,
)

__all__ = [
    "plan_object_trajectory",
    "plan_arm_trajectory",
]
