"""
SE(3) RRT planning subpackage.

Provides a collision-free SE(3) RRT-Connect planner for object
trajectory planning through the workspace.

Modules
-------
collision
    Object collision checking (single pose and segment).
utilities
    SE(3) distance, steering, random sampling, and segment
    collision checking.
planner
    ``RRTNode``, ``RRTResult`` data containers and
    :func:`rrt_connect_se3`.
smoothing
    SE(3)-specific path smoothing (shortcutting, Bézier, densification).
"""

from bimanual_manipulation.planning.rrt.se3.collision import (
    object_in_collision,
    check_object_pose_collision,
)
from bimanual_manipulation.planning.rrt.se3.utilities import (
    se3_pose_distance,
    sample_random_pose,
    steer,
    collision_free_segment,
)
from bimanual_manipulation.planning.rrt.se3.planner import (
    RRTNode,
    RRTResult,
    rrt_connect_se3,
)
from bimanual_manipulation.planning.rrt.se3.smoothing import (
    smooth_path,
    densify_path,
)

__all__ = [
    # Collision
    "object_in_collision",
    "check_object_pose_collision",
    # Utilities
    "se3_pose_distance",
    "sample_random_pose",
    "steer",
    "collision_free_segment",
    # Planner
    "RRTNode",
    "RRTResult",
    "rrt_connect_se3",
    # Smoothing
    "smooth_path",
    "densify_path",
]
