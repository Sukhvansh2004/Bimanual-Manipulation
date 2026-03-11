"""
Joint-space RRT planning subpackage.

Provides a collision-free joint-space RRT-Connect planner for one arm
while the other arm is held at a fixed configuration.

Modules
-------
collision
    Joint-space collision checking (single config and segment).
utilities
    Joint-space distance, steering, sampling, and FK-based EE
    position mapping.
planner
    ``JointRRTResult`` data container and
    :func:`rrt_connect_joint_space`.
smoothing
    Post-processing: shortcutting, Bézier smoothing, densification.
"""

from bimanual_manipulation.planning.rrt.joint.collision import (
    check_arm_config_collision,
    collision_free_joint_segment,
)
from bimanual_manipulation.planning.rrt.joint.utilities import (
    joint_distance,
    steer,
    sample_random_config,
    make_ee_pos_fn,
)
from bimanual_manipulation.planning.rrt.joint.planner import (
    JointRRTResult,
    rrt_connect_joint_space,
)
from bimanual_manipulation.planning.rrt.joint.smoothing import (
    shortcut_joint_path,
    bezier_smooth_joint_path,
    densify_joint_path,
)

__all__ = [
    # Collision
    "check_arm_config_collision",
    "collision_free_joint_segment",
    # Utilities
    "joint_distance",
    "steer",
    "sample_random_config",
    "make_ee_pos_fn",
    # Planner
    "JointRRTResult",
    "rrt_connect_joint_space",
    # Smoothing
    "shortcut_joint_path",
    "bezier_smooth_joint_path",
    "densify_joint_path",
]
