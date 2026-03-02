"""
RRT planning subpackage.

Contains the domain-agnostic RRT-Connect core, domain-specific
specialisations (SE(3), joint-space), path post-processing, and
Meshcat visualisation.

Subpackages
-----------
joint
    Joint-space RRT planning, collision checking, and smoothing.
se3
    SE(3) RRT planning, collision checking, and smoothing.

Modules
-------
core
    Generic bi-directional RRT-Connect parameterised by callbacks.
path_smoothing
    Generic (domain-agnostic) shortcutting, Bézier smoothing, and
    densification utilities.
visualizer
    Meshcat-based RRT tree / path visualiser.
"""

from robosuite_franka_manipulation.planning.rrt.core import (
    rrt_connect_generic,
)
from robosuite_franka_manipulation.planning.rrt.joint import (
    JointRRTResult,
    rrt_connect_joint_space,
    shortcut_joint_path,
    bezier_smooth_joint_path,
    densify_joint_path,
)
from robosuite_franka_manipulation.planning.rrt.se3 import (
    RRTNode,
    RRTResult,
    rrt_connect_se3,
    smooth_path,
    densify_path,
)
from robosuite_franka_manipulation.planning.rrt.path_smoothing import (
    shortcut_generic,
    densify_generic,
    bezier_smooth_generic,
)
from robosuite_franka_manipulation.planning.rrt.visualizer import (
    RRTVisualizer,
)

__all__ = [
    # Core
    "rrt_connect_generic",
    # SE(3)
    "RRTNode",
    "RRTResult",
    "rrt_connect_se3",
    # Joint-space
    "JointRRTResult",
    "rrt_connect_joint_space",
    "shortcut_joint_path",
    "bezier_smooth_joint_path",
    "densify_joint_path",
    # Path smoothing
    "shortcut_generic",
    "densify_generic",
    "bezier_smooth_generic",
    "smooth_path",
    "densify_path",
    # Visualizer
    "RRTVisualizer",
]
