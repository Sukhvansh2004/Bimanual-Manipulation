"""
RRT planning subpackage.

Contains the domain-agnostic RRT-Connect core, domain-specific
specialisations (SE(3), joint-space), path post-processing, and
Meshcat visualisation.

Modules
-------
core
    Generic bi-directional RRT-Connect parameterised by callbacks.
se3_planner
    SE(3) specialisation for object path planning.
joint_planner
    Joint-space specialisation for arm trajectory planning.
path_smoothing
    Shortcutting, Bézier smoothing, and densification.
visualizer
    Meshcat-based RRT tree / path visualiser.
"""

from robosuite_franka_manipulation.planning.rrt.core import (
    rrt_connect_generic,
)
from robosuite_franka_manipulation.planning.rrt.se3_planner import (
    RRTNode,
    RRTResult,
    rrt_connect_se3,
)
from robosuite_franka_manipulation.planning.rrt.joint_planner import (
    JointRRTResult,
    rrt_connect_joint_space,
    shortcut_joint_path,
    bezier_smooth_joint_path,
    densify_joint_path,
)
from robosuite_franka_manipulation.planning.rrt.path_smoothing import (
    shortcut_generic,
    densify_generic,
    bezier_smooth_generic,
    smooth_path,
    densify_path,
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
