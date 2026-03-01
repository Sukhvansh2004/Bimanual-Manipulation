"""robosuite_franka_manipulation.planning — manipulation planning package.

Subpackages
-----------
rrt
    RRT-Connect planners (generic core, SE(3), joint-space), path
    smoothing, and Meshcat visualization.
visualization
    Robot/object mesh visualization and trajectory plotting.
utils
    Low-level shared utilities (IK, collision, MuJoCo helpers, etc.).

Note
----
``DualArmParallelPlanner`` and ``ParallelExecutionResult`` are imported
lazily (via ``__getattr__``) because ``dual_arm`` depends on
``controllers.arm``, which in turn imports ``planning.utils``.  Eager
importing would create a circular dependency.
"""

import importlib as _importlib

from robosuite_franka_manipulation.planning.grasp_generator import GraspGenerator
from robosuite_franka_manipulation.planning.collision_checker import (
    find_feasible_grasp_pairs,
    check_single_grasp_feasibility,
    check_trajectory_feasibility,
    check_dual_trajectory_feasibility,
)
from robosuite_franka_manipulation.planning.trajectory_generator import (
    generate_end_effector_trajectory,
)
from robosuite_franka_manipulation.planning.trajectory import (
    plan_object_trajectory,
    plan_arm_trajectory,
)

# ── RRT subpackage ────────────────────────────────────────────────────────
from robosuite_franka_manipulation.planning.rrt import (
    rrt_connect_generic,
    rrt_connect_se3,
    RRTNode,
    RRTResult,
    RRTVisualizer,
    rrt_connect_joint_space,
    JointRRTResult,
    shortcut_joint_path,
    bezier_smooth_joint_path,
    densify_joint_path,
    shortcut_generic,
    densify_generic,
    bezier_smooth_generic,
    smooth_path,
    densify_path,
)

# ── Dual-arm execution (lazy — avoids circular import via controllers.arm) ────
# Accessed via __getattr__ at end of file.

# ── Segment assignment ────────────────────────────────────────────────────
from robosuite_franka_manipulation.planning.assignment import (
    compute_optimal_robot_assignment,
    AssignmentResult,
    save_assignment_result,
    load_assignment_result,
    load_all_trajectories,
    make_collision_check_fn,
    make_ik_solver_fn,
    make_jacobian_fns,
)

from robosuite_franka_manipulation.schema import GraspSchema
from robosuite_franka_manipulation.transforms import StampedPose

# ── Utils / trajectory ────────────────────────────────────────────────────
from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    WaypointFeasibilityResult,
)

# ── Visualization subpackage ─────────────────────────────────────────────
from robosuite_franka_manipulation.planning.visualization import (
    visualize_trajectory,
    RobotMeshVisualizer,
)

__all__ = [
    'GraspGenerator',
    'GraspSchema',
    'StampedPose',
    'find_feasible_grasp_pairs',
    'check_single_grasp_feasibility',
    'generate_end_effector_trajectory',
    'plan_object_trajectory',
    # RRT planner
    'rrt_connect_generic',
    'rrt_connect_se3',
    'RRTNode',
    'RRTResult',
    'RRTVisualizer',
    # Joint-space RRT planner
    'rrt_connect_joint_space',
    'JointRRTResult',
    'plan_arm_trajectory',
    'shortcut_joint_path',
    'bezier_smooth_joint_path',
    'densify_joint_path',
    # Path smoothing
    'shortcut_generic',
    'densify_generic',
    'bezier_smooth_generic',
    'smooth_path',
    'densify_path',
    # Dual-arm parallel planner
    'DualArmParallelPlanner',
    'ParallelExecutionResult',
    # Segment assignment planner
    'compute_optimal_robot_assignment',
    'AssignmentResult',
    'save_assignment_result',
    'load_assignment_result',
    'load_all_trajectories',
    'make_collision_check_fn',
    'make_ik_solver_fn',
    'make_jacobian_fns',
    # Trajectory utilities
    'check_trajectory_feasibility',
    'check_dual_trajectory_feasibility',
    'WaypointFeasibilityResult',
    # Visualization
    'visualize_trajectory',
    'RobotMeshVisualizer',
]


# ── Lazy loader for dual-arm planner (breaks circular import) ─────────────
def __getattr__(name: str):
    _LAZY = {
        "DualArmParallelPlanner": "dual_arm",
        "ParallelExecutionResult": "dual_arm",
    }
    if name in _LAZY:
        mod = _importlib.import_module(f".{_LAZY[name]}", __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache for subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
