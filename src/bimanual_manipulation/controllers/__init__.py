from bimanual_manipulation.controllers.arm import RobosuiteArmController
from bimanual_manipulation.controllers.gripper import RobosuiteGripperController
from bimanual_manipulation.controllers.trajectory_planner import (
    plan_quintic_trajectory,
    smoothen_trajectory,
)
from bimanual_manipulation.controllers.visualization import TrajectoryVisualizer


def __getattr__(name):
    if name == "TrajectoryExecutor":
        from bimanual_manipulation.controllers.trajectory_executor import TrajectoryExecutor
        return TrajectoryExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RobosuiteArmController",
    "RobosuiteGripperController",
    "TrajectoryExecutor",
    "TrajectoryVisualizer",
    "plan_quintic_trajectory",
    "smoothen_trajectory",
]