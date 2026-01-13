from robosuite_franka_manipulation.environments.dual_arm import (
    make_dual_arm_env,
    DualArmEnvironmentWrapper,
)

# Import helper functions from utils for backward compatibility
from robosuite_franka_manipulation.utils.env_utils import (
    get_available_ycb_objects,
    list_ycb_objects,
    create_ycb_object,
    create_primitive_object,
    YCBObject,
)

__all__ = [
    "make_dual_arm_env",
    "get_available_ycb_objects",
    "list_ycb_objects",
    "create_ycb_object",
    "create_primitive_object",
    "YCBObject",
    "DualArmEnvironmentWrapper",
]