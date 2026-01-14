import os
# Import the module where the original factory 
import robosuite_franka_manipulation.constants as constants
import robosuite.models.grippers.gripper_factory as gf_module
import robosuite.models.grippers
from robosuite.models.grippers.gripper_model import GripperModel

class PandaSuctionGripper(GripperModel):
    """
    Suction Gripper for Franka Panda (Cobot Pump).
    """
    def __init__(self, idn=0):
        # Construct absolute path using the project root from constants
        xml_path = constants.SUCTION_GRIPPER_XML_PATH
        
        # Verify file exists to avoid obscure errors later
        if not xml_path.exists():
            raise FileNotFoundError(f"Suction gripper XML not found at: {xml_path}")

        super().__init__(str(xml_path), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1

    @property
    def contact_geoms(self):
        return ["geo_snout"]


# 1. Save the original factory function
original_gripper_factory = gf_module

# 2. Define our wrapper
def custom_gripper_factory(name, *args, **kwargs):
    if name == "PandaSuctionGripper":
        return PandaSuctionGripper(*args, **kwargs)
    # Fallback to the original factory for standard grippers
    return original_gripper_factory(name, *args, **kwargs)

# 3. Apply the patch to both the module and the package namespace
robosuite.models.grippers.gripper_factory = custom_gripper_factory