import os
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import register_gripper
from bimanual_manipulation import constants

class PandaSuctionGripper(GripperModel):
    """
    Suction Gripper for Franka Panda (Cobot Pump).
    """
    def __init__(self, idn=0):
        # Construct absolute path using the project root from constants
        # Ensure constants.SUCTION_GRIPPER_XML_PATH is defined in your constants file
        # OR use: 
        # root = constants.get_project_root()
        # xml_path = root / "src/bimanual_manipulation/assets/grippers/panda_suction.xml"
        xml_path = constants.SUCTION_GRIPPER_XML_PATH
        
        if not xml_path.exists():
            raise FileNotFoundError(f"Suction gripper XML not found at: {xml_path}")

        super().__init__(str(xml_path), idn=idn)

    def format_action(self, action):
        """
        Args:
            action: -1 (open/release) to 1 (close/suck)
        """
        return action

    @property
    def init_qpos(self):
        # Robosuite resets gripper joints using this value; must be a numeric array.
        # Our XML defines a single slide joint (joint_suction) with range [0, 0.01].
        return np.array([0.0])

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1

    @property
    def contact_geoms(self):
        # Robosuite prefixes names when merging models; always return the
        # correctly-named geom to avoid lookup failures.
        return [self.correct_naming("geo_snout")]

    @property
    def _important_sites(self):
        """
        Explicitly define the sites Robosuite looks for. 
        This prevents the 'NoneType' error by ensuring it finds 'grip_site'.
        """
        return {
            "grip_site": "grip_site",
            "grip_cylinder": "grip_site_cylinder",
            "ee": "grip_site",  # Main end-effector frame
            "ee_x": "ee_x",
            "ee_y": "ee_y",
            "ee_z": "ee_z",
        }

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],   # Suction has no fingers
            "right_finger": [],
            "left_finger_pad": [],
            "right_finger_pad": [],
            "root": ["geo_snout"] # The main body geom
        }

# --- Registration ---
register_gripper(PandaSuctionGripper)