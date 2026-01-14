import os
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import register_gripper
from robosuite_franka_manipulation import constants

class PandaSuctionGripper(GripperModel):
    """
    Suction Gripper for Franka Panda (Cobot Pump).
    """
    def __init__(self, idn=0):
        # Construct absolute path using the project root from constants
        # Ensure constants.SUCTION_GRIPPER_XML_PATH is defined in your constants file
        # OR use: 
        # root = constants.get_project_root()
        # xml_path = root / "src/robosuite_franka_manipulation/assets/grippers/panda_suction.xml"
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
        # Suction grippers generally don't have actuated joints with initial positions
        return None

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1

    @property
    def contact_geoms(self):
        # Must match the collision geom name in your XML
        return ["geo_snout"]

    @property
    def _important_sites(self):
        """
        Explicitly define the sites Robosuite looks for. 
        This prevents the 'NoneType' error by ensuring it finds 'grip_site'.
        """
        return {
            "grip_site": "grip_site",
            "grip_cylinder": "grip_cylinder",
            "ee": "grip_site",  # Main end-effector frame
            "ee_x": "grip_site", 
            "ee_y": "grip_site",
            "ee_z": "grip_site",
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