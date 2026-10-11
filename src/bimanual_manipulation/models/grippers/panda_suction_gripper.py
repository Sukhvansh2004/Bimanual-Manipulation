import os
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers import register_gripper
from bimanual_manipulation import constants

class PandaSuctionGripper(GripperModel):
    """Suction Gripper for Franka Panda (Cobot Pump)."""
    def __init__(self, idn=0):
        """Load the suction gripper MJCF from ``constants.SUCTION_GRIPPER_XML_PATH``.

        Args:
            idn: Unique gripper instance id used for name prefixing.

        Raises:
            FileNotFoundError: If the gripper XML does not exist.
        """
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
        """Pass the action through unchanged.

        Args:
            action: -1 (open/release) to 1 (close/suck).

        Returns:
            The unmodified ``action``.
        """
        return action

    @property
    def init_qpos(self):
        """np.ndarray: Initial joint position ``[0.0]`` for the suction slide joint."""
        # Robosuite resets gripper joints using this value; must be a numeric array.
        # Our XML defines a single slide joint (joint_suction) with range [0, 0.01].
        return np.array([0.0])

    @property
    def speed(self):
        """float: Gripper actuation speed."""
        return 0.01

    @property
    def dof(self):
        """int: Number of gripper degrees of freedom (1)."""
        return 1

    @property
    def contact_geoms(self):
        """list[str]: Prefixed name of the suction snout contact geom."""
        # Robosuite prefixes names when merging models; always return the
        # correctly-named geom to avoid lookup failures.
        return [self.correct_naming("geo_snout")]

    @property
    def _important_sites(self):
        """dict: Sites Robosuite looks up, keyed by role.

        Defined explicitly so Robosuite always finds ``grip_site`` (avoids a
        ``NoneType`` error).
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
        """dict: Important geoms; finger entries are empty, ``root`` is the snout."""
        return {
            "left_finger": [],   # Suction has no fingers
            "right_finger": [],
            "left_finger_pad": [],
            "right_finger_pad": [],
            "root": ["geo_snout"] # The main body geom
        }

# --- Registration ---
register_gripper(PandaSuctionGripper)