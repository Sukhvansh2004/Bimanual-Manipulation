import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from ...controllers import RobosuiteArmController
from ...environments import make_dual_arm_handoff_env

class DualArmPoseController:
    """
    A robosuite-equivalent DualArmPoseController.
    This node sends pose goals to both arms in a robosuite environment.
    """
    def __init__(self, env):
        """
        Initializes the controller with the environment.
        """
        self.env = env
        self.left_arm = RobosuiteArmController(env, arm_id="left", robot_id=0)
        self.right_arm = RobosuiteArmController(env, arm_id="right", robot_id=1)
        print("Dual Arm Pose Controller Initialized.")

    def send_pose_goal(self, arm, position, orientation):
        """
        Sends a pose goal to the specified arm.

        Args:
            arm (str): The arm to command, either 'left' or 'right'.
            position (list): A list of [x, y, z] coordinates.
            orientation (list): A list of [x, y, z, w] quaternion components.
        """
        pos = np.array(position)
        quat = np.array(orientation)

        if arm.lower() == 'left':
            controller = self.left_arm
        elif arm.lower() == 'right':
            controller = self.right_arm
        else:
            print("Invalid arm specified. Use 'left' or 'right'.")
            return

        success = controller.move_to_pose(pos, quat)
        if success:
            print(f"Sent goal to {arm} arm.")
        else:
            print(f"Failed to send goal to {arm} arm.")

if __name__ == '__main__':
    try:
        env = make_dual_arm_handoff_env()
        env.reset()

        # Create an instance of our controller
        controller = DualArmPoseController(env)

        # Define a safe "home" or "ready" pose for the left arm
        left_arm_position = [0.8, 0.1, 0.4]
        # Identity quaternion (no rotation, end-effector points forward)
        left_arm_orientation = [0.0, 0.0, 0.0, 1.0]

        # Define a symmetrical pose for the right arm
        right_arm_position = [0.4, -0.2, 0.4]
        right_arm_orientation = [0.0, 0.0, 0.0, 1.0]

        # --- Send Commands ---
        print("Sending Left arm to its ready position...")
        controller.send_pose_goal('left', left_arm_position, left_arm_orientation)

        # Wait a moment before sending the next command
        time.sleep(2.0)

        print("Sending Right arm to its ready position...")
        controller.send_pose_goal('right', right_arm_position, right_arm_orientation)

        print("Both arms have been commanded to their poses.")

        env.close()

    except Exception as e:
        print(f"Error: {e}")