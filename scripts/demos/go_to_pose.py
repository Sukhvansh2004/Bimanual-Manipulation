import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.controllers import RobosuiteArmController
from robosuite_franka_manipulation.environments import make_dual_arm_env


class DualArmPoseController:
    """
    A robosuite-equivalent DualArmPoseController.
    This controller manages two Panda robots in an opposed configuration.
    
    Note: Each Panda robot has a single arm with arm_id="right".
    - Robot 0 (left side of table) 
    - Robot 1 (right side of table)
    """
    def __init__(self, env):
        """
        Initializes the controller with the environment.
        """
        self.env = env
        # Robot 0 is on one side, Robot 1 is on the other
        # Both Panda robots have arm_id="right" (single-arm robots)
        self.left_arm = RobosuiteArmController(env, arm_id="right", robot_id=0)
        self.right_arm = RobosuiteArmController(env, arm_id="right", robot_id=1)
        print("Dual Arm Pose Controller Initialized.")
        
        # Print initial positions
        self._print_arm_positions()
    
    def _print_arm_positions(self):
        """Print current arm positions for debugging."""
        left_pos = self.env.robots[0]._hand_pos["right"]
        right_pos = self.env.robots[1]._hand_pos["right"]
        print(f"  Robot 0 (left) EEF position: {left_pos}")
        print(f"  Robot 1 (right) EEF position: {right_pos}")

    def send_pose_goal(self, arm, position, orientation):
        """
        Sends a pose goal to the specified arm.

        Args:
            arm (str): The arm to command, either 'left' (robot 0) or 'right' (robot 1).
            position (list): A list of [x, y, z] coordinates.
            orientation (list): A list of [x, y, z, w] quaternion components.
        """
        pos = np.array(position)
        quat = np.array(orientation)

        if arm.lower() == 'left':
            controller = self.left_arm
            robot_name = "Robot 0 (left)"
            robot_id = 0
        elif arm.lower() == 'right':
            controller = self.right_arm
            robot_name = "Robot 1 (right)"
            robot_id = 1
        else:
            print("Invalid arm specified. Use 'left' or 'right'.")
            return

        # Get initial position
        initial_pos = self.env.robots[robot_id]._hand_pos["right"].copy()
        print(f"Sending {robot_name} to position {pos}")
        print(f"  Initial: {initial_pos}")
        print(f"  Delta:   {pos - initial_pos}")
        
        # Create StampedPose for the target
        from robosuite_franka_manipulation.transforms import StampedPose
        target_pose = StampedPose(
            position=pos,
            quaternion=quat,
            frame_id=controller.base_frame
        )
        
        success = controller.move_to_pose(target_pose)
        
        # Print final position and error
        final_pos = self.env.robots[robot_id]._hand_pos["right"]
        error = pos - final_pos
        print(f"  Final:   {final_pos}")
        print(f"  Error:   {error} (norm: {np.linalg.norm(error):.4f}m)")
        
        if success:
            print(f"  ✓ {robot_name} reached target.")
        else:
            print(f"  ✗ {robot_name} failed to reach target within threshold.")


if __name__ == '__main__':
    try:
        print("Creating dual-arm environment...")
        env = make_dual_arm_env()
        env.reset()

        # Create an instance of our controller
        controller = DualArmPoseController(env)

        # Get current positions to compute reasonable targets
        left_current = env.robots[0]._hand_pos["right"].copy()
        right_current = env.robots[1]._hand_pos["right"].copy()
        
        # Define relative movements from current position (safer than absolute poses)
        # Move left arm forward and up a bit
        left_arm_position = left_current + np.array([0.05, 0.0, 0.05])
        left_arm_orientation = [0.0, 0.0, 0.0, 1.0]

        # Move right arm forward and up a bit  
        right_arm_position = right_current + np.array([0.05, 0.0, 0.05])
        right_arm_orientation = [0.0, 0.0, 0.0, 1.0]

        # --- Send Commands ---
        print("\n--- Moving Robot 0 (left side) ---")
        controller.send_pose_goal('left', left_arm_position.tolist(), left_arm_orientation)
        
        # Render a few frames to show movement
        for _ in range(50):
            env.render()
            time.sleep(0.02)

        print("\n--- Moving Robot 1 (right side) ---")
        controller.send_pose_goal('right', right_arm_position.tolist(), right_arm_orientation)

        # Render final state
        print("\nHolding final position (press Ctrl+C to exit)...")
        while True:
            env.render()
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()