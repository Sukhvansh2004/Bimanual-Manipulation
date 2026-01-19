import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import threading

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.controllers import RobosuiteArmController
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation.transforms import StampedPose, TransformManager, MeshcatTFVisualizerConfig, stream_transforms_to_meshcat

import logging
# Configure logging
logging.basicConfig(
    level=constants.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)

class DualArmPoseController:
    """
    A robosuite-equivalent DualArmPoseController.
    This controller manages two Panda robots in an opposed configuration.
    
    Note: Each Panda robot has a single arm with arm_id="right".
    - Robot 0 (left side of table) 
    - Robot 1 (right side of table)
    """
    def __init__(self, env, env_lock, tf_manager):
        """
        Initializes the controller with the environment.
        """
        self.env = env
        # Robot 0 is on one side, Robot 1 is on the other
        # Both Panda robots have arm_id="right" (single-arm robots)
        self.left_arm = RobosuiteArmController(env, arm_id="right", robot_id=0, env_lock=env_lock, visualize=True, transform_manager=tf_manager)
        self.right_arm = RobosuiteArmController(env, arm_id="right", robot_id=1, env_lock=env_lock, visualize=True, transform_manager=tf_manager)
        print("Dual Arm Pose Controller Initialized.")
        
        # Print initial positions
        self._print_arm_positions()
    
    def _print_arm_positions(self):
        """Print current arm positions for debugging."""
        left_robot = self.env.robots[0]
        right_robot = self.env.robots[1]
        left_pose = left_robot.pose_in_base_from_name(left_robot.robot_model.eef_name["right"])
        right_pose = right_robot.pose_in_base_from_name(right_robot.robot_model.eef_name["right"])
        left_pos = left_pose[:3, 3].copy()
        right_pos = right_pose[:3, 3].copy()
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
        robot = self.env.robots[robot_id]
        pose4 = robot.pose_in_base_from_name(robot.robot_model.eef_name["right"])
        initial_pos = pose4[:3, 3].copy()
        print(f"Sending {robot_name} to position {pos}")
        print(f"  Initial: {initial_pos}")
        print(f"  Delta:   {pos - initial_pos}")
        
        target_pose = StampedPose(
            position=pos,
            quaternion=quat,
            frame_id=controller.base_frame
        )
        
        success = controller.move_to_pose(target_pose)
        
        # Print final position and error
        pose4_final = robot.pose_in_base_from_name(robot.robot_model.eef_name["right"])
        final_pos = pose4_final[:3, 3].copy()
        error = pos - final_pos
        print(f"  Final:   {final_pos}")
        print(f"  Error:   {error} (norm: {np.linalg.norm(error):.4f}m)")
        
        if success:
            print(f"  ✓ {robot_name} reached target.")
        else:
            print(f"  ✗ {robot_name} failed to reach target within threshold.")


if __name__ == '__main__':
    env = None
    try:
        print("Creating dual-arm environment...")
        env = make_dual_arm_env()
        env.reset()

        # Create an environment lock for thread-safe access
        env_lock = threading.Lock()

        # Initialize TransformManager and MeshcatTFVisualizerConfig
        tf_manager = TransformManager(env, env_lock=env_lock, auto_discover=True)
        vis_config = MeshcatTFVisualizerConfig(fps=20.0, clear=True, labels=True)
        stop_event = threading.Event()

        # Start MeshCat TF visualizer in a separate thread
        vis_thread = threading.Thread(
            target=stream_transforms_to_meshcat,
            kwargs={
                "env": env,
                "env_lock": env_lock,
                "tf_manager": tf_manager,
                "config": vis_config,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        vis_thread.start()
        
        # Create an instance of our controller
        controller = DualArmPoseController(env, env_lock, tf_manager)

        # Get current positions to compute reasonable targets
        robot0 = env.robots[0]
        robot1 = env.robots[1]
        
        left_current_pose = robot0.pose_in_base_from_name(robot0.robot_model.eef_name["right"])
        right_current_pose = robot1.pose_in_base_from_name(robot1.robot_model.eef_name["right"])
        
        left_current_pos = left_current_pose[:3, 3].copy()
        right_current_pos = right_current_pose[:3, 3].copy()
        
        left_current_quat = R.from_matrix(left_current_pose[:3, :3]).as_quat()
        right_current_quat = R.from_matrix(right_current_pose[:3, :3]).as_quat()
        
        # Define relative movements from current position
        # Move left arm forward and up a bit
        left_arm_position = left_current_pos + np.array([0.3, 0.3, 0.2])
        left_arm_orientation = [0, 0, 0, 1]

        # Move right arm forward and up a bit  
        right_arm_position = right_current_pos + np.array([0.3, 0.3, 0.2])
        right_arm_orientation = [0, 0, 0, 1]

        # --- Send Commands ---
        print("\n--- Moving Robot 0 (left side) ---")
        controller.send_pose_goal('left', left_arm_position.tolist(), left_arm_orientation)
        
        # Render a few frames to show movement
        for _ in range(50):
            env.render()
            time.sleep(0.02)

        print("\n--- Moving Robot 1 (right side) ---")
        controller.send_pose_goal('right', right_arm_position.tolist(), right_arm_orientation)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        # Signal the visualization thread to stop
        if 'stop_event' in locals() and stop_event is not None:
            stop_event.set()