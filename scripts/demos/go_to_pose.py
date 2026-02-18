import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import threading

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.controllers import RobosuiteArmController
from robosuite_franka_manipulation.environments import make_dual_arm_env, DualArmEnvironment
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
    def __init__(self, env: DualArmEnvironment, env_lock: threading.RLock, tf_manager: TransformManager):
        """
        Initializes the controller with the environment.
        """
        self.env = env
        self.tf_manager = tf_manager
        # Robot 0 is on one side, Robot 1 is on the other
        # Both Panda robots have arm_id="right" (single-arm robots)
        self.left_arm = RobosuiteArmController(env, arm_id="right", robot_id=0, env_lock=env_lock, visualize=True, transform_manager=tf_manager)
        self.right_arm = RobosuiteArmController(env, arm_id="right", robot_id=1, env_lock=env_lock, visualize=True, transform_manager=tf_manager)
        print("Dual Arm Pose Controller Initialized.")
        
        # Print initial positions
        self._print_arm_positions()
    
    def _print_arm_positions(self):
        """Print current arm positions for debugging."""
        left_eef = "robot0_right_eef"
        right_eef = "robot1_right_eef"
        
        try:
            left_pose = self.tf_manager.get_frame_pose(left_eef, "robot0_base")
            right_pose = self.tf_manager.get_frame_pose(right_eef, "robot1_base")
            
            left_pos = left_pose.position
            right_pos = right_pose.position
            
            print(f"  Robot 0 (left) EEF position: {left_pos}")
            print(f"  Robot 1 (right) EEF position: {right_pos}")
        except Exception as e:
            print(f"  Error getting arm positions: {e}")

    def send_pose_goal(self, arm: str, position: list, orientation: list):
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
        # robot = self.env.robots[robot_id]
        eef_frame = f"robot{robot_id}_right_eef"
        base_frame = f"robot{robot_id}_base"
        
        try:
            pose = self.tf_manager.get_frame_pose(eef_frame, base_frame)
            initial_pos = pose.position
            print(f"Sending {robot_name} to position {pos}")
            print(f"  Initial: {initial_pos}")
            print(f"  Delta:   {pos - initial_pos}")
        except Exception as e:
            print(f"Error getting initial pose: {e}")
            return
        
        target_pose = StampedPose(
            position=pos,
            quaternion=quat,
            frame_id=controller.base_frame
        )
        
        success = controller.move_to_pose(target_pose)
        
        # Print final position and error
        try:
            final_pose = self.tf_manager.get_frame_pose(eef_frame, base_frame)
            final_pos = final_pose.position
            error = pos - final_pos
            print(f"  Final:   {final_pos}")
            print(f"  Error:   {error} (norm: {np.linalg.norm(error):.4f}m)")
        except Exception as e:
            print(f"Error getting final pose: {e}")
        
        if success:
            print(f"  ✓ {robot_name} reached target.")
        else:
            print(f"  ✗ {robot_name} failed to reach target within threshold.")


if __name__ == '__main__':
    env = None
    try:
        # Create environment
        env = make_dual_arm_env()
        env.reset()

        # Create reentrant lock for thread safety
        env_lock = threading.RLock()

        # Create TransformManager
        tf_manager = TransformManager(env, env_lock=env_lock)
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
        # robot0 = env.robots[0]
        # robot1 = env.robots[1]
        
        pose0 = tf_manager.get_frame_pose("robot0_right_eef", "robot0_base")
        pose1 = tf_manager.get_frame_pose("robot1_right_eef", "robot1_base")
        
        left_current_pos = pose0.position
        right_current_pos = pose1.position
        
        # left_current_quat = pose0.quaternion
        # right_current_quat = pose1.quaternion
        
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