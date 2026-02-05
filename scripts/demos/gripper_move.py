from robosuite_franka_manipulation.controllers import RobosuiteGripperController
from robosuite_franka_manipulation.environments import make_dual_arm_env
from robosuite_franka_manipulation import constants
import logging

# Configure logging
logging.basicConfig(
    level=constants.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True,
)
logger = logging.getLogger(__name__)


def test_gripper_sequence(env):
    """
    A test script to open and close both grippers in robosuite.
    """
    logger.info("Initializing gripper controllers...")
    left_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=0)
    right_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=1)

    input("Press Enter to start the gripper test sequence...")
    
    logger.info("Starting gripper test sequence.")

    # --- Open both ---
    logger.info("--- Releasing both grippers ---")
    left_gripper.release()
    right_gripper.release()
    input("Press Enter to continue...")

    # --- Close both ---
    logger.info("--- Grasping both grippers ---")
    left_gripper.grasp()
    right_gripper.grasp()
    input("Press Enter to continue...")

    # --- Open both again ---
    logger.info("--- Releasing both grippers again ---")
    left_gripper.release()
    right_gripper.release()
    input("Press Enter to continue...")
    
    logger.info("Gripper test sequence finished.")

if __name__ == '__main__':
    try:
        env = make_dual_arm_env()
        env.reset()
        test_gripper_sequence(env)
        env.close()
    except Exception as e:
        logger.exception(f"Error: {e}")