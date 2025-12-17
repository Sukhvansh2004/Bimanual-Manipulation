import time

from ...controllers import RobosuiteGripperController
from ...environments import make_dual_arm_handoff_env

def test_gripper_sequence(env):
    """
    A test script to open and close both grippers in robosuite.
    """
    print("Initializing gripper controllers...")
    left_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=0)
    right_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=1)

    print("Starting gripper test sequence.")

    # --- Open both ---
    print("--- Opening both grippers ---")
    left_gripper.open()
    right_gripper.open()
    time.sleep(2)

    # --- Close both ---
    print("--- Closing both grippers ---")
    left_gripper.close()
    right_gripper.close()
    time.sleep(2)

    # --- Open both again ---
    print("--- Opening both grippers again ---")
    left_gripper.open()
    right_gripper.open()
    time.sleep(2)
    
    print("Gripper test sequence finished.")

if __name__ == '__main__':
    try:
        env = make_dual_arm_handoff_env()
        env.reset()
        test_gripper_sequence(env)
        env.close()
    except Exception as e:
        print(f"Error: {e}")