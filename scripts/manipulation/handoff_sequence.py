import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from typing import Dict, Any

from robosuite_franka_manipulation.controllers import RobosuiteArmController, RobosuiteGripperController
from robosuite_franka_manipulation.environments import make_dual_arm_handoff_env
from robosuite_franka_manipulation.perception import ObjectDetector, GraspPredictor

def create_pose(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> Dict:
    """
    Creates a pose dict with position and quaternion.
    """
    quat = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_quat()
    return {"pos": np.array([x, y, z]), "quat": quat}

def run_handoff_sequence(env: Any, visualize: bool = False) -> bool:
    """
    Executes the dual-arm handoff sequence in a robosuite environment.
    """
    print("Starting handoff sequence...")

    # Initialize controllers for both arms
    left_arm = RobosuiteArmController(env, arm_id="right", robot_id=0)  # Assuming robot 0 is left
    right_arm = RobosuiteArmController(env, arm_id="right", robot_id=1)
    left_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=0)
    right_gripper = RobosuiteGripperController(env, arm_id="right", robot_id=1)

    # Initialize perception
    detector = ObjectDetector(env)
    predictor = GraspPredictor()

    # Detect objects
    objects = detector.detect_objects()
    if not objects:
        print("No objects detected!")
        return
    obj = objects[0]  # Take first object
    object_pose = obj["pose"]
    object_pcd = obj["pcd"]

    # Predict grasps
    grasps = predictor.predict_grasps(object_pcd, object_pose, num_grasps=5)
    if not grasps:
        print("No grasps predicted!")
        return
    # Use the first grasp
    grasp_pose = grasps[0]

    # --- Define Key Poses ---
    PI = np.pi
    ready_pose_left = create_pose(-0.3, 0.3, 0.5, 0, PI/2, 0)
    ready_pose_right = create_pose(0.3, 0.3, 0.5, -PI/2, 0, PI/2)
    pick_orientation = {"quat": R.from_euler('xyz', [0, PI/2, 0]).as_quat()}
    handoff_orientation = {"quat": R.from_euler('xyz', [-PI/2, 0, PI/2]).as_quat()}
    place_orientation = {"quat": R.from_euler('xyz', [0, PI/2, PI]).as_quat()}

    # Use predicted grasp pose
    pre_grasp_pose = create_pose(grasp_pose["pos"][0] - 0.1, grasp_pose["pos"][1], grasp_pose["pos"][2] + 0.1, 0, PI/2, 0)
    post_grasp_pose = create_pose(grasp_pose["pos"][0], grasp_pose["pos"][1], grasp_pose["pos"][2] + 0.2, 0, PI/2, 0)
    pre_handoff_pose = create_pose(grasp_pose["pos"][0] + 0.25, grasp_pose["pos"][1], grasp_pose["pos"][2] + 0.22, -PI/2, 0, PI/2)
    handoff_pose = create_pose(grasp_pose["pos"][0] + 0.1, grasp_pose["pos"][1], grasp_pose["pos"][2] + 0.22, -PI/2, 0, PI/2)
    place_x, place_y = -grasp_pose["pos"][0], -grasp_pose["pos"][1]
    place_pose = create_pose(place_x, place_y, 0.3, 0, PI/2, PI)

    # --- Start Sequence ---
    try:
        # 1. Go to Ready Poses
        print("STAGE 1: Moving both arms to ready poses.")
        left_arm.move_to_pose(ready_pose_left["pos"], ready_pose_left["quat"])
        right_arm.move_to_pose(ready_pose_right["pos"], ready_pose_right["quat"])
        
        # 2. Left Arm Picks Object
        print("STAGE 2: Left arm picking the object.")
        left_gripper.open()
        left_arm.move_to_pose(pre_grasp_pose["pos"], pre_grasp_pose["quat"])
        left_arm.move_to_pose(grasp_pose["pos"], grasp_pose["quat"])
        left_gripper.close()
        time.sleep(0.5)
        left_arm.move_to_pose(post_grasp_pose["pos"], post_grasp_pose["quat"])
        
        # 3. Handoff
        print("STAGE 3: Handoff from left arm to right arm.")
        right_gripper.open()
        right_arm.move_to_pose(pre_handoff_pose["pos"], pre_handoff_pose["quat"])
        right_arm.move_to_pose(handoff_pose["pos"], handoff_pose["quat"])
        right_gripper.close()
        time.sleep(0.5)
        left_gripper.open()

        # 4. Retract and Right Arm Places Object
        print("STAGE 4: Right arm placing object.")
        left_arm.move_to_pose(ready_pose_left["pos"], ready_pose_left["quat"])
        right_arm.move_to_pose(place_pose["pos"], place_pose["quat"])
        right_gripper.open()
        time.sleep(0.5)

        # 5. Return to Ready Poses
        print("STAGE 5: Returning to ready poses.")
        right_arm.move_to_pose(ready_pose_right["pos"], ready_pose_right["quat"])

        print("Handoff sequence completed successfully!")

    except Exception as e:
        print(f"An error occurred during the sequence: {e}")

if __name__ == '__main__':
    try:
        env = make_dual_arm_handoff_env()
        run_handoff_sequence(env)
        env.close()
    except Exception as e:
        print(f"Error running handoff sequence: {e}")
