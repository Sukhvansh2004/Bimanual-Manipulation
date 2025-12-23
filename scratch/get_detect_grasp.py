detections = detect() # type: ignore
grasps = get_grasps(detections[0]) # type: ignore
from robosuite_franka_manipulation.transforms import StampedPose
grasp = StampedPose.from_dict(grasps[0])

# exec(open("scratch/get_detect_grasp.py").read())  # type: ignore[union-attr] --- IGNORE ---
# system.left_arm.move_to_pose(grasp)