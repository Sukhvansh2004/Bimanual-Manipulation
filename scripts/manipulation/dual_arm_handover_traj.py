#!/usr/bin/env python3
"""
Dual-Arm Handover Demo using Trajectory Files
"""

import os
import time
import logging
import numpy as np
import argparse
import threading
from typing import List, Optional
import meshcat

from robosuite_franka_manipulation import (
    DualArmManipulationSystem, constants, StampedPose
)
from robosuite_franka_manipulation.utils.env_utils import create_ycb_object, create_primitive_object, register_mesh_extraction_env
from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    load_trajectory, WaypointFeasibilityResult
)
from robosuite_franka_manipulation.planning.utils.visualization_utils import (
    visualize_trajectory, visualize_handover_plan
)
from robosuite_franka_manipulation.utils.primitives import ObjectGraspPrimitive, DualArmHandoverPlanner, extract_grasp_transform

logger = logging.getLogger(__name__)

def setup_logging(level: int = logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

def find_first_feasible_point(waypoint_results: List[WaypointFeasibilityResult]) -> Optional[int]:
    """Find the index of the first feasible waypoint."""
    for i, res in enumerate(waypoint_results):
        if res.is_feasible:
            return i
    return None

def find_last_feasible_point(waypoint_results: List[WaypointFeasibilityResult]) -> Optional[int]:
    """Find the index of the last feasible waypoint."""
    for i in reversed(range(len(waypoint_results))):
        if waypoint_results[i].is_feasible:
            return i
    return None

def main():
    parser = argparse.ArgumentParser(description="Dual-Arm Handover using Trajectory Files")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization in robosuite")
    parser.add_argument("--meshcat-tf", action="store_true", help="Stream TF and Scene to MeshCat")
    parser.add_argument("--smoothen", action="store_true", help="Whether to smoothen trajectories during execution")
    parser.add_argument("--traj0", type=str, default="data/robot0_trajectory_8.npz", help="Path to robot0 trajectory NPZ file")
    parser.add_argument("--traj1", type=str, default="data/robot1_trajectory_2.npz", help="Path to robot1 trajectory NPZ file")
    parser.add_argument("--zmq-url", type=str, default=str(constants.MESHCAT_ZMQ_URL), help="MeshCat ZMQ URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)

    # 1. Environment Setup (Shelf and Plate)
    shelf_pos = [-0.7, 0.0, 0.0]
    shelf_quat = [0.0, 0.0, 0.70710678, 0.70710678]
    plate_pos = [0.0, 0.0, 0.85]
    plate_quat = [0, 0, 0, 1]

    ycb_objects = [
        create_ycb_object(ycb_name="shelf", position=shelf_pos, quaternion=shelf_quat)
    ]
    custom_objects = [
        create_primitive_object(obj_type='box', name='flatPlate', size=[0.3, 0.3, 0.02], position=plate_pos, quat=plate_quat)
    ]

    if args.visualize:
        vis = meshcat.Visualizer(zmq_url=args.zmq_url)
        vis.delete()  # Clear MeshCat

    system = DualArmManipulationSystem(
        visualize=args.visualize, 
        meshcat_tf=args.meshcat_tf,
        ycb_objects=ycb_objects,
        custom_objects=custom_objects,
        use_fastsam=False, 
        use_grasp_service=False,
        has_renderer=True
    )
    system.start(start_teleop_thread=False)
    register_mesh_extraction_env(system.env)

    try:
        # 2. Load Trajectories
        traj_path0 = args.traj0
        traj_path1 = args.traj1
        
        logger.info(f"Loading trajectories: {traj_path0}, {traj_path1}")
        eef_traj0, obj_traj0, joints0, res0 = load_trajectory(traj_path0)
        eef_traj1, obj_traj1, joints1, res1 = load_trajectory(traj_path1)
        
        if res0 is None or res1 is None:
            logger.error("Feasibility results missing from trajectory files.")
            return
        if joints0 is None or joints1 is None:
            logger.error("Joint config data missing from trajectory files.")
            return

        # 3. Visualization of Trajectories
        if args.visualize:
            logger.info("Visualizing trajectories in MeshCat...")
            visualize_trajectory(vis, eef_traj0, res0, name="robot0_traj", tf_manager=system.tf_manager)
            visualize_trajectory(vis, eef_traj1, res1, name="robot1_traj", tf_manager=system.tf_manager)

        # 4. Extract grasps from first points of trajectories
        # G = inv(T_ee) @ T_obj — object pose in EE frame
        logger.info("Extracting grasps from trajectory first waypoints...")
        grasp0 = extract_grasp_transform(eef_traj0[0], obj_traj0[0])
        grasp1 = extract_grasp_transform(eef_traj1[0], obj_traj1[0])

        if not grasp0 or not grasp1:
            logger.error("Failed to extract necessary grasps.")
            return

        # 5. Execution Loop
        first0 = find_first_feasible_point(res0)
        first1 = find_first_feasible_point(res1)
        
        if first0 is None and first1 is None:
            logger.error("No feasible starting point found.")
            return
            
        current_robot = 0 if (first0 is not None and (first1 is None or first0 <= first1)) else 1
        current_idx = first0 if current_robot == 0 else first1
        
        logger.info(f"Robot {current_robot} starting at waypoint {current_idx}")
        current_arm = system.left_arm if current_robot == 0 else system.right_arm
        current_gripper = system.left_gripper if current_robot == 0 else system.right_gripper
        current_grasp = grasp0 if current_robot == 0 else grasp1
        
        obj_pose = (obj_traj0 if current_robot == 0 else obj_traj1)[current_idx]
        grasp_primitive = ObjectGraspPrimitive(object_pose=obj_pose, grasp_transform=current_grasp)
        grasp_traj_data = grasp_primitive.generate_trajectory()
        
        # Use planning joint config as IK hint for initial grasp
        current_joints = (joints0 if current_robot == 0 else joints1)
        q_warm_start_grasp = current_joints[current_idx] if current_joints[current_idx] is not None else None
        
        logger.info(f"Robot {current_robot} moving to pre-grasp pose...")
        current_arm.move_to_pose(grasp_traj_data["pregrasp"], q_warm_start=q_warm_start_grasp)

        logger.info(f"Robot {current_robot} following grasp approach trajectory...")
        current_arm.follow_trajectory(grasp_traj_data["approach"], q_warm_start=q_warm_start_grasp)

        current_gripper.grasp()
        time.sleep(0.5)

        while current_idx < len(res0) - 1:
            res_current = res0 if current_robot == 0 else res1
            res_other = res1 if current_robot == 0 else res0
            
            next_handover_idx = -1
            for i in range(current_idx + 1, len(res_current)):
                if not res_current[i].is_feasible:
                    for j in reversed(range(current_idx, i)):
                        if res_current[j].is_feasible and res_other[j].is_feasible:
                            next_handover_idx = j
                            break
                    break
            
            if next_handover_idx == -1:
                last_feasible_other = find_last_feasible_point(res_other)
                last_feasible_current = find_last_feasible_point(res_current)
                if last_feasible_other is not None and last_feasible_current is not None and last_feasible_other > last_feasible_current:
                    for j in reversed(range(current_idx, last_feasible_current + 1)):
                        if res_current[j].is_feasible and res_other[j].is_feasible:
                            next_handover_idx = j
                            break
            
            if next_handover_idx != -1 and next_handover_idx > current_idx:
                logger.info(f"Robot {current_robot} following segment to handover at {next_handover_idx}...")
                joint_segment = list((joints0 if current_robot == 0 else joints1)[current_idx+1 : next_handover_idx+1])
                current_arm.follow_joint_trajectory(joint_segment, gripper_action=1.0, num_steps=200, smoothen=args.smoothen)
                current_idx = next_handover_idx
                
                logger.info(f"Executing handover at {current_idx}...")
                obj_pose_h = (obj_traj0 if current_robot == 0 else obj_traj1)[current_idx]
                h_planner = DualArmHandoverPlanner(
                    grasp_transfer=grasp0 if current_robot == 0 else grasp1,
                    grasp_receiver=grasp1 if current_robot == 0 else grasp0,
                    handover_object_pose=obj_pose_h
                )
                h_traj = h_planner.generate_trajectory()
                
                # Visualize handover plan in MeshCat
                if args.visualize:
                    visualize_handover_plan(
                        vis, h_traj,
                        name=f"handover_{current_idx}",
                        tf_manager=system.tf_manager,
                    )
                
                other_arm = system.right_arm if current_robot == 0 else system.left_arm
                other_gripper = system.right_gripper if current_robot == 0 else system.left_gripper
                
                # Use planning joint configs as IK hints for the handover poses
                other_joints = joints1 if current_robot == 0 else joints0
                q_warm_start_receiver = other_joints[current_idx] if other_joints[current_idx] is not None else None
                
                logger.info(f"Robot {1 - current_robot} moving to handover pre-pose...")
                other_arm.move_to_pose(h_traj["receiver_pre"][0])

                logger.info(f"Robot {current_robot} moving to handover hold pose...")
                current_arm.move_to_pose(h_traj["transfer_hold"][0], gripper_action=1.0)

                logger.info(f"Robot {1 - current_robot} executing handover approach...")
                other_gripper.release()
                other_arm.follow_trajectory(h_traj["receiver_approach"], q_warm_start=q_warm_start_receiver)
                
                other_gripper.grasp()
                time.sleep(0.5)
                current_gripper.release()
                time.sleep(0.5)
                
                logger.info(f"Robot {current_robot} following handover retreat trajectory...")
                current_arm.follow_trajectory(h_traj["transfer_retreat"])
                current_robot = 1 - current_robot
                current_arm = other_arm
                current_gripper = other_gripper
            else:
                last_idx = find_last_feasible_point(res_current)
                if last_idx is not None and last_idx > current_idx:
                    logger.info(f"Robot {current_robot} following final segment to {last_idx}...")
                    joint_segment = list((joints0 if current_robot == 0 else joints1)[current_idx+1 : last_idx+1])
                    current_arm.follow_joint_trajectory(joint_segment, gripper_action=1.0, smoothen=args.smoothen, num_steps=200)
                    current_idx = last_idx
                break

        logger.info("Releasing object...")
        current_gripper.release()
        time.sleep(1.0)
        logger.info("Done!")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        system.stop()

if __name__ == "__main__":
    main()
