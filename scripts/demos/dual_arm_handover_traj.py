#!/usr/bin/env python3
"""
Dual-Arm Handover Demo using Trajectory Files
"""

import os
import time
import logging
import numpy as np
from typing import List, Optional
import meshcat

from robosuite_franka_manipulation import (
    DualArmManipulationSystem, constants, StampedPose
)
from robosuite_franka_manipulation.utils.env_utils import create_ycb_object
from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    load_trajectory, WaypointFeasibilityResult
)
from robosuite_franka_manipulation.planning.utils.visualization_utils import (
    visualize_trajectory
)
from robosuite_franka_manipulation.utils.primitives import ObjectGraspPrimitive, DualArmHandoverPlanner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # 1. Environment Setup (Plate)
    ycb_objects = [create_ycb_object(ycb_name="plate", position=[0.0, 0.0, 0.85])]
    system = DualArmManipulationSystem(
        visualize=True, 
        ycb_objects=ycb_objects, 
        use_fastsam=False, 
        has_renderer=True
    )
    system.start(start_teleop_thread=False)
    
    vis = meshcat.Visualizer(zmq_url=str(constants.MESHCAT_ZMQ_URL))
    vis.delete()

    try:
        # 2. Load Trajectories
        traj_path0 = "data/robot0_trajectory_0.npz"
        traj_path1 = "data/robot1_trajectory_0.npz"
        
        logger.info(f"Loading trajectories: {traj_path0}, {traj_path1}")
        eef_traj0, obj_traj0, joints0, res0 = load_trajectory(traj_path0)
        eef_traj1, obj_traj1, joints1, res1 = load_trajectory(traj_path1)
        
        if res0 is None or res1 is None:
            logger.error("Feasibility results missing from trajectory files.")
            return

        # 3. Visualization of Trajectories
        logger.info("Visualizing trajectories in MeshCat...")
        visualize_trajectory(vis, eef_traj0, res0, name="robot0_traj", tf_manager=system.tf_manager)
        visualize_trajectory(vis, eef_traj1, res1, name="robot1_traj", tf_manager=system.tf_manager)

        # 4. Extract grasps from first points of trajectories
        # T_ee = T_obj @ inv(G) => G = inv(inv(T_obj) @ T_ee) = inv(T_ee) @ T_obj
        logger.info("Extracting grasps from trajectory first waypoints...")
        
        # Robot 0
        T_obj0 = obj_traj0[0].to_matrix()
        T_ee0 = eef_traj0[0].to_matrix()
        G0_mat = np.linalg.inv(T_ee0) @ T_obj0
        grasp0 = StampedPose.from_matrix(G0_mat, frame_id=obj_traj0[0].frame_id)
        
        # Robot 1
        T_obj1 = obj_traj1[0].to_matrix()
        T_ee1 = eef_traj1[0].to_matrix()
        G1_mat = np.linalg.inv(T_ee1) @ T_obj1
        grasp1 = StampedPose.from_matrix(G1_mat, frame_id=obj_traj1[0].frame_id)

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
        
        current_arm.move_to_pose(grasp_traj_data["pregrasp"])
        current_gripper.release()
        current_arm.follow_trajectory(grasp_traj_data["approach"])
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
                segment = (eef_traj0 if current_robot == 0 else eef_traj1)[current_idx+1 : next_handover_idx+1]
                current_arm.follow_trajectory(segment, gripper_action=1.0)
                current_idx = next_handover_idx
                
                logger.info(f"Executing handover at {current_idx}...")
                obj_pose_h = (obj_traj0 if current_robot == 0 else obj_traj1)[current_idx]
                h_planner = DualArmHandoverPlanner(
                    grasp_transfer=grasp0 if current_robot == 0 else grasp1,
                    grasp_receiver=grasp1 if current_robot == 0 else grasp0,
                    handover_object_pose=obj_pose_h
                )
                h_traj = h_planner.generate_trajectory()
                
                other_arm = system.right_arm if current_robot == 0 else system.left_arm
                other_gripper = system.right_gripper if current_robot == 0 else system.left_gripper
                
                other_arm.move_to_pose(h_traj["receiver_pre"][0])
                current_arm.move_to_pose(h_traj["transfer_hold"][0], gripper_action=1.0)
                other_arm.follow_trajectory(h_traj["receiver_approach"])
                
                other_gripper.grasp()
                time.sleep(0.5)
                current_gripper.release()
                time.sleep(0.5)
                
                current_arm.follow_trajectory(h_traj["transfer_retreat"])
                current_robot = 1 - current_robot
                current_arm = other_arm
                current_gripper = other_gripper
            else:
                last_idx = find_last_feasible_point(res_current)
                if last_idx is not None and last_idx > current_idx:
                    logger.info(f"Robot {current_robot} following final segment to {last_idx}...")
                    segment = (eef_traj0 if current_robot == 0 else eef_traj1)[current_idx+1 : last_idx+1]
                    current_arm.follow_trajectory(segment, gripper_action=1.0)
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
