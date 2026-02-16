#!/usr/bin/env python3
"""
Dual-Arm Handover Demo using Trajectory Files

This script:
1. Loads trajectory files for both robots.
2. Identifies feasibility segments for suction (Robot 0) and finger (Robot 1).
3. Executes a grasp on a plate using Robot 0 (suction) at its first feasible point.
4. Robot 0 follows its feasible trajectory segment.
5. Performs a handover to Robot 1 (finger) at a common feasibility point.
6. Robot 1 takes the object to its last feasible trajectory point.
"""

import os
import time
import logging
import numpy as np
from typing import List, Tuple, Optional

from robosuite_franka_manipulation import constants
from robosuite_franka_manipulation.services.services_core import ServiceClient
from robosuite_franka_manipulation.transforms import StampedPose
from robosuite_franka_manipulation.planning import GraspGenerator
from robosuite_franka_manipulation.planning.utils import load_trajectory, WaypointFeasibilityResult
from robosuite_franka_manipulation.utils.env_utils import create_ycb_object, env_step
from robosuite_franka_manipulation.utils.primitives import ObjectGraspPrimitive, DualArmHandoverPlanner
from scripts.manipulation.dual_arm_manipulation import DualArmManipulationSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_first_feasible_point(waypoint_results: List[WaypointFeasibilityResult]) -> Optional[int]:
    """Find the index of the first feasible waypoint."""
    for i, res in enumerate(waypoint_results):
        if res.is_feasible:
            return i
    return None

def find_common_feasibility_point(res0: List[WaypointFeasibilityResult], res1: List[WaypointFeasibilityResult]) -> Optional[int]:
    """Find the first index where both robots are feasible."""
    min_len = min(len(res0), len(res1))
    for i in range(min_len):
        if res0[i].is_feasible and res1[i].is_feasible:
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
    system = DualArmManipulationSystem(visualize=True, ycb_objects=ycb_objects, use_fastsam=False)
    system.start(start_teleop_thread=False)
    
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

        # 3. Grasp Generation
        generator = GraspGenerator(system.env, service_client=system.service_client)
        
        # Suction grasps for Robot 0 (Plate)
        logger.info("Generating suction grasps for Robot 0...")
        grasps0 = generator.generate_grasps("plate_0", gripper_types="suction", num_grasps=5)
        # Finger grasps for Robot 1 (Plate)
        logger.info("Generating finger grasps for Robot 1...")
        grasps1 = generator.generate_grasps("plate_0", gripper_types="finger", num_grasps=5)

        if not grasps0 or not grasps1:
            logger.error("Failed to generate necessary grasps.")
            return
        
        grasp0 = grasps0[0]
        grasp1 = grasps1[0]

        # 4. Multi-Handover Execution Loop
        current_idx = 0
        # Determine starting robot (first one feasible)
        first0 = find_first_feasible_point(res0)
        first1 = find_first_feasible_point(res1)
        
        if first0 is None and first1 is None:
            logger.error("No feasible starting point found for either robot.")
            return
            
        current_robot = 0 if (first0 is not None and (first1 is None or first0 <= first1)) else 1
        current_idx = first0 if current_robot == 0 else first1
        
        # Initial Grasp
        logger.info(f"Robot {current_robot} starting at waypoint {current_idx}")
        current_grasp = grasp0 if current_robot == 0 else grasp1
        current_arm = system.left_arm if current_robot == 0 else system.right_arm
        
        obj_pose = obj_traj0[current_idx] if current_robot == 0 else obj_traj1[current_idx]
        grasp_primitive = ObjectGraspPrimitive(object_pose=obj_pose, grasp_transform=current_grasp)
        grasp_traj_data = grasp_primitive.generate_trajectory()
        
        current_arm.move_to_pose(grasp_traj_data["pregrasp"])
        current_arm.follow_trajectory(grasp_traj_data["approach"])
        env_step(system.env, action=[-1.0], robot_id=current_robot, arm="right")
        time.sleep(0.5)

        while current_idx < len(res0) - 1:
            # Look ahead for feasibility changes
            res_current = res0 if current_robot == 0 else res1
            res_other = res1 if current_robot == 0 else res0
            
            # Find how far we can go with current robot
            next_handover_idx = -1
            for i in range(current_idx + 1, len(res_current)):
                if not res_current[i].is_feasible:
                    # Current robot becomes infeasible! Must handover before this.
                    # Find a handover point before 'i' where both are feasible.
                    for j in reversed(range(current_idx, i)):
                        if res_current[j].is_feasible and res_other[j].is_feasible:
                            next_handover_idx = j
                            break
                    break
            
            # If current robot stays feasible till the end, check if other robot is needed for the final goal
            if next_handover_idx == -1:
                last_feasible_other = find_last_feasible_point(res_other)
                last_feasible_current = find_last_feasible_point(res_current)
                
                if last_feasible_other > last_feasible_current:
                    # Other robot can go further! Find a common point to switch.
                    for j in reversed(range(current_idx, last_feasible_current + 1)):
                        if res_current[j].is_feasible and res_other[j].is_feasible:
                            next_handover_idx = j
                            break
            
            if next_handover_idx != -1 and next_handover_idx > current_idx:
                # Follow segment to handover point
                logger.info(f"Robot {current_robot} following segment to handover at {next_handover_idx}...")
                segment = (eef_traj0 if current_robot == 0 else eef_traj1)[current_idx+1 : next_handover_idx+1]
                current_arm.follow_trajectory(segment, gripper_action=-1.0)
                current_idx = next_handover_idx
                
                # Execute Handover
                logger.info(f"Executing handover at waypoint {current_idx} (Robot {current_robot} -> {1-current_robot})...")
                obj_pose_h = (obj_traj0 if current_robot == 0 else obj_traj1)[current_idx]
                
                h_planner = DualArmHandoverPlanner(
                    grasp_transfer=grasp0 if current_robot == 0 else grasp1,
                    grasp_receiver=grasp1 if current_robot == 0 else grasp0,
                    handover_object_pose=obj_pose_h
                )
                h_traj = h_planner.generate_trajectory()
                
                # Receiver (other robot) moves to pre-alignment
                other_arm = system.right_arm if current_robot == 0 else system.left_arm
                other_arm.move_to_pose(h_traj["receiver_pre"][0])
                current_arm.move_to_pose(h_traj["transfer_hold"][0], gripper_action=-1.0)
                
                # Receiver approach
                other_arm.follow_trajectory(h_traj["receiver_approach"])
                
                # Switch grippers
                env_step(system.env, action=[-1.0], robot_id=1-current_robot, arm="right")
                time.sleep(0.5)
                env_step(system.env, action=[1.0], robot_id=current_robot, arm="right")
                time.sleep(0.5)
                
                # Transfer retreat
                current_arm.follow_trajectory(h_traj["transfer_retreat"])
                
                # Update current state
                current_robot = 1 - current_robot
                current_arm = other_arm
            else:
                # No more handovers needed or possible. Go as far as possible.
                last_idx = find_last_feasible_point(res_current)
                if last_idx > current_idx:
                    logger.info(f"Robot {current_robot} following final segment to {last_idx}...")
                    segment = (eef_traj0 if current_robot == 0 else eef_traj1)[current_idx+1 : last_idx+1]
                    current_arm.follow_trajectory(segment, gripper_action=-1.0)
                    current_idx = last_idx
                break

        logger.info("Releasing object...")
        env_step(system.env, action=[1.0], robot_id=current_robot, arm="right")
        time.sleep(1.0)
        logger.info("Demo complete!")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        system.stop()

if __name__ == "__main__":
    main()
