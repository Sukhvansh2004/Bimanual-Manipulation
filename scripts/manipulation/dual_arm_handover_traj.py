#!/usr/bin/env python3
"""
Dual-Arm Handover Demo using Trajectory Files and Experiment JSON Configs.

Usage:
    python dual_arm_handover_traj.py --experiment experiments/flatPlate_table_to_shelf.json --traj0 data/flatPlate_table_to_shelf/robot0_trajectory_8.npz --traj1 data/flatPlate_table_to_shelf/robot1_trajectory_2.npz
    python dual_arm_handover_traj.py --experiment experiments/flatPlate_shelf_to_box.json --traj0 ... --traj1 ...
"""

import os
import time
import logging
import numpy as np
import argparse
import threading
from typing import List, Optional
import meshcat
from contextlib import nullcontext
from robosuite_franka_manipulation import (
    DualArmManipulationSystem, constants, StampedPose
)
from robosuite_franka_manipulation.utils.env_utils import create_ycb_object, create_primitive_object, env_step, register_mesh_extraction_env
from robosuite_franka_manipulation.experiment_config import load_experiment, ExperimentConfig
from robosuite_franka_manipulation.planning.utils.trajectory_utils import (
    load_trajectory, WaypointFeasibilityResult
)
from robosuite_franka_manipulation.planning.utils.visualization_utils import (
    visualize_trajectory, visualize_handover_plan
)
from robosuite_franka_manipulation.utils.primitives import (
    ObjectGraspPrimitive, DualArmHandoverPlanner, ReleaseRetreatPrimitive, extract_grasp_transform
)
from robosuite_franka_manipulation.planning.dual_arm_planner import DualArmParallelPlanner

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

def stabilise(env, env_lock: threading.RLock, steps: int = 10):
    """Step the environment a few times to stabilize after gripper actions."""
    with env_lock if env_lock else nullcontext():
        for _ in range(steps):
            env_step(env)

def main():
    parser = argparse.ArgumentParser(description="Dual-Arm Handover using Trajectory Files")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment JSON config (e.g. experiments/flatPlate_table_to_shelf.json)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (overrides JSON)")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load experiment configuration
    # ---------------------------------------------------------------
    exp_cfg = load_experiment(args.experiment)
    logger.info("Loaded experiment: %s – %s", exp_cfg.name, exp_cfg.description)

    exec_cfg = exp_cfg.execution
    vis_cfg = exp_cfg.visualization
    rt_cfg = exp_cfg.runtime

    debug = args.debug or rt_cfg.debug
    if debug:
        setup_logging(logging.DEBUG)
    else:
        setup_logging(logging.INFO)

    # 1. Environment Setup
    ycb_objects = exp_cfg.ycb_objects
    custom_objects = exp_cfg.custom_objects
    enable_meshcat = vis_cfg.enable_meshcat

    vis = None
    if exec_cfg.visualize and enable_meshcat:
        vis = meshcat.Visualizer(zmq_url=vis_cfg.zmq_url)
        vis.delete()  # Clear MeshCat

    system = DualArmManipulationSystem(
        visualize=exec_cfg.visualize, 
        meshcat_tf=exec_cfg.meshcat_tf and enable_meshcat,
        ycb_objects=ycb_objects,
        custom_objects=custom_objects,
        use_fastsam=False, 
        use_grasp_service=False,
        has_renderer=True,
        table_full_size=exp_cfg.table_full_size,
    )
    system.start(start_teleop_thread=False)
    register_mesh_extraction_env(system.env)

    try:
        # 2. Load Trajectories
        # Resolve trajectory paths from experiment config
        if not exec_cfg.traj0 or not exec_cfg.traj1:
            logger.error("Trajectory paths (traj0, traj1) must be set in the experiment JSON execution section.")
            return
        traj_path0 = exec_cfg.traj0
        traj_path1 = exec_cfg.traj1
        
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
        if exec_cfg.visualize and vis is not None:
            logger.info("Visualizing trajectories in MeshCat...")
            visualize_trajectory(vis, eef_traj0, res0, name="robot0_traj", tf_manager=system.tf_manager)
            visualize_trajectory(vis, eef_traj1, res1, name="robot1_traj", tf_manager=system.tf_manager)
            # Object trajectories (no feasibility results, use distinct colors)
            if obj_traj0:
                visualize_trajectory(vis, obj_traj0, name="object_traj0", color=0xff00ff, tf_manager=system.tf_manager)
            if obj_traj1:
                visualize_trajectory(vis, obj_traj1, name="object_traj1", color=0xffff00, tf_manager=system.tf_manager)

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
        stabilise(system.env, system.env_lock)

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
                # -------------------------------------------------------
                # PARALLEL EXECUTION: While arm A follows its trajectory
                # segment to the handover point, arm B simultaneously
                # moves to its handover pre-pose.  This saves the time
                # that arm B would otherwise spend waiting.
                # -------------------------------------------------------
                other_arm = system.right_arm if current_robot == 0 else system.left_arm
                other_gripper = system.right_gripper if current_robot == 0 else system.left_gripper

                # Pre-compute the handover plan so we know arm B's target
                obj_pose_h = (obj_traj0 if current_robot == 0 else obj_traj1)[next_handover_idx]
                h_planner = DualArmHandoverPlanner(
                    grasp_transfer=grasp0 if current_robot == 0 else grasp1,
                    grasp_receiver=grasp1 if current_robot == 0 else grasp0,
                    handover_object_pose=obj_pose_h
                )
                h_traj = h_planner.generate_trajectory()

                # Visualize handover plan in MeshCat
                if exec_cfg.visualize and vis is not None:
                    visualize_handover_plan(
                        vis, h_traj,
                        name=f"handover_{next_handover_idx}",
                        tf_manager=system.tf_manager,
                    )

                # Arm A's joint segment (carrying the object)
                joint_segment = list((joints0 if current_robot == 0 else joints1)[current_idx+1 : next_handover_idx+1])

                # Arm B's target: the handover receiver pre-pose
                receiver_pre_pose = h_traj["receiver_pre"][0]
                other_joints = joints1 if current_robot == 0 else joints0
                q_warm_start_receiver = other_joints[next_handover_idx] if other_joints[next_handover_idx] is not None else None

                logger.info(
                    f"PARALLEL: Robot {current_robot} following segment to handover at {next_handover_idx} "
                    f"while Robot {1 - current_robot} moves to handover pre-pose"
                )

                # Use the DualArmParallelPlanner for synchronised execution.
                # mink's CollisionAvoidanceLimit is configured so that
                # arm B avoids arm A AND the grasped object, while arm A
                # (frozen in the QP) follows its pre-planned trajectory.
                parallel_planner = DualArmParallelPlanner(
                    env=system.env,
                    arm_a=current_arm,
                    arm_b=other_arm,
                    env_lock=system.env_lock,
                    tf_manager=system.tf_manager,
                    collision_check=True,
                    manipulated_object=exp_cfg.task.manipulated_object,
                )
                par_result = parallel_planner.execute_parallel(
                    traj_a=joint_segment,
                    target_b=receiver_pre_pose,
                    gripper_a=1.0,       # keep gripper A closed (holding object)
                    gripper_b=-1.0,      # open gripper B (ready to receive)
                    q_warm_start_b=q_warm_start_receiver,
                    num_steps_override=200,
                )

                if not par_result.success_a:
                    logger.warning(f"Arm A did not fully converge (err={par_result.final_pos_error_a:.4f})")
                if not par_result.success_b:
                    logger.warning(f"Arm B did not fully converge (err={par_result.final_pos_error_b:.4f})")
                if par_result.collision_free:
                    logger.info("Parallel execution completed collision-free.")
                else:
                    logger.warning("Parallel execution had QP solver fallbacks — check collision avoidance.")

                current_idx = next_handover_idx

                # -------------------------------------------------------
                # Handover sequence (sequential — physical contact phase)
                # Arm B is already at the handover pre-pose from parallel
                # execution above.
                # -------------------------------------------------------
                logger.info(f"Executing handover at {current_idx}...")

                logger.info(f"Robot {current_robot} moving to handover hold pose...")
                current_arm.move_to_pose(h_traj["transfer_hold"][0], gripper_action=1.0, num_steps=50)

                logger.info(f"Robot {1 - current_robot} executing handover approach...")
                other_gripper.release()
                other_arm.follow_trajectory(h_traj["receiver_approach"], q_warm_start=q_warm_start_receiver)
                
                other_gripper.grasp()
                stabilise(system.env, system.env_lock)
                current_gripper.release()
                
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
                    current_arm.follow_joint_trajectory(joint_segment, gripper_action=1.0, smoothen=exec_cfg.smoothen, num_steps=200)
                    current_idx = last_idx
                break

        # 6. Release and Retreat
        logger.info(f"Robot {current_robot} releasing object and retreating...")
        release_pose = (eef_traj0 if current_robot == 0 else eef_traj1)[current_idx]
        release_primitive = ReleaseRetreatPrimitive()
        release_traj_data = release_primitive.generate_trajectory(release_pose)

        current_gripper.release()
        stabilise(system.env, system.env_lock)

        logger.info(f"Robot {current_robot} following retreat trajectory...")
        current_arm.follow_trajectory(release_traj_data["retreat"], gripper_action=-1.0)
        stabilise(system.env, system.env_lock)

        time.sleep(5)  # Pause to observe final state
        logger.info("Done!")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        system.stop()

if __name__ == "__main__":
    main()
