#!/usr/bin/env python3
"""
Dual-Arm Handover Demo using Trajectory Files and Experiment JSON Configs.

By default, loads the optimal assignment result (``assignment_results.npz``)
from the experiment's data directory.  Use ``--traj0`` and ``--traj1`` to
bypass the assignment and specify trajectory files explicitly.

Usage::

    # Default — uses the assignment result in data/<experiment_name>/
    python dual_arm_handover_traj.py --experiment experiments/flatPlate_table_to_shelf.json

    # Explicit trajectories (bypasses assignment)
    python dual_arm_handover_traj.py --experiment experiments/flatPlate_table_to_shelf.json \\
        --traj0 data/flatPlate_table_to_shelf/robot0_trajectory_5.npz \\
        --traj1 data/flatPlate_table_to_shelf/robot1_trajectory_1.npz
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
from robosuite_franka_manipulation.planning.visualization.trajectory import (
    visualize_trajectory, visualize_handover_plan
)
from robosuite_franka_manipulation.utils.primitives import (
    ObjectGraspPrimitive, DualArmHandoverPlanner, ReleaseRetreatPrimitive, extract_grasp_transform
)
from robosuite_franka_manipulation.planning.dual_arm import DualArmParallelPlanner
from robosuite_franka_manipulation.planning.rrt.visualizer import RRTVisualizer
from robosuite_franka_manipulation.planning.assignment import (
    load_assignment_result,
    AssignmentResult,
)    

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


def _visualize_assignment_segments(
    vis,
    assignment: AssignmentResult,
    traj_cache: dict,
    tf_manager,
) -> None:
    """Visualize only the executed segments, grouped by (robot, traj_idx).

    Each contiguous run of the same (robot_id, traj_idx) in the assignment
    is drawn as a separate trajectory in MeshCat, named after the NPZ file
    so the user can tell which trajectory file each segment comes from.
    """

    seq = assignment.sequence
    traj_paths = assignment.trajectory_paths
    n = len(seq)

    # Build list of contiguous runs: (robot_id, traj_path, start, end)
    runs: List[tuple] = []
    run_start = 0
    for i in range(1, n):
        if seq[i] != seq[run_start] or traj_paths[i] != traj_paths[run_start]:
            runs.append((int(seq[run_start]), traj_paths[run_start], run_start, i - 1))
            run_start = i
    runs.append((int(seq[run_start]), traj_paths[run_start], run_start, n - 1))

    robot_colors = {0: 0x1f77b4, 1: 0xff7f0e}

    for run_idx, (rid, tpath, start, end) in enumerate(runs):
        eef, obj, _, res, fname = traj_cache[tpath]
        stem = os.path.splitext(fname)[0]
        # Slice only the executed waypoints
        seg_eef = eef[start : end + 1]
        seg_res = res[start : end + 1] if res else None
        seg_obj = obj[start : end + 1] if obj else None

        vis_name = f"{stem}_seg{start}-{end}"
        visualize_trajectory(
            vis, seg_eef, seg_res,
            name=vis_name,
            color=robot_colors.get(rid, 0x0000ff),
            tf_manager=tf_manager,
        )
        if seg_obj:
            visualize_trajectory(
                vis, seg_obj,
                name=f"{vis_name}_object",
                color=0xff00ff,
                tf_manager=tf_manager,
            )
        logger.info(
            "  Visualized %s: segments %d–%d (robot %d)",
            vis_name, start, end, rid,
        )


def main():
    parser = argparse.ArgumentParser(description="Dual-Arm Handover using Trajectory Files")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Path to experiment JSON config (e.g. experiments/flatPlate_table_to_shelf.json)")
    parser.add_argument("--traj0", type=str, default=None,
                        help="Path to robot-0 trajectory NPZ (bypasses assignment).")
    parser.add_argument("--traj1", type=str, default=None,
                        help="Path to robot-1 trajectory NPZ (bypasses assignment).")
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
        # -------------------------------------------------------------------
        # Two modes:
        #   a) Default — load assignment result from the data directory
        #   b) --traj0 / --traj1 — explicit trajectory files (no assignment)
        # -------------------------------------------------------------------
        assignment: Optional[AssignmentResult] = None

        # Trajectory data keyed by trajectory path (NPZ filename).
        # Each value is (eef_traj, obj_traj, joints, res, file_path).
        traj_cache: dict = {}

        def _load_cached(traj_path: str, directory: str):
            """Load a trajectory file once and cache it."""
            if traj_path not in traj_cache:
                fpath = os.path.join(directory, traj_path)
                eef, obj, jts, res = load_trajectory(fpath)
                traj_cache[traj_path] = (eef, obj, jts, res, traj_path)
            return traj_cache[traj_path]

        if args.traj0 and args.traj1:
            # ---- Explicit trajectory mode --------------------------------
            eef0, obj0, jts0, res0 = load_trajectory(args.traj0)
            eef1, obj1, jts1, res1 = load_trajectory(args.traj1)
            t0_name = os.path.basename(args.traj0)
            t1_name = os.path.basename(args.traj1)
            traj_cache[t0_name] = (eef0, obj0, jts0, res0, t0_name)
            traj_cache[t1_name] = (eef1, obj1, jts1, res1, t1_name)
            # Build a synthetic assignment-like mapping for explicit mode
            _explicit_paths = {0: t0_name, 1: t1_name}
            logger.info("Explicit trajectory mode (--traj0 / --traj1).")
        else:
            # ---- Assignment mode (default) -------------------------------
            data_dir = (
                exp_cfg.output.output_dir
                if exp_cfg.output.output_dir is not None
                else os.path.join(constants.DATA_DIR, exp_cfg.name)
            )
            assignment_file = (
                exec_cfg.assignment_path
                if exec_cfg.assignment_path is not None
                else os.path.join(data_dir, "assignment_results.npz")
            )

            if not os.path.isfile(assignment_file):
                logger.error(
                    "Assignment result not found at %s.  Run segment_assignment.py "
                    "first, or use --traj0 / --traj1 for explicit trajectories.",
                    assignment_file,
                )
                return

            logger.info("Loading assignment result from %s ...", assignment_file)
            assignment = load_assignment_result(assignment_file)

            assignment_dir = os.path.dirname(os.path.abspath(assignment_file))

            # Pre-load every unique trajectory path referenced by the assignment.
            unique_paths = set(assignment.trajectory_paths)
            for tpath in sorted(unique_paths):
                _load_cached(tpath, assignment_dir)

            logger.info(
                "Loaded %d unique trajectories for assignment: %s",
                len(unique_paths),
                sorted(unique_paths),
            )
            logger.info("Handover points: %s", assignment.handover_points)

        # Convenience accessors for the two-trajectory explicit mode
        def _get_traj_data(traj_path: str):
            """Return (eef, obj, joints, res, filename) for a trajectory."""
            return traj_cache[traj_path]

        # Validate all loaded trajectories
        for tpath, (eef, obj, jts, res, fname) in traj_cache.items():
            if res is None:
                logger.error("Feasibility results missing from %s.", fname)
                return
            if jts is None:
                logger.error("Joint config data missing from %s.", fname)
                return

        # ------------------------------------------------------------------
        # 3. Visualize only the executed trajectory segments
        # ------------------------------------------------------------------
        if exec_cfg.visualize and vis is not None:
            logger.info("Visualizing executed trajectory segments in MeshCat...")
            if assignment is not None:
                _visualize_assignment_segments(
                    vis, assignment, traj_cache, system.tf_manager,
                )
            else:
                # Explicit mode — show full trajectories
                for tpath, (eef, obj, jts, res, fname) in traj_cache.items():
                    stem = os.path.splitext(fname)[0]
                    visualize_trajectory(
                        vis, eef, res,
                        name=stem,
                        tf_manager=system.tf_manager,
                    )
                    if obj:
                        visualize_trajectory(
                            vis, obj, name=f"{stem}_object",
                            color=0xff00ff, tf_manager=system.tf_manager,
                        )

        # 4. Extract grasps per trajectory path
        # G = inv(T_ee) @ T_obj — object pose in EE frame
        logger.info("Extracting grasps from trajectory first waypoints...")
        grasp_cache: dict = {}
        for tpath, (eef, obj, jts, res, fname) in traj_cache.items():
            if obj:
                g = extract_grasp_transform(eef[0], obj[0])
                if g is None:
                    logger.error("Failed to extract grasp from %s.", fname)
                    return
                grasp_cache[tpath] = g
            else:
                logger.error("Object trajectory missing in %s — cannot extract grasp.", fname)
                return

        # ------------------------------------------------------------------
        # 5. Execution Loop
        # ------------------------------------------------------------------
        # Helper to look up the current segment's trajectory path from the
        # assignment or from the explicit trajectory mapping.
        def _current_path(robot_id: int, segment_idx: int) -> str:
            """Return the trajectory path (cache key) for *segment_idx*."""
            if assignment is not None:
                return assignment.trajectory_paths[segment_idx]
            return _explicit_paths[robot_id]

        if assignment is not None:
            current_robot = assignment.sequence[0]
            current_idx = 0
        else:
            # Explicit mode — find first feasible waypoint
            _, _, _, res0, _ = _get_traj_data(_explicit_paths[0])
            _, _, _, res1, _ = _get_traj_data(_explicit_paths[1])
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

        cur_path = _current_path(current_robot, current_idx)
        eef_cur, obj_cur, joints_cur, res_cur, _ = _get_traj_data(cur_path)
        current_grasp = grasp_cache[cur_path]
        
        obj_pose = obj_cur[current_idx]
        grasp_primitive = ObjectGraspPrimitive(object_pose=obj_pose, grasp_transform=current_grasp)
        grasp_traj_data = grasp_primitive.generate_trajectory()
        
        # Use planning joint config as IK hint for initial grasp
        q_warm_start_grasp = joints_cur[current_idx] if joints_cur[current_idx] is not None else None
        
        logger.info(f"Robot {current_robot} moving to pre-grasp pose...")
        current_arm.move_to_pose(grasp_traj_data["pregrasp"], q_warm_start=q_warm_start_grasp)

        logger.info(f"Robot {current_robot} following grasp approach trajectory...")
        current_arm.follow_trajectory(grasp_traj_data["approach"], q_warm_start=q_warm_start_grasp)

        current_gripper.grasp()
        stabilise(system.env, system.env_lock)

        # Create RRT visualizer for the dual arm planner
        rrt_vis = None
        if exec_cfg.visualize and vis is not None:
            rrt_vis = RRTVisualizer(vis)

        # Total number of segments (from the first trajectory's length)
        n_segments = len(res_cur)

        while current_idx < n_segments - 1:
            cur_path = _current_path(current_robot, current_idx)
            eef_cur, obj_cur, joints_cur, res_cur, _ = _get_traj_data(cur_path)
            
            next_handover_idx = -1

            if assignment is not None:
                # ---- Assignment-driven handover detection ----------------
                for seg in range(current_idx + 1, len(assignment.sequence)):
                    if assignment.sequence[seg] != current_robot:
                        next_handover_idx = seg
                        break
            else:
                # ---- Heuristic fallback (original logic) -----------------
                _, _, _, res_other, _ = _get_traj_data(_explicit_paths[1 - current_robot])
                for i in range(current_idx + 1, len(res_cur)):
                    if not res_cur[i].is_feasible:
                        for j in reversed(range(current_idx, i)):
                            if res_cur[j].is_feasible and res_other[j].is_feasible:
                                next_handover_idx = j
                                break
                        break

                if next_handover_idx == -1:
                    last_feasible_other = find_last_feasible_point(res_other)
                    last_feasible_current = find_last_feasible_point(res_cur)
                    if (last_feasible_other is not None
                            and last_feasible_current is not None
                            and last_feasible_other > last_feasible_current):
                        for j in reversed(range(current_idx, last_feasible_current + 1)):
                            if res_cur[j].is_feasible and res_other[j].is_feasible:
                                next_handover_idx = j
                                break
            
            if next_handover_idx != -1 and next_handover_idx > current_idx:
                # -------------------------------------------------------
                # PARALLEL EXECUTION: While arm A follows its trajectory
                # segment to the handover point, arm B simultaneously
                # moves to its handover pre-pose.
                # -------------------------------------------------------
                other_arm = system.right_arm if current_robot == 0 else system.left_arm
                other_gripper = system.right_gripper if current_robot == 0 else system.left_gripper

                # Resolve trajectory data for the current and next segments
                cur_path = _current_path(current_robot, current_idx)
                next_path = _current_path(1 - current_robot, next_handover_idx)
                _, obj_cur, joints_cur, _, _ = _get_traj_data(cur_path)
                _, _, joints_next, _, _ = _get_traj_data(next_path)

                grasp_transfer = grasp_cache[cur_path]
                grasp_receiver = grasp_cache[next_path]

                # Pre-compute the handover plan so we know arm B's target
                obj_pose_h = obj_cur[next_handover_idx]
                h_planner = DualArmHandoverPlanner(
                    grasp_transfer=grasp_transfer,
                    grasp_receiver=grasp_receiver,
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
                joint_segment = list(joints_cur[current_idx+1 : next_handover_idx+1])

                # Arm B's target: the handover receiver pre-pose
                receiver_pre_pose = h_traj["receiver_pre"][0]
                q_warm_start_receiver = (
                    joints_next[next_handover_idx]
                    if joints_next[next_handover_idx] is not None else None
                )

                logger.info(
                    f"PARALLEL: Robot {current_robot} following segment to handover at {next_handover_idx} "
                    f"while Robot {1 - current_robot} moves to handover pre-pose"
                )

                parallel_planner = DualArmParallelPlanner(
                    env=system.env,
                    arm_a=current_arm,
                    arm_b=other_arm,
                    env_lock=system.env_lock,
                    tf_manager=system.tf_manager,
                    collision_check=True,
                    manipulated_object=exp_cfg.task.manipulated_object,
                    rrt_vis=rrt_vis,
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
                last_idx = find_last_feasible_point(res_cur)
                if last_idx is not None and last_idx > current_idx:
                    logger.info(f"Robot {current_robot} following final segment to {last_idx}...")
                    joint_segment = list(joints_cur[current_idx+1 : last_idx+1])
                    current_arm.follow_joint_trajectory(joint_segment, gripper_action=1.0, smoothen=exec_cfg.smoothen, num_steps=200)
                    current_idx = last_idx
                break

        # 6. Release and Retreat
        logger.info(f"Robot {current_robot} releasing object and retreating...")
        cur_path = _current_path(current_robot, current_idx)
        eef_cur, _, _, _, _ = _get_traj_data(cur_path)
        release_pose = eef_cur[current_idx]
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
