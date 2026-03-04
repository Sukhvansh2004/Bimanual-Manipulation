# Scripts

## Directory Structure

```
scripts/
├── pipeline/          # Core planning & execution pipeline
├── evaluation/        # Experiment orchestration, metrics & analysis
├── utils/             # Visualization & interactive tools
├── demos/             # Simple demos & teleoperation
├── generate_all_trajs.py   # Batch trajectory generation
├── run_paper_eval.sh       # Shell wrapper for paper evaluation
└── run_overnight.sh        # Unattended full pipeline runner
```

---

## `pipeline/` — Core Pipeline

These scripts implement the plan-assign-execute pipeline. They are typically called via the evaluation orchestrators, not directly.

| Script | Purpose |
|--------|---------|
| `trajectory_generation.py` | Single-arm planner: loads experiment → generates object trajectory → grasp generation → feasibility filtering → saves trajectories |
| `dual_trajectory_generation.py` | Dual-arm planner: generates grasps for both robots, evaluates simultaneous feasibility of all (N×M) pairs via multiprocessing |
| `segment_assignment.py` | Optimal segment-to-robot assignment via DP: loads trajectories, computes cost matrix (reachability + manipulability + switching), solves, saves `assignment_results.npz` |
| `find_feasible_pair.py` | Finds jointly-feasible (robot0, robot1) trajectory pairs from pre-computed per-waypoint feasibility data |
| `dual_arm_handover_traj.py` | Executes the assigned handover trajectory in MuJoCo simulation with smoothing, gripper actions, and collision tracking |

**Typical pipeline flow:**
```
trajectory_generation.py → segment_assignment.py → dual_arm_handover_traj.py
```

---

## `evaluation/` — Experiment Orchestration

Scripts for running batch experiments, computing metrics, and generating paper results.

| Script | Purpose |
|--------|---------|
| `run_paper_evaluation.py` | Main orchestrator for the 4 paper tasks × 5 configs (planning → execution → metrics). Run via `run_paper_eval.sh` |
| `run_all.py` | General-purpose batch orchestrator: runs assignment planning for every (experiment, config) pair with optional execution and analysis |
| `configs.py` | Defines all ablation/baseline configurations (Proposed, Greedy, Single Robot, No Switch Penalty, No Manipulability, etc.) |
| `metrics.py` | Data classes (`PlanningMetrics`, `ExecutionMetrics`, `ExperimentResult`) and metric extraction helpers |
| `paper_metrics.py` | Loads all result JSONs and prints paper-ready quantitative tables |
| `results_store.py` | JSON persistence for `ExperimentResult` objects; numpy-safe encoder; summary builder |
| `analyze.py` | Generates summary tables (console/LaTeX/CSV), bar charts, ablation plots, and heatmaps |

**Quick usage:**
```bash
# Full paper evaluation (plan + execute + metrics)
./scripts/run_paper_eval.sh

# Planning only
./scripts/run_paper_eval.sh --plan-only

# Execution only (needs existing assignments)
./scripts/run_paper_eval.sh --exec-only

# Print metrics from existing results
./scripts/run_paper_eval.sh --metrics-only
```

---

## `utils/` — Visualization & Interactive Tools

| Script | Purpose |
|--------|---------|
| `visualize_robot_trajectory.py` | Loads a trajectory `.npz` and renders EEF path + robot meshes (green/red by feasibility) in MeshCat |
| `dual_arm_manipulation.py` | Launches services + interactive shell (detect, grasp, go_to_pose, etc.) for live exploration |
| `launch_env.py` | Creates the dual-arm environment from an experiment JSON, optionally with random actions |
| `visualize_rrt.py` | Visualizes RRT-Connect search tree, samples, raw/smoothed paths in MeshCat |
| `transform_visualizer_meshcat.py` | Streams all TF frames from an auto-created env to MeshCat (RViz-like) |

---

## `demos/` — Simple Demos

| Script | Purpose |
|--------|---------|
| `go_to_pose.py` | Demo of `DualArmPoseController` — moves two Pandas to static poses |
| `gripper_move.py` | Simple open/close gripper test sequence |
| `teleop_dual_arm.py` | Keyboard teleoperation of both robots |

---

## Top-Level Scripts

| Script | Purpose |
|--------|---------|
| `generate_all_trajs.py` | Batch runner: invokes `trajectory_generation.py` + `segment_assignment.py` for every experiment JSON |
| `run_paper_eval.sh` | Shell wrapper: activates conda, calls `run_paper_evaluation.py`, tees to log |
| `run_overnight.sh` | Full unattended pipeline: backup data → regen trajectories → planning × configs → execution → metrics |
