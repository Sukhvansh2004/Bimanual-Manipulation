# Scripts

## Directory Structure

```
scripts/
├── pipeline/                          # Core planning & execution pipeline
├── evaluation/                        # Experiment orchestration, metrics & analysis
├── utils/                             # Visualization & interactive tools
├── demos/                             # Simple demos & teleoperation
├── paper_1_generate_trajectories.sh   # Step 1: Generate EEF trajectories (DPP + no-DPP)
├── paper_2_segment_assignment.sh      # Step 2: Run segment assignment for all configs
├── paper_3_execute_all.sh             # Step 3: Headless execution & metric collection
├── generate_all_trajs.py              # Batch trajectory generation
└── run_paper_eval.sh                  # Shell wrapper for paper evaluation
```

---

## Paper Evaluation Pipeline (3-Step)

The complete evaluation for the IROS 2026 paper is a 3-step pipeline.
Each step can be run independently; later steps consume the outputs of
earlier ones.

### Paper Tasks (4)

| Experiment JSON | Short name |
|---|---|
| `ycb_plate_table_to_shelf` | Table → Shelf |
| `ycb_plate_shelf_to_table` | Shelf → Table |
| `mug_inversion`            | Mug Inversion |
| `flatPlate_box_to_box`     | Box → Box |

### Paper Configs (6)

Defined in `evaluation/configs.py`. Table 1 (method comparison) and
Table 2 (ablation study) share the same set:

| Config | Key change | Data dir |
|---|---|---|
| **Proposed** (full method) | beam=unlimited, α=10, β=1 | `data/<task>/` |
| **Greedy** | beam=1 | `data/<task>/` |
| **Single Robot** | switch_cost=∞ | `data/<task>/` |
| **No Diversity** | use_dpp=False (top-N by score) | `data/<task>_no_dpp/` |
| **No Switching Penalty** | C_switch=0 | `data/<task>/` |
| **No Manipulability** | β=0 | `data/<task>/` |

> **No Diversity** is the only config that uses different trajectory data.
> Trajectories are generated without DPP entropy maximization and stored
> in `data/<task>_no_dpp/`. The pipeline handles this automatically:
> `run_experiment()` detects `use_dpp=False` and auto-switches the data
> directory.

### Data Layout

```
data/
├── ycb_plate_table_to_shelf/           # DPP trajectories
│   ├── object_trajectory.npz
│   ├── robot0_trajectory_*.npz
│   ├── robot1_trajectory_*.npz
│   └── configs/
│       ├── Proposed/assignment_results.npz
│       ├── Greedy_beam1/assignment_results.npz
│       ├── Single_Robot/assignment_results.npz
│       ├── No_Switching_Penalty_Cswitch0/assignment_results.npz
│       └── No_Manipulability_β0/assignment_results.npz
├── ycb_plate_table_to_shelf_no_dpp/    # Non-DPP trajectories
│   ├── object_trajectory.npz           # (copied from DPP dir)
│   ├── robot0_trajectory_*.npz
│   ├── robot1_trajectory_*.npz
│   └── configs/
│       └── No_Diversity/assignment_results.npz
└── ...
```

### Step 1: Generate Trajectories

```bash
./scripts/paper_1_generate_trajectories.sh
```

For each task, generates end-effector trajectories **with DPP** (default)
and **without DPP** (for the No Diversity ablation). Object trajectories
are shared: the no-DPP dir copies `object_trajectory.npz` from the DPP
directory.

| Flag | Effect |
|---|---|
| `--workers N` | Number of parallel workers (default: 4) |
| `--skip-dpp` | Only generate no-DPP trajectories |
| `--skip-no-dpp` | Only generate DPP trajectories |

### Step 2: Segment Assignment

```bash
./scripts/paper_2_segment_assignment.sh
```

Runs `run_paper_evaluation.py --plan-only` which invokes the DP segment
assignment solver for every (task × config) pair. For `No Diversity`,
the auto-switching in `run_experiment()` points to `data/<task>_no_dpp/`.

Assignment results are saved to:
- `data/<task>/configs/<config_id>/assignment_results.npz`
- `data/<task>_no_dpp/configs/No_Diversity/assignment_results.npz`

Planning metrics are saved to:
- `results/<task>/<config_id>.json`

| Flag | Effect |
|---|---|
| `--workers N` | Parallel planning workers (default: 4) |

### Step 3: Headless Execution

```bash
./scripts/paper_3_execute_all.sh
```

For each (task, config) with a valid assignment, runs
`dual_arm_handover_traj.py --headless --smoothen` to execute in MuJoCo
and collect execution metrics (task success, handover success, collisions).

For `No Diversity`, passes `--data-dir data/<task>_no_dpp` so the
execution script loads trajectories from the correct directory.

| Flag | Effect |
|---|---|
| `--workers N` | Parallel execution workers (default: 2) |
| `--timeout S` | Per-run timeout in seconds (default: 600) |

### Quick: Full Pipeline

```bash
./scripts/paper_1_generate_trajectories.sh
./scripts/paper_2_segment_assignment.sh
./scripts/paper_3_execute_all.sh
python scripts/evaluation/paper_metrics.py   # print paper tables
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
| `run_paper_evaluation.py` | Main orchestrator for the 4 paper tasks × 6 configs (planning → execution → metrics). Run via `run_paper_eval.sh` |
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

