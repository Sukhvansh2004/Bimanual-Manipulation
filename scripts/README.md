# Scripts

## Directory Structure

```
scripts/
├── pipeline/                          # Core planning & execution pipeline
├── evaluation/                        # Experiment orchestration, metrics & analysis
├── utils/                             # Visualization & interactive tools
├── demos/                             # Simple demos & teleoperation
├── generate_trajs.sh                  # Step 1: Generate EEF trajectories
├── segment_assignment.sh              # Step 2: Run segment assignment
└── execution.sh                       # Step 3: Headless execution & metric collection
```

---

## Evaluation Pipeline

The evaluation pipeline consists of 3 steps. Each step can be run independently; later steps consume the outputs of earlier ones.

### Usage

All shell scripts support a `--paper` flag to run configurations used in the paper, or can be run for all experiments.

#### 1. Generate Trajectories
Generates candidate robot trajectories for a given object trajectory.
```bash
./scripts/generate_trajs.sh [--paper] [--workers N] [--only EXP_NAME]
```

#### 2. Segment Assignment
Solves the optimization problem to assign segments to robots.
```bash
./scripts/segment_assignment.sh [--paper] [--workers N] [--only EXP_NAME]
```

#### 3. Execution
Executes the assigned trajectories in a headless simulation to collect metrics.
```bash
./scripts/execution.sh [--paper] [--workers N] [--only EXP_NAME]
```

### Analysis
After running the pipeline, analyze the results:
```bash
python scripts/evaluation/analyze.py
```

### Iterative Pipeline (Robustness)
To run the full pipeline multiple times with different random seeds:
```bash
python scripts/evaluation/pipeline_iterative.py --experiments mug_inversion --iterations 5
```

---

## Paper Evaluation (IROS 2026)

To reproduce paper results:
1. `./scripts/generate_trajs.sh --paper`
2. `./scripts/segment_assignment.sh --paper`
3. `./scripts/execution.sh --paper`
4. `python scripts/evaluation/analyze.py`

### Paper Tasks (4)
- `ycb_plate_table_to_shelf`
- `ycb_plate_shelf_to_table`
- `mug_inversion`
- `flatPlate_box_to_box`

### Paper Configs (6)
- **Proposed** (full method)
- **Greedy** (beam=1)
- **Single Robot** (switch_cost=∞)
- **No Diversity** (use_dpp=False)
- **No Switching Penalty** (C_switch=0)
- **No Manipulability** (β=0)
