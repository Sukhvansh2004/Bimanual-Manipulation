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
Both standard (DPP diversity) and no-DPP (top-N by score) variants are generated
so that the `No Diversity` ablation config has the data it needs.
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

# Paper configs only (Proposed, Greedy, Single Robot, No Diversity, ...):
python scripts/evaluation/pipeline_iterative.py --paper --iterations 5

# Resume an interrupted run (skips completed iterations):
python scripts/evaluation/pipeline_iterative.py --paper --iterations 5 --resume
```

Each iteration generates both standard and no-DPP trajectories, then runs
assignment + execution for all requested configs.

### Alpha/Beta Grid Search (Assignment Weights)
Sweep assignment weights and compare task success + tracking metrics:
```bash
# Uses default sweep ranges:
python scripts/evaluation/alpha_beta_grid.py --paper

# Custom sweep for selected experiments:
python scripts/evaluation/alpha_beta_grid.py \
	--only flatPlate_box_to_box mug_inversion \
	--alpha-values 5 10 20 \
	--beta-values 0 0.5 1 2

# Resume after interruption:
python scripts/evaluation/alpha_beta_grid.py --paper --resume

# Parallel sweep with 4 workers:
python scripts/evaluation/alpha_beta_grid.py --paper --workers 4
```

Notes:
- The script does not regenerate trajectories.
- All alpha/beta pairs reuse the same pre-generated object trajectories, ensuring fair comparison.
- `--workers N` runs `(experiment, alpha, beta)` jobs in parallel.
- `--resume` skips completed `(experiment, alpha, beta)` jobs using existing result JSONs.
- Outputs are written to `results/alpha_beta_grid/` by default:
	- `alpha_beta_grid_details.csv`
	- `alpha_beta_grid_summary.csv`
	- `alpha_beta_grid_summary.json`

---
