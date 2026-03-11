#!/bin/bash
# ============================================================================
# Step 3: Execute all valid assignments headlessly and collect metrics.
#
# For each (task, config) with a valid assignment_results.npz, runs
# dual_arm_handover_traj.py --headless --smoothen and saves metrics.
#
# Execution jobs are run in parallel (up to --workers concurrent processes).
# Each execution is an independent subprocess, so parallelism is safe.
#
# Results go to: results/<task>/<config_id>_exec.json
# Combined results are merged into results/<task>/<config_id>.json
#
# Usage:
#   ./scripts/paper_3_execute_all.sh                  # all tasks, all configs
#   ./scripts/paper_3_execute_all.sh --workers 2      # 2 parallel executions
#   ./scripts/paper_3_execute_all.sh --timeout 900    # 15-minute timeout per run
# ============================================================================

set -euo pipefail

# Kill all child processes on exit/interrupt
trap 'kill -- -$$ 2>/dev/null; wait' EXIT INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────
WORKERS=2
TIMEOUT=1200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Conda ─────────────────────────────────────────────────────────────────
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_arm_manipulation

TASKS=(
    "ycb_plate_table_to_shelf"
    "ycb_plate_shelf_to_table"
    "mug_inversion"
    "flatPlate_box_to_box"
)

EXEC_SCRIPT="scripts/pipeline/dual_arm_handover_traj.py"

echo "============================================================"
echo "  Paper Step 3: Headless Execution"
echo "  Tasks:   ${TASKS[*]}"
echo "  Workers: $WORKERS"
echo "  Timeout: ${TIMEOUT}s per run"
echo "  Time:    $(date)"
echo "============================================================"

mkdir -p results

# ── Collect all (task, config, assignment_file, experiment_json) tuples ───
# We build a list of jobs, then dispatch them in parallel.

JOBS=()        # Each entry: "task|config_id|config_label|assignment_file|experiment_json|data_dir_type"

# DPP configs: Proposed, Greedy, Single Robot, No Switch Penalty, No Manipulability
DPP_CONFIGS=(
    "Proposed"
    "Greedy_beam1"
    "Single_Robot"
    "No_Switching_Penalty_Cswitch0"
    "No_Manipulability_β0"
)

for task in "${TASKS[@]}"; do
    exp_json="experiments/${task}.json"

    # DPP-based configs
    for cfg_id in "${DPP_CONFIGS[@]}"; do
        asgn_file="data/${task}/configs/${cfg_id}/assignment_results.npz"
        if [[ -f "$asgn_file" ]]; then
            JOBS+=("${task}|${cfg_id}|${asgn_file}|${exp_json}|dpp")
        fi
    done

    # No Diversity (no-DPP data)
    asgn_file="data/${task}_no_dpp/configs/No_Diversity/assignment_results.npz"
    if [[ -f "$asgn_file" ]]; then
        JOBS+=("${task}|No_Diversity|${asgn_file}|${exp_json}|no_dpp")
    fi
done

echo ""
echo "  Found ${#JOBS[@]} execution jobs"

if [[ ${#JOBS[@]} -eq 0 ]]; then
    echo "  No assignments found — run Step 2 first."
    exit 1
fi

# ── Execute jobs in parallel ──────────────────────────────────────────────

TOTAL=${#JOBS[@]}
SUCCEEDED=0
FAILED=0
PIDS=()
JOB_INFOS=()

run_single_execution() {
    local job="$1"
    IFS='|' read -r task cfg_id asgn_file exp_json data_type <<< "$job"

    local results_subdir="results/${task}"
    mkdir -p "$results_subdir"
    local metrics_file="${results_subdir}/${cfg_id}_exec.json"
    local log_file="${results_subdir}/${cfg_id}_exec.log"

    echo "  [START] ${task} × ${cfg_id}"

    # Build the command — add --data-dir for no-DPP configs so trajectories
    # are loaded from the correct directory (data/<task>_no_dpp/).
    local cmd=(
        timeout "$TIMEOUT" python "$EXEC_SCRIPT"
        --experiment "$exp_json"
        --assignment "$asgn_file"
        --output-metrics "$metrics_file"
        --headless
        --smoothen
    )
    if [[ "$data_type" == "no_dpp" ]]; then
        cmd+=(--data-dir "data/${task}_no_dpp")
    fi

    "${cmd[@]}" > "$log_file" 2>&1
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "  [DONE]  ${task} × ${cfg_id}"
    elif [[ $rc -eq 124 ]]; then
        echo "  [TIMEOUT] ${task} × ${cfg_id} (${TIMEOUT}s)"
    else
        echo "  [FAIL]  ${task} × ${cfg_id} (exit $rc)"
    fi
    return $rc
}

# Dispatch with job-level parallelism
RUNNING=0
JOB_IDX=0

for job in "${JOBS[@]}"; do
    # Wait if at max workers
    while [[ $RUNNING -ge $WORKERS ]]; do
        # Wait for any child to finish
        wait -n 2>/dev/null || true
        RUNNING=$((RUNNING - 1))
    done

    run_single_execution "$job" &
    RUNNING=$((RUNNING + 1))
    JOB_IDX=$((JOB_IDX + 1))
done

# Wait for all remaining jobs
wait

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Execution Complete — $(date)"
echo "============================================================"

for task in "${TASKS[@]}"; do
    echo "  $task:"
    for exec_json in "results/${task}"/*_exec.json; do
        if [[ -f "$exec_json" ]]; then
            cfg_id=$(basename "$exec_json" _exec.json)
            # Extract task_success from JSON
            success=$(python3 -c "import json; d=json.load(open('$exec_json')); print(d.get('task_success', '?'))" 2>/dev/null || echo "?")
            collisions=$(python3 -c "import json; d=json.load(open('$exec_json')); print(d.get('n_collision_incidents', '?'))" 2>/dev/null || echo "?")
            echo "    $cfg_id: success=$success  collisions=$collisions"
        fi
    done
done

# ── Merge execution results into combined result JSONs ────────────────────
echo ""
echo "  Merging execution metrics into result files..."
python3 -c "
import json, os, sys
from pathlib import Path

tasks = ['ycb_plate_table_to_shelf', 'ycb_plate_shelf_to_table', 'mug_inversion', 'flatPlate_box_to_box']
results_root = Path('results')

merged = 0
for task in tasks:
    task_dir = results_root / task
    if not task_dir.is_dir():
        continue
    for exec_file in sorted(task_dir.glob('*_exec.json')):
        cfg_id = exec_file.stem.replace('_exec', '')
        result_file = task_dir / f'{cfg_id}.json'

        # Load execution metrics
        with open(exec_file) as f:
            exec_data = json.load(f)

        if result_file.is_file():
            # Merge into existing result
            with open(result_file) as f:
                result = json.load(f)
            result['execution'] = exec_data
        else:
            # Create minimal result
            result = {
                'experiment': task,
                'config_label': exec_data.get('config_label', cfg_id),
                'config_id': cfg_id,
                'execution': exec_data,
            }

        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        merged += 1

print(f'  Merged {merged} execution results.')
"

echo ""
echo "  Run 'python scripts/evaluation/paper_metrics.py' to view final metrics."
