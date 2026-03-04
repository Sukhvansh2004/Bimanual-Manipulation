#!/bin/bash
# ============================================================================
# Step 2: Run segment assignment for all paper configs × 4 tasks.
#
# Paper configs (from scripts/evaluation/configs.py):
#   1. Proposed        (α=10, β=1, beam=unlimited)
#   2. Greedy          (beam=1)
#   3. Single Robot    (switch_cost=inf)
#   4. No Switch Pen.  (switch_cost=0)
#   5. No Manipulability (β=0)
#   -- These 5 run on DPP trajectory data (data/<task>/)
#
#   6. No Diversity    (use_dpp=false → separate trajectory data)
#   -- Runs on non-DPP trajectory data (data/<task>_no_dpp/)
#
# Each assignment writes to data/<task>[_no_dpp]/configs/<config_id>/assignment_results.npz
# Results (planning metrics) go to results/<task>/<config_id>.json
#
# Usage:
#   ./scripts/paper_2_segment_assignment.sh             # all tasks, all configs
#   ./scripts/paper_2_segment_assignment.sh --workers 4  # parallel workers
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────
WORKERS=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
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

echo "============================================================"
echo "  Paper Step 2: Segment Assignment"
echo "  Tasks:   ${TASKS[*]}"
echo "  Workers: $WORKERS"
echo "  Time:    $(date)"
echo "============================================================"

mkdir -p results

# ── Run the paper evaluation script in plan-only mode ─────────────────────
# This handles Proposed, Greedy, Single Robot, NoSwitch, NoManip automatically
# using the DPP trajectory data.
echo ""
echo "================================================================"
echo "  A) Assignment for DPP configs (Proposed, Greedy, Single,"
echo "     No Switch Penalty, No Manipulability)"
echo "================================================================"

python scripts/evaluation/run_paper_evaluation.py \
    --plan-only \
    --workers "$WORKERS" \
    2>&1 | tee results/paper_step2a_assignment.log

# ── Now handle "No Diversity" (no-DPP data) ──────────────────────────────
# This config needs trajectories from data/<task>_no_dpp/.
# We run segment_assignment.py directly, pointing --data-dir there.
echo ""
echo "================================================================"
echo "  B) Assignment for No Diversity ablation (no-DPP data)"
echo "================================================================"

SEGMENT_ASSIGNMENT="scripts/pipeline/segment_assignment.py"
FAILED=0

run_no_diversity_assignment() {
    local task="$1"
    local exp_json="experiments/${task}.json"
    local data_dir="data/${task}_no_dpp"
    local config_id="No_Diversity"
    local out_dir="${data_dir}/configs/${config_id}"

    if [[ ! -d "$data_dir" ]]; then
        echo "  SKIP $task — no-DPP data dir not found: $data_dir"
        return 1
    fi

    # Check that we have robot trajectories
    local n_trajs
    n_trajs=$(ls "$data_dir"/robot*_trajectory_*.npz 2>/dev/null | wc -l)
    if [[ "$n_trajs" -eq 0 ]]; then
        echo "  SKIP $task — no robot trajectories in $data_dir"
        return 1
    fi

    echo "  Running No Diversity assignment for $task ($n_trajs trajectories)..."
    mkdir -p "$out_dir"

    # No Diversity = use_dpp was False (already done at traj gen),
    # at assignment level we use default params (same as Proposed).
    python "$SEGMENT_ASSIGNMENT" \
        --experiment "$exp_json" \
        --data-dir "$data_dir" \
        --no-plot \
        --clearance-radius 0.01 \
        2>&1 | tee "results/${task}_no_diversity_assignment.log"

    # Move assignment result into the config subdir
    if [[ -f "${data_dir}/assignment_results.npz" ]]; then
        mv "${data_dir}/assignment_results.npz" "${out_dir}/assignment_results.npz"
        echo "  ✓ $task No Diversity assignment saved to $out_dir"
    else
        echo "  ✗ $task No Diversity — no assignment_results.npz produced"
        return 1
    fi
}

for task in "${TASKS[@]}"; do
    echo ""
    echo "── $task (No Diversity) ──────────────────────────────────"
    run_no_diversity_assignment "$task" || FAILED=$((FAILED+1))
done

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Segment Assignment Complete — $(date)"
echo "============================================================"

for task in "${TASKS[@]}"; do
    echo "  $task:"
    # Count config dirs with assignment_results.npz
    for cfg_dir in "data/$task/configs/"*/; do
        if [[ -f "${cfg_dir}assignment_results.npz" ]]; then
            cfg_name=$(basename "$cfg_dir")
            echo "    ✓ $cfg_name"
        fi
    done 2>/dev/null
    # No Diversity (no-DPP)
    if [[ -f "data/${task}_no_dpp/configs/No_Feasibility_Filter/assignment_results.npz" ]]; then
        echo "    ✓ No_Feasibility_Filter (no-DPP)"
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo "  WARNING: $FAILED No Diversity assignment(s) failed!"
    exit 1
else
    echo "  All assignments succeeded."
fi
