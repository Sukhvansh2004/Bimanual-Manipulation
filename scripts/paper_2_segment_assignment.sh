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
#   6. No Diversity    (use_dpp=false → separate trajectory data)
#
# Configs 1–5 run on DPP trajectory data (data/<task>/).
# Config  6 (No Diversity) uses data/<task>_no_dpp/ and is handled
#   automatically by run_paper_evaluation.py via auto-switching in
#   run_experiment().
#
# Each assignment writes to data/<task>[_no_dpp]/configs/<config_id>/assignment_results.npz
# Results (planning metrics) go to results/<task>/<config_id>.json
#
# Usage:
#   ./scripts/paper_2_segment_assignment.sh             # all tasks, all configs
#   ./scripts/paper_2_segment_assignment.sh --workers 4  # parallel workers
# ============================================================================

set -euo pipefail

# Kill all child processes on exit/interrupt
trap 'kill -- -$$ 2>/dev/null; wait' EXIT INT TERM

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
# This handles ALL paper configs: Proposed, Greedy, Single Robot,
# No Switch Penalty, No Manipulability, and No Diversity.
# No Diversity is auto-switched to data/<task>_no_dpp/ by run_experiment().
echo ""
echo "================================================================"
echo "  Running assignment for ALL paper configs"
echo "  (incl. No Diversity → auto-switches to no-DPP data)"
echo "================================================================"

python scripts/evaluation/run_paper_evaluation.py \
    --plan-only \
    --resume \
    --workers "$WORKERS" \
    2>&1 | tee results/paper_step2_assignment.log

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
    if [[ -f "data/${task}_no_dpp/configs/No_Diversity/assignment_results.npz" ]]; then
        echo "    ✓ No_Diversity (no-DPP)"
    fi
done

echo "  Done."
