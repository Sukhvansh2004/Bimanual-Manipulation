#!/bin/bash
# ============================================================================
# Generate EEF trajectories for experiments.
#
# Usage:
#   ./scripts/generate_trajs.sh              # all experiments
#   ./scripts/generate_trajs.sh --paper      # only paper experiments
#   ./scripts/generate_trajs.sh --workers 4  # override worker count
#   ./scripts/generate_trajs.sh --only exp1  # only specific experiments
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────
WORKERS=4
PAPER_MODE=false
ONLY_EXPS=()

# ── Parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers) WORKERS="$2"; shift 2 ;;
        --paper) PAPER_MODE=true; shift ;;
        --only) shift; while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do ONLY_EXPS+=("$1"); shift; done ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Resolve Tasks ─────────────────────────────────────────────────────────
if [[ "$PAPER_MODE" == true ]]; then
    TASKS=("ycb_plate_table_to_shelf" "ycb_plate_shelf_to_table" "mug_inversion" "flatPlate_box_to_box")
elif [[ ${#ONLY_EXPS[@]} -gt 0 ]]; then
    TASKS=("${ONLY_EXPS[@]}")
else
    # All .json files in experiments/
    TASKS=($(ls experiments/*.json | xargs -n 1 basename | sed 's/\.json//'))
fi

TRAJ_GEN="scripts/pipeline/trajectory_generation.py"

echo "============================================================"
echo "  Trajectory Generation"
echo "  Tasks:   ${TASKS[*]}"
echo "  Workers: $WORKERS"
echo "  Paper:   $PAPER_MODE"
echo "============================================================"

clean_robot_trajs() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        echo "  Cleaning robot trajectories in $dir ..."
        find "$dir" -name 'robot*_trajectory_*.npz' -delete
        rm -rf "$dir/configs"
    fi
}

for task in "${TASKS[@]}"; do
    echo ""
    echo "── $task ────────────────────────────────────────────────"
    
    # Standard run (DPP)
    DATA_DIR="data/$task"
    clean_robot_trajs "$DATA_DIR"
    python "$TRAJ_GEN" --experiment "experiments/${task}.json" --save-traj --reuse-object-traj --no-meshcat --workers "$WORKERS"
    
    # Generate No-DPP trajectories (needed by the No Diversity ablation config).
    echo "── $task (No-DPP) ──────────────────────────────────────"
    DATA_DIR_NO_DPP="data/${task}_no_dpp"
    clean_robot_trajs "$DATA_DIR_NO_DPP"
    python "$TRAJ_GEN" --experiment "experiments/${task}.json" --save-traj --reuse-object-traj --no-meshcat --workers "$WORKERS" --no-dpp
done

echo "Done."
