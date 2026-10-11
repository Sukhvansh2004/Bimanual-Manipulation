#!/bin/bash
# Leave-one-seed-out cross-validation of the learned handover filter.
#
# For each seed S in SEEDS:
#   1. Train a transformer on the exact + beam handover logs of every other
#      seed (never on S) -> models/cv/fold_sS.pt
#   2. Run exact DP with that fold's filter at each cost slack in KAPPAS on
#      the instances of S that unfiltered exact DP solves.
# Requires the base campaign (configs beam + exact) to have finished for all
# seeds. Results are compared against the exact-DP runs by
# analyze_campaign.py (configs exact_learned_cv_k080 / _k100).
#
# Usage: scripts/learned_filter/run_cv.sh

set -uo pipefail

PY="$HOME/anaconda3/envs/dual_arm_manipulation/bin/python"
SEEDS=(1001 1002 1003 1004 1005)
KAPPAS=(0.8 1.0)
TASKS=(flatPlate_table_to_shelf ycb_plate_shelf_to_table mug_inversion flatPlate_box_to_box)

cd "$(dirname "$0")/../.."
mkdir -p models/cv

# Print the tasks whose unfiltered exact DP found a plan for seed $1.
solvable_tasks() {
    "$PY" - "$1" "${TASKS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
seed, tasks = sys.argv[1], sys.argv[2:]
for task in tasks:
    rec = Path(f"results/campaign/{task}/s{seed}/exact.json")
    if rec.exists() and json.loads(rec.read_text())["planning_success"]:
        print(task)
PYEOF
}

for seed in "${SEEDS[@]}"; do
    train=()
    for other in "${SEEDS[@]}"; do
        [ "$other" = "$seed" ] && continue
        train+=(results/campaign/*/s"$other"/handover_log_exact.npz results/campaign/*/s"$other"/handover_log_beam.npz)
    done
    echo "=== fold s${seed}: training on ${#train[@]} log patterns"
    "$PY" scripts/learned_filter/train_filter.py --arch transformer \
        --train "${train[@]}" \
        --test results/campaign/*/s"$seed"/handover_log_exact.npz \
        --output models/cv/fold_s"$seed".pt

    mapfile -t tasks < <(solvable_tasks "$seed")
    [ "${#tasks[@]}" -eq 0 ] && { echo "seed ${seed}: no solvable instances"; continue; }
    for kappa in "${KAPPAS[@]}"; do
        tag="_cv_k$(printf '%03d' "$(echo "$kappa * 100 / 1" | bc)")"
        "$PY" scripts/learned_filter/campaign.py --tasks "${tasks[@]}" --seeds "$seed" \
            --plan-timeout 3600 --configs exact_learned \
            --model models/cv/fold_s"$seed".pt --threshold 0.1 --cost-slack "$kappa" --tag "$tag"
    done
done
echo "Cross-validation finished at $(date)"
