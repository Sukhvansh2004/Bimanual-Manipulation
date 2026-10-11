#!/bin/bash
# Run the learned-filter and baseline configs on one held-out test seed.
#
# Waits until the base campaign (beam + exact) for the seed has finished,
# then runs, in priority order: exact/beam with the transformer filter,
# a feasibility-only ablation, the regrasp-graph baseline, a cost-slack
# sensitivity run, and the MLP architecture ablation.
#
# Usage: scripts/learned_filter/run_test_seed.sh <seed>

set -uo pipefail

SEED="$1"
PY="$HOME/anaconda3/envs/dual_arm_manipulation/bin/python"
CAMPAIGN="scripts/learned_filter/campaign.py"
TASKS=(mug_inversion flatPlate_table_to_shelf flatPlate_box_to_box ycb_plate_shelf_to_table)
TRANSFORMER="models/handover_filter_transformer.pt"
MLP="models/handover_filter_mlp.pt"

cd "$(dirname "$0")/../.."

# Wait for the base campaign on this seed: it has started (some exact result
# exists) and no campaign process for this seed is still running.
until compgen -G "results/campaign/*/s${SEED}/exact.json" > /dev/null \
        && ! pgrep -f "campaign.py .*--seeds ${SEED} " > /dev/null; do
    sleep 30
done
echo "Base campaign for seed ${SEED} finished at $(date)"

run() {
    "$PY" "$CAMPAIGN" --tasks "${TASKS[@]}" --seeds "$SEED" --plan-timeout 3600 "$@"
}

run --configs exact_learned beam_learned --model "$TRANSFORMER" --threshold 0.1 --cost-slack 0.65 --tag _k065
run --configs exact_learned --model "$TRANSFORMER" --threshold 0.1 --cost-slack 0 --tag _k0
"$PY" "$CAMPAIGN" --tasks "${TASKS[@]}" --seeds "$SEED" --plan-timeout 1800 --configs regrasp_graph
run --configs exact_learned --model "$TRANSFORMER" --threshold 0.1 --cost-slack 0.8 --tag _k080
run --configs exact_learned --model "$MLP" --threshold 0.1 --cost-slack 0.65 --tag _mlp_k065

echo "Test seed ${SEED} finished at $(date)"
