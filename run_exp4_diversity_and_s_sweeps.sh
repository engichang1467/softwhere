#!/usr/bin/env bash
#
# Experiment 4: Diversity and fovea-count sweeps.
# This script assumes the div1/S4 head exists from experiment 1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_env.sh"

softwhere_init "exp4_diversity_and_s_sweeps"

TL_SR_MODE="${TL_SR_MODE:-conv}"
VARIANT="${VARIANT:-v10}"
DIVERSITIES="${DIVERSITIES:-0 0.5 2}"
read -r -a DIVERSITY_ARR <<< "$DIVERSITIES"
read -r -a K_VALUES_ARR <<< "${K_VALUES:-16 72 128 136}"
read -r -a NMS_DISTS_ARR <<< "${NMS_DISTS:-2}"

require_lw_file "softwhere_head_v10_sr_div1.pt" \
  "Run ./run_exp1_resolution_parity.sh first; exp4 compares against the existing div1/S4 head."

log "distilling diversity heads: $DIVERSITIES"
for diversity in "${DIVERSITY_ARR[@]}"; do
  run_lw resolution_parity.py \
    --tl-sr-mode "$TL_SR_MODE" \
    --variant "$VARIANT" \
    --diversity "$diversity" \
    --stage distill
done

run_lw nms_robustness_sweep.py \
  --head div0,softwhere_head_v10_sr_div0.pt,4,v10,conv \
  --head div0p5,softwhere_head_v10_sr_div0.5.pt,4,v10,conv \
  --head div1,softwhere_head_v10_sr_div1.pt,4,v10,conv \
  --head div2,softwhere_head_v10_sr_div2.pt,4,v10,conv \
  --k-values "${K_VALUES_ARR[@]}" \
  --nms-dists "${NMS_DISTS_ARR[@]}" \
  --out-csv softwhere_nms_robustness_diversity.csv

log "distilling fovea-count heads: S=2 and S=8"
run_lw resolution_parity.py \
  --tl-sr-mode "$TL_SR_MODE" \
  --variant "$VARIANT" \
  --num-tokens 2 \
  --diversity 1 \
  --stage distill \
  --tag S2

run_lw resolution_parity.py \
  --tl-sr-mode "$TL_SR_MODE" \
  --variant "$VARIANT" \
  --num-tokens 8 \
  --diversity 1 \
  --stage distill \
  --tag S8

run_lw nms_robustness_sweep.py \
  --head S2,softwhere_head_v10_sr_div1_S2.pt,2,v10,conv \
  --head S4,softwhere_head_v10_sr_div1.pt,4,v10,conv \
  --head S8,softwhere_head_v10_sr_div1_S8.pt,8,v10,conv \
  --k-values "${K_VALUES_ARR[@]}" \
  --nms-dists "${NMS_DISTS_ARR[@]}" \
  --out-csv softwhere_nms_robustness_S.csv
