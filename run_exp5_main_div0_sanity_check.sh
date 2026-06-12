#!/usr/bin/env bash
#
# Experiment 5: Sanity check the main distilled div0/S4 head.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_env.sh"

softwhere_init "exp5_main_div0_sanity_check"

DISTILLED="${DISTILLED:-softwhere_head_v10_sr_div0.pt}"
TL_SR_MODE="${TL_SR_MODE:-conv}"
VARIANT="${VARIANT:-v10}"
NUM_TOKENS="${NUM_TOKENS:-4}"
read -r -a K_VALUES_ARR <<< "${K_VALUES:-16 72 128 136}"
read -r -a NMS_DISTS_ARR <<< "${NMS_DISTS:-2}"

require_lw_file "$DISTILLED" "Run ./run_exp4_diversity_and_s_sweeps.sh first, or set DISTILLED=..."

run_lw resolution_parity.py \
  --tl-sr-mode "$TL_SR_MODE" \
  --variant "$VARIANT" \
  --diversity 0 \
  --stage eval \
  --distilled "$DISTILLED" \
  --eval-ade20k

run_lw experiment_softwhere.py \
  --distilled "$DISTILLED" \
  --tl-sr-mode "$TL_SR_MODE" \
  --variant "$VARIANT" \
  --num-tokens "$NUM_TOKENS"

run_lw nms_robustness_sweep.py \
  --head div0,"$DISTILLED",4,v10,conv \
  --k-values "${K_VALUES_ARR[@]}" \
  --nms-dists "${NMS_DISTS_ARR[@]}" \
  --out-csv softwhere_nms_robustness_main_div0.csv
