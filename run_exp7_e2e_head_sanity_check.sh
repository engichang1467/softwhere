#!/usr/bin/env bash
#
# Experiment 7: Sanity check the mini E2E CLS head.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_env.sh"

softwhere_init "exp7_e2e_head_sanity_check"

BASE_HEAD="${BASE_HEAD:-softwhere_head_v10_sr_div0.pt}"
E2E_HEAD="${E2E_HEAD:-softwhere_head_v10_sr_div0_mini_e2e_cls.pt}"
TL_SR_MODE="${TL_SR_MODE:-conv}"
VARIANT="${VARIANT:-v10}"
NUM_TOKENS="${NUM_TOKENS:-4}"
read -r -a K_VALUES_ARR <<< "${K_VALUES:-16 72 128 136}"
read -r -a NMS_DISTS_ARR <<< "${NMS_DISTS:-2}"

require_lw_file "$BASE_HEAD" "Run ./run_exp4_diversity_and_s_sweeps.sh first, or set BASE_HEAD=..."
require_lw_file "$E2E_HEAD" "Run ./run_exp6_mini_e2e_cls.sh first, or set E2E_HEAD=..."

run_lw resolution_parity.py \
  --tl-sr-mode "$TL_SR_MODE" \
  --variant "$VARIANT" \
  --num-tokens "$NUM_TOKENS" \
  --stage eval \
  --distilled "$E2E_HEAD" \
  --eval-ade20k

run_lw experiment_softwhere.py \
  --distilled "$E2E_HEAD" \
  --tl-sr-mode "$TL_SR_MODE" \
  --variant "$VARIANT" \
  --num-tokens "$NUM_TOKENS"

run_lw nms_robustness_sweep.py \
  --head div0,"$BASE_HEAD",4,v10,conv \
  --head e2e_cls,"$E2E_HEAD",4,v10,conv \
  --k-values "${K_VALUES_ARR[@]}" \
  --nms-dists "${NMS_DISTS_ARR[@]}" \
  --out-csv softwhere_nms_robustness_div0_vs_e2e_cls_sanity.csv
