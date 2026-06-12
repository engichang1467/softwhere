#!/usr/bin/env bash
#
# Experiment 6: Mini end-to-end CLS training from the main div0/S4 head.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_env.sh"

softwhere_init "exp6_mini_e2e_cls"

INIT_HEAD="${INIT_HEAD:-softwhere_head_v10_sr_div0.pt}"
OUT_HEAD="${OUT_HEAD:-softwhere_head_v10_sr_div0_mini_e2e_cls.pt}"
TL_SR_MODE="${TL_SR_MODE:-conv}"
VARIANT="${VARIANT:-v10}"
NUM_TOKENS="${NUM_TOKENS:-4}"
STEPS="${STEPS:-1000}"
LAMBDA_CLS="${LAMBDA_CLS:-1}"
LAMBDA_PAT="${LAMBDA_PAT:-0}"
LAMBDA_MAP="${LAMBDA_MAP:-1}"
LAMBDA_DIV="${LAMBDA_DIV:-0}"
read -r -a K_VALUES_ARR <<< "${K_VALUES:-16 72 128 136}"
read -r -a NMS_DISTS_ARR <<< "${NMS_DISTS:-2}"

require_lw_file "$INIT_HEAD" "Run ./run_exp4_diversity_and_s_sweeps.sh first, or set INIT_HEAD=..."

run_lw mini_end_to_end.py \
  --tl-sr-mode "$TL_SR_MODE" \
  --variant "$VARIANT" \
  --num-tokens "$NUM_TOKENS" \
  --init-head "$INIT_HEAD" \
  --steps "$STEPS" \
  --lambda-cls "$LAMBDA_CLS" \
  --lambda-pat "$LAMBDA_PAT" \
  --lambda-map "$LAMBDA_MAP" \
  --lambda-div "$LAMBDA_DIV" \
  --out "$OUT_HEAD"

run_lw nms_robustness_sweep.py \
  --head div0,"$INIT_HEAD",4,v10,conv \
  --head div0_e2e_cls,"$OUT_HEAD",4,v10,conv \
  --k-values "${K_VALUES_ARR[@]}" \
  --nms-dists "${NMS_DISTS_ARR[@]}" \
  --out-csv softwhere_nms_robustness_div0_vs_e2e_cls.csv
