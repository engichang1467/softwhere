#!/usr/bin/env bash
#
# Experiment 3: NMS robustness sweep for the diversity=1 SR head.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_env.sh"

softwhere_init "exp3_nms_robustness_div1"

HEAD_FILE="${HEAD_FILE:-softwhere_head_v10_sr_div1.pt}"
HEAD_SPEC="${HEAD_SPEC:-div1,$HEAD_FILE,4,v10,conv}"
OUT_CSV="${OUT_CSV:-softwhere_nms_robustness_div1.csv}"
read -r -a K_VALUES_ARR <<< "${K_VALUES:-16 72 128 136}"
read -r -a NMS_DISTS_ARR <<< "${NMS_DISTS:-1 2 3 4}"

require_lw_file "$HEAD_FILE" "Run ./run_exp1_resolution_parity.sh first, or set HEAD_FILE=..."

run_lw nms_robustness_sweep.py \
  --head "$HEAD_SPEC" \
  --k-values "${K_VALUES_ARR[@]}" \
  --nms-dists "${NMS_DISTS_ARR[@]}" \
  --out-csv "$OUT_CSV"
