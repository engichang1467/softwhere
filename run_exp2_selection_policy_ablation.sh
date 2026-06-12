#!/usr/bin/env bash
#
# Experiment 2: Selection-policy ablation for the div1 SR head.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_env.sh"

softwhere_init "exp2_selection_policy_ablation"

DISTILLED="${DISTILLED:-softwhere_head_v10_sr_div1.pt}"
TL_SR_MODE="${TL_SR_MODE:-conv}"
VARIANT="${VARIANT:-v10}"

require_lw_file "$DISTILLED" "Run ./run_exp1_resolution_parity.sh first, or set DISTILLED=..."

run_lw selection_policy_ablation.py \
  --distilled "$DISTILLED" \
  --tl-sr-mode "$TL_SR_MODE" \
  --variant "$VARIANT"
