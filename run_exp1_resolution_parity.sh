#!/usr/bin/env bash
#
# Experiment 1: Resolution-Parity TokenLearner-SR.
# Produces softwhere_head_v10_sr_div1.pt and runs ADE20K coverage eval.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/experiment_env.sh"

softwhere_init "exp1_resolution_parity"

TL_SR_MODE="${TL_SR_MODE:-conv}"
VARIANT="${VARIANT:-v10}"
DIVERSITY="${DIVERSITY:-1}"

run_lw resolution_parity.py \
  --tl-sr-mode "$TL_SR_MODE" \
  --variant "$VARIANT" \
  --diversity "$DIVERSITY" \
  --stage both \
  --eval-ade20k
