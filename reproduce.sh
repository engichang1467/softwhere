#!/usr/bin/env bash
#
# reproduce.sh — automate reproduction of the SoftWhere (P3) preliminary results
# documented in claude_logs/2026-06-07_P3_SoftWhere_REPRODUCE.md and reported in
# SoftWhere_Proposal.pdf.
#
# It (1) clones the two project forks at branch project-e-2, (2) builds ONE shared
# uv venv from softwhere/requirements.txt, (3) downloads the checkpoint + imagenette
# + ADE20K, then (4) runs the full spike pipeline: OTL tests, gradient-flow proof,
# multi-foveal viz, distillation sweeps (v1.0/v1.1), teacher-agreement proxy,
# map-diversity, ADE20K coverage, and (if python-docx is available) the proposal docx.
#
# Usage:
#   ./reproduce.sh                  # full run (downloads ADE20K ~923MB)
#   SKIP_ADE20K=1 ./reproduce.sh    # skip the ADE20K download + coverage step
#   CUDA_VISIBLE_DEVICES=0 ./reproduce.sh
#
# Override any UPPER_CASE config var via the environment.
#
# PREREQUISITE: the two forks at branch project-e-2 must already contain the spike
# code (modified lookwhere/modeling.py + OTL tokenlearner/modules.py, all new
# lookwhere/*.py scripts, and the OTL_PATH-derived data-path edits). Push your
# local repos to the engichang1467 forks on branch project-e-2 before running.
#
# NOTE: softwhere/requirements.txt does not include python-docx, so the final
# proposal-docx step is skipped automatically. Add it (uv pip install python-docx)
# if you want that artifact.
# -----------------------------------------------------------------------------
set -euo pipefail

# ============================= CONFIG =============================
# Repo URLs (as requested). NOTE: the local OTL origin is "OpenTokenLearner"
# (no hyphen) — if the hyphenated clone 404s, set OTL_URL accordingly.
LOOKWHERE_URL="${LOOKWHERE_URL:-git@github.com:engichang1467/lookwhere.git}"
OTL_URL="${OTL_URL:-git@github.com:engichang1467/Open-TokenLearner.git}"
BRANCH="${BRANCH:-project-e-2}"

# Where to clone + work. Defaults to this script's own directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SOFTWHERE_BASE:-$SCRIPT_DIR}"
LW_REPO="$BASE_DIR/lookwhere"
OTL_REPO="$BASE_DIR/Open-TokenLearner"

# One shared uv venv built from the combined lockfile next to this script.
VENV="${VENV:-$BASE_DIR/.venv}"
REQ="${REQ:-$SCRIPT_DIR/requirements.txt}"
PY_VERSION="${PY_VERSION:-3.12}"

# GPU: the original spike used one MIG A100 slice. Default to GPU 0; override by
# exporting CUDA_VISIBLE_DEVICES (e.g. a MIG UUID) before running.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Pipeline knobs (defaults reproduce the runbook numbers).
DIST_STEPS="${DIST_STEPS:-600}"
DIV_WEIGHTS="${DIV_WEIGHTS:-0 0.5 1 2}"
PROXY_IMAGES="${PROXY_IMAGES:-200}"
DIVERSITY_IMAGES="${DIVERSITY_IMAGES:-100}"
ADE_IMAGES="${ADE_IMAGES:-500}"
SKIP_ADE20K="${SKIP_ADE20K:-0}"

# Asset URLs.
CKPT_URL="https://huggingface.co/antofuller/lookwhere/resolve/main/lookwhere_dinov2.pt"
IMAGENETTE_URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
ADE_URL="https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

# ============================= HELPERS =============================
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "required tool not found: $1"; }

fetch() {  # fetch <url> <dest>  (skip if already present + non-empty)
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then ok "exists, skip download: $(basename "$dest")"; return; fi
  log "downloading $(basename "$dest")"
  curl -fL --retry 3 -C - -o "$dest" "$url" || die "download failed: $url"
}

clone_or_update() {  # clone_or_update <url> <dir> <branch>
  local url="$1" dir="$2" branch="$3"
  if [[ -d "$dir/.git" ]]; then
    log "updating $(basename "$dir")"
    git -C "$dir" fetch origin "$branch"
    git -C "$dir" checkout "$branch"
    git -C "$dir" reset --hard "origin/$branch"
  else
    log "cloning $(basename "$dir") @ $branch"
    git clone --branch "$branch" "$url" "$dir" \
      || die "clone failed for $url (branch $branch). Check the URL/branch."
  fi
}

# ============================= PRECHECKS =============================
need git; need uv; need curl; need unzip; need tar
[[ -f "$REQ" ]] || die "requirements file not found: $REQ"
mkdir -p "$BASE_DIR"
log "config"
echo "    BASE_DIR=$BASE_DIR"
echo "    venv=$VENV  (python $PY_VERSION, from $(basename "$REQ"))"
echo "    branch=$BRANCH  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "    SKIP_ADE20K=$SKIP_ADE20K  DIST_STEPS=$DIST_STEPS  DIV_WEIGHTS='$DIV_WEIGHTS'"

# ============================= 1. CLONE =============================
clone_or_update "$LOOKWHERE_URL" "$LW_REPO"  "$BRANCH"
clone_or_update "$OTL_URL"       "$OTL_REPO" "$BRANCH"

# OTL is imported by lookwhere via sys.path; imagenette path is derived from it.
export OTL_PATH="$OTL_REPO"

# ============================= 2. ENVIRONMENT (single uv venv) =============================
# One shared environment runs both the OTL pytest and all lookwhere scripts:
# tokenlearner.modules is pure-torch and imported via OTL_PATH on sys.path, and the
# combined requirements.txt was frozen to satisfy both repos.
log "creating uv venv ($VENV) with python $PY_VERSION"
[[ -d "$VENV" ]] || uv venv --python="$PY_VERSION" "$VENV"
log "installing dependencies from $(basename "$REQ")"
# --no-deps: requirements.txt is a complete freeze of both envs (every transitive
# dependency is already pinned), so install the exact set without re-resolving.
# Required because the freeze contains pins uv's resolver rejects but pip installed
# anyway (e.g. fsspec==2026.4.0 vs datasets 4.8.5's fsspec<=2026.2.0 cap).
uv pip install --no-deps --python "$VENV/bin/python" -r "$REQ"
PY="$VENV/bin/python"
ok "environment ready: $("$PY" -c 'import torch; print("torch", torch.__version__)')"

# ============================= 3. DATA =============================
log "fetching pretrained checkpoint"
fetch "$CKPT_URL" "$LW_REPO/lookwhere_dinov2.pt"

log "fetching imagenette2-320 (-> OTL/data)"
mkdir -p "$OTL_REPO/data"
if [[ -d "$OTL_REPO/data/imagenette2-320/val" ]]; then
  ok "imagenette already present, skip"
else
  fetch "$IMAGENETTE_URL" "$OTL_REPO/data/imagenette2-320.tgz"
  tar xzf "$OTL_REPO/data/imagenette2-320.tgz" -C "$OTL_REPO/data"
  ok "imagenette extracted"
fi

if [[ "$SKIP_ADE20K" != "1" ]]; then
  log "fetching ADE20K val (~923MB)"
  mkdir -p "$LW_REPO/ade_data"
  if [[ -d "$LW_REPO/ade_data/ADEChallengeData2016/images/validation" ]]; then
    ok "ADE20K already present, skip"
  else
    fetch "$ADE_URL" "$LW_REPO/ade_data/ade.zip"
    unzip -q "$LW_REPO/ade_data/ade.zip" -d "$LW_REPO/ade_data"
    ok "ADE20K extracted"
  fi
fi

# ============================= 4. PIPELINE =============================
# Helper to run a lookwhere script from inside the repo (cwd matters for the
# relative ice_cream.jpg / checkpoint / ade_data paths).
run_lw() { ( cd "$LW_REPO" && "$PY" "$@" ); }

log "[Item 1] OpenTokenLearner tests"
( cd "$OTL_REPO" && "$PY" -m pytest tests/ -q )

log "[Item 5] gradient-flow proof"
run_lw grad_sanity.py

log "[Item 3] untrained multi-foveal visualization"
run_lw experiment_softwhere.py

log "[Items 4 + A] distill v1.0 across diversity weights {$DIV_WEIGHTS}"
for d in $DIV_WEIGHTS; do
  run_lw distill_decompose.py --variant v10 --steps "$DIST_STEPS" --diversity "$d"
done

log "[Item B] distill v1.1 across diversity weights {$DIV_WEIGHTS}"
for d in $DIV_WEIGHTS; do
  run_lw distill_decompose.py --variant v11 --steps "$DIST_STEPS" --diversity "$d"
done

log "[Item 4] distilled multi-foveal figures (v1.0, v1.1)"
run_lw experiment_softwhere.py --distilled softwhere_head_v10_div1.pt
run_lw experiment_softwhere.py --distilled softwhere_head_v11_div1.pt

log "[Item 6] teacher-agreement proxy (v1.0, v1.1)"
run_lw coverage_proxy.py --distilled softwhere_head_v10_div1.pt --variant v10 --n_images "$PROXY_IMAGES"
run_lw coverage_proxy.py --distilled softwhere_head_v11_div1.pt --variant v11 --n_images "$PROXY_IMAGES"

log "[Strengthener A/B] map-diversity vs fidelity (v1.0, v1.1)"
run_lw map_diversity.py --variant v10 --n_images "$DIVERSITY_IMAGES"
run_lw map_diversity.py --variant v11 --n_images "$DIVERSITY_IMAGES"

if [[ "$SKIP_ADE20K" != "1" ]]; then
  log "[Strengthener C] ADE20K multi-object coverage (in-domain distilled head)"
  run_lw distill_decompose.py --variant v10 --steps 800 --diversity 1 --n_images 64 \
    --batch_size 8 --image_glob "ade_data/ADEChallengeData2016/images/training/*.jpg" --tag _ade
  run_lw ade20k_coverage.py --distilled softwhere_head_v10_div1_ade.pt --variant v10 --n_images "$ADE_IMAGES"
else
  ok "[Strengthener C] skipped (SKIP_ADE20K=1)"
fi


# ============================= SUMMARY =============================
log "DONE — artifacts in $LW_REPO"
cat <<EOF
    Figures : softwhere_v10_untrained.png, softwhere_v11_untrained.png,
              softwhere_v10_distilled.png, softwhere_v11_distilled.png
    Heads   : softwhere_head_v1{0,1}_div{0,0.5,1,2}.pt, softwhere_head_v10_div1_ade.pt

  Expected headline numbers (approximate; ±0.01-0.03):
    grad_sanity        -> PASS (6 head params, non-zero grads, no leakage)
    proxy v1.0         -> recall ~0.42  (random ~0.10, chance ~0.10)
    map-diversity v1.0 -> div=1: overlap ~0.02, fidelity ~0.094
    v1.0 vs v1.1       -> v1.0 KL ~0.094 > v1.1 KL ~0.144 (v1.0 wins fidelity)
    ADE20K coverage    -> LookWhere 0.70 > random 0.61 > SoftWhere-agg 0.47 > multifoveal 0.36
                          (the honest negative result; see proposal §5.5)
EOF
