#!/usr/bin/env bash
#
# Shared helpers for the SoftWhere experiment wrapper scripts.
#
# Usage:
#   source ./experiment_env.sh
#   softwhere_init "exp_name"
#   run_lw some_script.py ...
#
# Bootstrap usage:
#   ./experiment_env.sh prepare
#   SOFTWHERE_AUTO_PREPARE=1 ./run_exp1_resolution_parity.sh

set -euo pipefail

_SOFTWHERE_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOFTWHERE_LOGGING_STARTED="${SOFTWHERE_LOGGING_STARTED:-0}"

log() {
  printf '\n==> %s\n' "$*"
}

ok() {
  printf '    %s\n' "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

need() {
  command -v "$1" >/dev/null 2>&1 || die "required tool not found: $1"
}

softwhere_config() {
  # Repo URLs. Override if you want to use a local mirror or a different fork.
  LOOKWHERE_URL="${LOOKWHERE_URL:-git@github.com:engichang1467/lookwhere.git}"
  OTL_URL="${OTL_URL:-git@github.com:engichang1467/Open-TokenLearner.git}"
  BRANCH="${BRANCH:-project-e-2}"

  # Directory layout. SOFTWHERE_BASE is kept for compatibility with reproduce.sh.
  SOFTWHERE_DIR="${SOFTWHERE_DIR:-${SOFTWHERE_BASE:-$_SOFTWHERE_ENV_DIR}}"
  LW_DIR="${LW_DIR:-$SOFTWHERE_DIR/lookwhere}"
  OTL_PATH="${OTL_PATH:-$SOFTWHERE_DIR/Open-TokenLearner}"
  VENV="${VENV:-$SOFTWHERE_DIR/.venv}"
  REQ="${REQ:-$SOFTWHERE_DIR/requirements.txt}"
  PY_VERSION="${PY_VERSION:-3.12}"
  PY="${PY:-$VENV/bin/python}"

  # Runtime knobs.
  LOG_DIR="${LOG_DIR:-$SOFTWHERE_DIR/personal/logs}"
  SKIP_ADE20K="${SKIP_ADE20K:-0}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export OTL_PATH

  # Data URLs.
  CKPT_URL="${CKPT_URL:-https://huggingface.co/antofuller/lookwhere/resolve/main/lookwhere_dinov2.pt}"
  IMAGENETTE_URL="${IMAGENETTE_URL:-https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz}"
  ADE_URL="${ADE_URL:-https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip}"
}

start_log() {
  local name="${1:?log name required}"

  mkdir -p "$LOG_DIR"
  if [[ "$SOFTWHERE_LOGGING_STARTED" != "1" ]]; then
    LOG_FILE="${LOG_FILE:-$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log}"
    exec > >(tee "$LOG_FILE") 2>&1
    SOFTWHERE_LOGGING_STARTED=1
  fi
}

fetch() {
  local url="${1:?url required}"
  local dest="${2:?destination required}"

  if [[ -s "$dest" ]]; then
    ok "exists, skip download: $(basename "$dest")"
    return
  fi

  mkdir -p "$(dirname "$dest")"
  log "downloading $(basename "$dest")"
  curl -fL --retry 3 -C - -o "$dest" "$url" || die "download failed: $url"
}

clone_or_update() {
  local url="${1:?repo url required}"
  local dir="${2:?repo dir required}"
  local branch="${3:?branch required}"

  if [[ -d "$dir/.git" ]]; then
    log "checking $(basename "$dir")"
    git -C "$dir" fetch origin "$branch"

    local current_branch
    current_branch="$(git -C "$dir" branch --show-current || true)"
    if [[ "$current_branch" == "$branch" ]]; then
      ok "$(basename "$dir") already on $branch; local changes are preserved"
      return
    fi

    if [[ -n "$(git -C "$dir" status --porcelain)" ]]; then
      ok "$(basename "$dir") has local changes; leaving it on ${current_branch:-detached HEAD}"
      ok "stash/commit changes first if you want to switch to $branch"
      return
    fi

    git -C "$dir" checkout "$branch"
    ok "$(basename "$dir") checked out $branch"
  else
    log "cloning $(basename "$dir") @ $branch"
    git clone --branch "$branch" "$url" "$dir" \
      || die "clone failed for $url (branch $branch). Check the URL/branch."
  fi
}

softwhere_install_uv_env() {
  need uv
  [[ -f "$REQ" ]] || die "requirements file not found: $REQ"

  log "creating uv venv ($VENV) with python $PY_VERSION"
  [[ -d "$VENV" ]] || uv venv --python="$PY_VERSION" "$VENV"

  log "installing dependencies from $(basename "$REQ")"
  uv pip install --no-deps --python "$VENV/bin/python" -r "$REQ"

  PY="$VENV/bin/python"
  export PY
  ok "environment ready: $("$PY" -c 'import torch; print("torch", torch.__version__)')"
}

softwhere_prepare_data() {
  need curl
  need unzip
  need tar

  [[ -d "$LW_DIR" ]] || die "lookwhere repo not found: $LW_DIR"
  [[ -d "$OTL_PATH" ]] || die "Open-TokenLearner repo not found: $OTL_PATH"

  log "fetching pretrained checkpoint"
  fetch "$CKPT_URL" "$LW_DIR/lookwhere_dinov2.pt"

  log "fetching imagenette2-320"
  mkdir -p "$OTL_PATH/data"
  if [[ -d "$OTL_PATH/data/imagenette2-320/val" ]]; then
    ok "imagenette already present, skip"
  else
    fetch "$IMAGENETTE_URL" "$OTL_PATH/data/imagenette2-320.tgz"
    tar xzf "$OTL_PATH/data/imagenette2-320.tgz" -C "$OTL_PATH/data"
    ok "imagenette extracted"
  fi

  if [[ "$SKIP_ADE20K" == "1" ]]; then
    ok "ADE20K skipped (SKIP_ADE20K=1)"
    return
  fi

  log "fetching ADE20K val (~923MB)"
  mkdir -p "$LW_DIR/ade_data"
  if [[ -d "$LW_DIR/ade_data/ADEChallengeData2016/images/validation" ]]; then
    ok "ADE20K already present, skip"
  else
    fetch "$ADE_URL" "$LW_DIR/ade_data/ade.zip"
    unzip -q "$LW_DIR/ade_data/ade.zip" -d "$LW_DIR/ade_data"
    ok "ADE20K extracted"
  fi
}

softwhere_prepare() {
  softwhere_config

  need git
  need curl
  need unzip
  need tar
  need uv
  [[ -f "$REQ" ]] || die "requirements file not found: $REQ"

  mkdir -p "$SOFTWHERE_DIR"

  log "prepare SoftWhere workspace"
  echo "SOFTWHERE_DIR=$SOFTWHERE_DIR"
  echo "LW_DIR=$LW_DIR"
  echo "OTL_PATH=$OTL_PATH"
  echo "VENV=$VENV"
  echo "REQ=$REQ"
  echo "BRANCH=$BRANCH"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "SKIP_ADE20K=$SKIP_ADE20K"

  clone_or_update "$LOOKWHERE_URL" "$LW_DIR" "$BRANCH"
  clone_or_update "$OTL_URL" "$OTL_PATH" "$BRANCH"
  softwhere_install_uv_env
  softwhere_prepare_data
}

softwhere_init() {
  local experiment_name="${1:?experiment name required}"

  softwhere_config
  start_log "$experiment_name"

  log "$experiment_name"
  echo "SOFTWHERE_DIR=$SOFTWHERE_DIR"
  echo "LW_DIR=$LW_DIR"
  echo "PY=$PY"
  echo "OTL_PATH=$OTL_PATH"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "LOG_FILE=$LOG_FILE"

  if [[ "${SOFTWHERE_AUTO_PREPARE:-0}" == "1" ]]; then
    softwhere_prepare
  fi

  [[ -d "$LW_DIR" ]] || die "lookwhere repo not found: $LW_DIR. Run './experiment_env.sh prepare' or 'make prepare'."
  [[ -x "$PY" ]] || die "python is not executable: $PY. Run './experiment_env.sh prepare', 'make prepare', or set PY=/path/to/python."
}

run_lw() {
  log "lookwhere/$*"
  (cd "$LW_DIR" && "$PY" "$@")
}

require_lw_file() {
  local rel_path="${1:?relative path required}"
  local hint="${2:-Create it with an earlier experiment first.}"

  if [[ ! -e "$LW_DIR/$rel_path" ]]; then
    die "missing lookwhere/$rel_path. $hint"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  softwhere_config
  start_log "prepare"

  case "${1:-prepare}" in
    prepare|prep|bootstrap)
      softwhere_prepare
      ;;
    *)
      die "usage: $0 [prepare]"
      ;;
  esac
fi
