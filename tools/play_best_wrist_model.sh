#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
PYTHON_BIN="${G1_MJLAB_PYTHON:-/mnt/hdd/miniforge3/envs/unitree_rl_mjlab/bin/python}"
CHECKPOINT="${G1_WRIST_CHECKPOINT:-$PROJECT_ROOT/logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_12-12-08/model_4999.pt}"
NUM_ENVS="${G1_PLAY_NUM_ENVS:-1}"
MPL_CACHE="${MPLCONFIGDIR:-/tmp/g1-wrist-matplotlib-${UID}}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: mjlab Python not found or not executable: $PYTHON_BIN"
  echo "Override it with G1_MJLAB_PYTHON=/path/to/python."
  exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  echo "Fetch it first with: ./tools/remote_fetch_best_wrist_model.sh"
  exit 1
fi

mkdir -p "$MPL_CACHE"
cd "$PROJECT_ROOT"

export MPLCONFIGDIR="$MPL_CACHE"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" scripts/play.py \
  Unitree-G1-Wrist-Recovery-Teacher \
  --checkpoint-file "$CHECKPOINT" \
  --num-envs "$NUM_ENVS" \
  "$@"
