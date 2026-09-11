#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <evaluation-name> <remote-checkpoint> [evaluator-args...]"
  exit 1
fi

EVAL_NAME="$1"
REMOTE_CHECKPOINT="$2"
shift 2

if [[ ! "$EVAL_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid evaluation name: $EVAL_NAME"
  exit 1
fi

REMOTE_HOST="unitree-trainer"
REMOTE_REPO="/home/dev/unitree_rl_mjlab"
REMOTE_OUTPUT="$REMOTE_REPO/.autotune/evaluations/$EVAL_NAME"
LOCAL_OUTPUT="results/remote_eval/$EVAL_NAME"

if [ -e "$LOCAL_OUTPUT" ]; then
  echo "Local output already exists; choose a unique evaluation name:"
  echo "  $LOCAL_OUTPUT"
  exit 1
fi

if ssh "$REMOTE_HOST" "test -e '$REMOTE_OUTPUT'"; then
  echo "Remote output already exists; choose a unique evaluation name:"
  echo "  $REMOTE_OUTPUT"
  exit 1
fi

printf -v EXTRA_ARGS '%q ' "$@"
printf -v REMOTE_CMD '%q' "
set -euo pipefail
source /home/dev/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab
export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/cuda-compat:/home/dev/nvidia-550.127/lib:\$CONDA_PREFIX/lib\"
export LD_PRELOAD=/home/dev/nvidia-550.127/lib/libnvidia-ml.so.1
export NCCL_CUMEM_HOST_ENABLE=0
mkdir -p '$REMOTE_OUTPUT'
cd '$REMOTE_REPO'
python scripts/evaluate_wrist_recovery.py \
  --checkpoint-file '$REMOTE_CHECKPOINT' \
  --output-dir '$REMOTE_OUTPUT' \
  $EXTRA_ARGS
"

ssh "$REMOTE_HOST" "bash -lc $REMOTE_CMD"

mkdir -p "$(dirname "$LOCAL_OUTPUT")"
scp -r "$REMOTE_HOST:$REMOTE_OUTPUT" "$LOCAL_OUTPUT"
echo "Copied evaluation results to: $LOCAL_OUTPUT"
