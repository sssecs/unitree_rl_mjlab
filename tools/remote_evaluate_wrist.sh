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
REMOTE_JOB="$REMOTE_REPO/.autotune/evaluation_jobs/$EVAL_NAME"
REMOTE_SESSION="eval_$EVAL_NAME"
LOCAL_OUTPUT="results/remote_eval/$EVAL_NAME"
POLL_SECONDS="${EVAL_POLL_SECONDS:-15}"
MAX_WAIT_SECONDS="${EVAL_MAX_WAIT_SECONDS:-3600}"
EVALUATOR_SCRIPT="${G1_EVALUATOR_SCRIPT:-scripts/evaluate_wrist_recovery.py}"
case "$EVALUATOR_SCRIPT" in
  scripts/evaluate_wrist_recovery.py|scripts/diagnose_wrist_clutch.py) ;;
  *) echo "Unsupported evaluator script: $EVALUATOR_SCRIPT"; exit 1 ;;
esac

has_local_result() {
  [ -d "$LOCAL_OUTPUT" ] \
    && [ "$(find "$LOCAL_OUTPUT" -type f -name summary.json | wc -l)" -eq 1 ] \
    && [ "$(find "$LOCAL_OUTPUT" -type f -name episodes.csv | wc -l)" -eq 1 ]
}

copy_remote_result() {
  mkdir -p "$LOCAL_OUTPUT"
  # Merge into a partial local copy so an interrupted transfer is recoverable.
  ssh "$REMOTE_HOST" "cd '$REMOTE_OUTPUT' && tar -cf - ." \
    | tar -C "$LOCAL_OUTPUT" -xf -
  if ! has_local_result; then
    echo "Expected exactly one summary.json and episodes.csv under $LOCAL_OUTPUT"
    return 1
  fi
  echo "Copied evaluation results to: $LOCAL_OUTPUT"
  find "$LOCAL_OUTPUT" -type f \( -name summary.json -o -name manifest.json \) -print
}

if has_local_result; then
  echo "Reusing completed local evaluation: $LOCAL_OUTPUT"
  exit 0
fi

if ssh "$REMOTE_HOST" \
  "test -d '$REMOTE_OUTPUT' && find '$REMOTE_OUTPUT' -type f -name summary.json -print -quit | grep -q .";
then
  copy_remote_result
  exit 0
fi

if ! ssh "$REMOTE_HOST" "test -f '$REMOTE_CHECKPOINT'"; then
  echo "Remote checkpoint not found: $REMOTE_CHECKPOINT"
  exit 1
fi

job_exists=0
job_done=0
session_exists=0
ssh "$REMOTE_HOST" "test -d '$REMOTE_JOB'" && job_exists=1
ssh "$REMOTE_HOST" "test -f '$REMOTE_JOB/DONE'" && job_done=1
ssh "$REMOTE_HOST" "tmux has-session -t '$REMOTE_SESSION' 2>/dev/null" \
  && session_exists=1

if [ "$job_exists" -eq 0 ] \
  || { [ "$job_done" -eq 0 ] && [ "$session_exists" -eq 0 ]; };
then
  EXTRA_ARGS=""
  if [ "$#" -gt 0 ]; then
    printf -v EXTRA_ARGS ' %q' "$@"
  fi
  REMOTE_CMD="
set -o pipefail
date -Is > '$REMOTE_JOB/started_at'
source /home/dev/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab
export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/cuda-compat:/home/dev/nvidia-550.127/lib:\$CONDA_PREFIX/lib\"
export LD_PRELOAD=/home/dev/nvidia-550.127/lib/libnvidia-ml.so.1
export NCCL_CUMEM_HOST_ENABLE=0
cd '$REMOTE_REPO'
python '$EVALUATOR_SCRIPT' \
  --checkpoint-file '$REMOTE_CHECKPOINT' \
  --output-dir '$REMOTE_OUTPUT'$EXTRA_ARGS \
  2>&1 | tee '$REMOTE_JOB/evaluate.log'
rc=\${PIPESTATUS[0]}
echo \$rc > '$REMOTE_JOB/exit_code'
date -Is > '$REMOTE_JOB/DONE'
exit \$rc
"
  printf -v TMUX_CMD '%q' "$REMOTE_CMD"
  ssh "$REMOTE_HOST" \
    "mkdir -p '$REMOTE_JOB' && tmux new-session -d -s '$REMOTE_SESSION' bash -lc $TMUX_CMD"
  echo "Started detached evaluation: $EVAL_NAME"
else
  echo "Reattaching to existing evaluation state: $EVAL_NAME"
fi

started_waiting=$SECONDS
while ! ssh -o ConnectTimeout=10 "$REMOTE_HOST" "test -f '$REMOTE_JOB/DONE'" \
  2>/dev/null;
do
  if (( SECONDS - started_waiting >= MAX_WAIT_SECONDS )); then
    echo "Timed out after ${MAX_WAIT_SECONDS}s: $EVAL_NAME"
    echo "The detached remote evaluation was not killed and may still complete."
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

exit_code="$(ssh "$REMOTE_HOST" "cat '$REMOTE_JOB/exit_code'")"
if [ "$exit_code" != "0" ]; then
  echo "Evaluation failed with exit code $exit_code: $EVAL_NAME"
  ssh "$REMOTE_HOST" "tail -n 200 '$REMOTE_JOB/evaluate.log'" || true
  exit "$exit_code"
fi

copy_remote_result
