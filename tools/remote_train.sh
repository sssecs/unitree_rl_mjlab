#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage:"
    echo "  $0 <session-name> <train-args...>"
    exit 1
fi

SESSION="$1"
shift

if [[ ! "$SESSION" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "Invalid session name: $SESSION"
    exit 1
fi

REMOTE_HOST="unitree-trainer"
REMOTE_REPO="/home/dev/unitree_rl_mjlab"
RUN_DIR="$REMOTE_REPO/.autotune/$SESSION"

if ssh "$REMOTE_HOST" "tmux has-session -t '$SESSION' 2>/dev/null"; then
    echo "ERROR: tmux session '$SESSION' already exists."
    exit 1
fi

printf -v TRAIN_ARGS '%q ' "$@"

REMOTE_CMD="
set -o pipefail
mkdir -p '$RUN_DIR'
rm -f '$RUN_DIR/DONE' '$RUN_DIR/exit_code'

cd '$REMOTE_REPO'

./run_train.sh $TRAIN_ARGS 2>&1 | tee '$RUN_DIR/train.log'

rc=\${PIPESTATUS[0]}

echo \$rc > '$RUN_DIR/exit_code'
date -Is > '$RUN_DIR/DONE'

exit \$rc
"

printf -v TMUX_CMD '%q' "$REMOTE_CMD"

ssh "$REMOTE_HOST" \
    "tmux new-session -d -s '$SESSION' bash -lc $TMUX_CMD"

mkdir -p .autotune
echo "$SESSION" > .autotune/current_trial

sleep 1

if ssh "$REMOTE_HOST" "tmux has-session -t '$SESSION' 2>/dev/null"; then
    echo "Started: $SESSION"
else
    echo "Training exited very quickly; check:"
    echo "  ssh $REMOTE_HOST 'cat $RUN_DIR/train.log'"
fi