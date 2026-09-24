#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage:"
    echo "  $0 <session-name> <train-args...>"
    echo "Note: --env.scene.num-envs is per process/GPU, not a global total."
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
LOCAL_NOTIFY_CONFIG="$(git rev-parse --show-toplevel)/.autotune/notify.env"
REMOTE_NOTIFY_CONFIG="$REMOTE_REPO/.autotune/notify.env"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"

if ssh "$REMOTE_HOST" "tmux has-session -t '$SESSION' 2>/dev/null"; then
    echo "ERROR: tmux session '$SESSION' already exists."
    exit 1
fi

if ssh "$REMOTE_HOST" "test -e '$RUN_DIR'"; then
    echo "ERROR: run directory already exists; choose a unique session name:"
    echo "  $RUN_DIR"
    exit 1
fi

printf -v TRAIN_ARGS '%q ' "$@"

METADATA_DIR="$(mktemp -d)"
cleanup() {
    rm -rf -- "$METADATA_DIR"
}
trap cleanup EXIT

git -C "$PROJECT_ROOT" rev-parse HEAD > "$METADATA_DIR/git_commit"
git -C "$PROJECT_ROOT" status --short > "$METADATA_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff --binary HEAD > "$METADATA_DIR/git_diff.patch"
git -C "$PROJECT_ROOT" ls-files --cached --others --exclude-standard -z \
    > "$METADATA_DIR/source_files.zlist"
git -C "$PROJECT_ROOT" ls-files --others --exclude-standard -z \
    > "$METADATA_DIR/untracked_files.zlist"
tar -C "$PROJECT_ROOT" --no-recursion --null \
    --files-from="$METADATA_DIR/untracked_files.zlist" \
    -czf "$METADATA_DIR/untracked_files.tar.gz"
while IFS= read -r -d '' SOURCE_FILE; do
    (cd "$PROJECT_ROOT" && sha256sum "$SOURCE_FILE")
done < "$METADATA_DIR/source_files.zlist" \
    > "$METADATA_DIR/source_files.sha256"
printf '%s\n' "$TRAIN_ARGS" > "$METADATA_DIR/train_args"

# Save provenance before tmux starts, including runs that fail before Python.
ssh "$REMOTE_HOST" "mkdir -p '$RUN_DIR'"
scp "$METADATA_DIR/git_commit" \
    "$METADATA_DIR/git_status.txt" \
    "$METADATA_DIR/git_diff.patch" \
    "$METADATA_DIR/untracked_files.tar.gz" \
    "$METADATA_DIR/source_files.sha256" \
    "$METADATA_DIR/train_args" \
    "$REMOTE_HOST:$RUN_DIR/"

# The training process runs remotely and must be able to notify without a
# workstation-side watcher. Keep SMTP credentials out of Git and provenance.
NOTIFY_ENABLED=0
if [[ -f "$LOCAL_NOTIFY_CONFIG" ]]; then
    NOTIFY_ENABLED=1
    REMOTE_NOTIFY_TMP="$REMOTE_NOTIFY_CONFIG.$SESSION.tmp"
    ssh "$REMOTE_HOST" \
        "umask 077; cat > '$REMOTE_NOTIFY_TMP' && chmod 600 '$REMOTE_NOTIFY_TMP' && mv '$REMOTE_NOTIFY_TMP' '$REMOTE_NOTIFY_CONFIG'" \
        < "$LOCAL_NOTIFY_CONFIG"
fi

REMOTE_CMD="
set -o pipefail
date -Is > '$RUN_DIR/started_at'
cd '$REMOTE_REPO'

AUTOTUNE_RUN_DIR='$RUN_DIR' bash ./run_train.sh $TRAIN_ARGS 2>&1 | tee '$RUN_DIR/train.log'
rc=\${PIPESTATUS[0]}

echo \$rc > '$RUN_DIR/exit_code'
date -Is > '$RUN_DIR/ended_at'
if test '$NOTIFY_ENABLED' = 1 && test -f '$REMOTE_NOTIFY_CONFIG'; then
  if python3 tools/remote_train_notify.py \
    --session '$SESSION' --exit-code \"\$rc\" \
    --run-dir '$RUN_DIR' --config '$REMOTE_NOTIFY_CONFIG' \
    > '$RUN_DIR/notify.log' 2>&1; then
    echo sent > '$RUN_DIR/notify_status'
  else
    echo failed > '$RUN_DIR/notify_status'
  fi
else
  echo disabled > '$RUN_DIR/notify_status'
fi
date -Is > '$RUN_DIR/DONE'
exit \$rc
"

printf -v TMUX_CMD '%q' "$REMOTE_CMD"
ssh "$REMOTE_HOST" \
    "tmux new-session -d -s '$SESSION' bash -lc $TMUX_CMD"

if [ "${AUTOTUNE_NO_REGISTER:-0}" != "1" ]; then
    mkdir -p .autotune/pending
    : > ".autotune/pending/$SESSION"
    CURRENT_TRIAL_TMP=".autotune/current_trial.$$"
    printf '%s\n' "$SESSION" > "$CURRENT_TRIAL_TMP"
    mv "$CURRENT_TRIAL_TMP" .autotune/current_trial
fi

sleep 1

if ssh "$REMOTE_HOST" "tmux has-session -t '$SESSION' 2>/dev/null"; then
    echo "Started: $SESSION"
else
    echo "Training exited very quickly; check:"
    echo "  ssh $REMOTE_HOST 'cat $RUN_DIR/train.log'"
fi
