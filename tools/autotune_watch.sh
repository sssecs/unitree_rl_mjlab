#!/usr/bin/env bash
set -u

REMOTE_HOST="unitree-trainer"
REMOTE_REPO="/home/dev/unitree_rl_mjlab"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
STATE_DIR="$PROJECT_ROOT/.autotune"
HANDLED_DIR="$STATE_DIR/handled"

mkdir -p "$HANDLED_DIR"

cd "$PROJECT_ROOT"

echo "[autotune] watcher started"

while true; do
    if [ ! -f "$STATE_DIR/current_trial" ]; then
        sleep 60
        continue
    fi

    TRIAL="$(cat "$STATE_DIR/current_trial")"

    if [ -z "$TRIAL" ]; then
        sleep 60
        continue
    fi

    if [ -f "$HANDLED_DIR/$TRIAL" ]; then
        sleep 60
        continue
    fi

    if ! ssh -o ConnectTimeout=10 "$REMOTE_HOST" \
        "test -f '$REMOTE_REPO/.autotune/$TRIAL/DONE'" \
        2>/dev/null
    then
        # 网络断开或者训练还没有结束，都继续等待
        sleep 60
        continue
    fi

    EXIT_CODE="$(
        ssh "$REMOTE_HOST" \
        "cat '$REMOTE_REPO/.autotune/$TRIAL/exit_code'" \
        2>/dev/null || echo unknown
    )"

    echo
    echo "[autotune] completed trial: $TRIAL"
    echo "[autotune] exit code: $EXIT_CODE"

    # 先标记，避免重复触发
    touch "$HANDLED_DIR/$TRIAL"

    PROMPT="$(
        cat tools/autotune_prompt.md

        printf '\n\nCompleted trial: %s\n' "$TRIAL"
        printf 'Remote exit code: %s\n' "$EXIT_CODE"

        printf '\nRemote run directory:\n'
        printf '%s/.autotune/%s\n' "$REMOTE_REPO" "$TRIAL"
    )"

    echo "[autotune] launching Codex..."

    codex exec "$PROMPT"

    echo "[autotune] Codex finished."

    sleep 60
done
