#!/usr/bin/env bash
set -uo pipefail

REMOTE_HOST="unitree-trainer"
REMOTE_REPO="/home/dev/unitree_rl_mjlab"
POLL_INTERVAL_SECONDS="${AUTOTUNE_POLL_INTERVAL_SECONDS:-60}"
RETRY_INTERVAL_SECONDS="${AUTOTUNE_RETRY_INTERVAL_SECONDS:-900}"
MAX_CODEX_ATTEMPTS="${AUTOTUNE_MAX_CODEX_ATTEMPTS:-3}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
STATE_DIR="$PROJECT_ROOT/.autotune"
HANDLED_DIR="$STATE_DIR/handled"
FAILED_DIR="$STATE_DIR/failed"
ATTEMPTS_DIR="$STATE_DIR/attempts"
RESULTS_DIR="$STATE_DIR/codex_results"

mkdir -p "$HANDLED_DIR" "$FAILED_DIR" "$ATTEMPTS_DIR" "$RESULTS_DIR"

# Only one watcher may trigger Codex for this worktree.
exec 9>"$STATE_DIR/watcher.lock"
if ! flock -n 9; then
    echo "[autotune] another watcher already owns $STATE_DIR/watcher.lock"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "[autotune] watcher started; shell polling uses no model tokens"
echo "[autotune] poll interval: ${POLL_INTERVAL_SECONDS}s"

while true; do
    if [ ! -f "$STATE_DIR/current_trial" ]; then
        sleep "$POLL_INTERVAL_SECONDS"
        continue
    fi

    TRIAL="$(cat "$STATE_DIR/current_trial")"

    if [ -z "$TRIAL" ]; then
        sleep "$POLL_INTERVAL_SECONDS"
        continue
    fi

    if [[ ! "$TRIAL" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "[autotune] invalid trial name in current_trial: $TRIAL"
        sleep "$POLL_INTERVAL_SECONDS"
        continue
    fi

    if [ -f "$HANDLED_DIR/$TRIAL" ] || [ -f "$FAILED_DIR/$TRIAL" ]; then
        sleep "$POLL_INTERVAL_SECONDS"
        continue
    fi

    if ! ssh -o ConnectTimeout=10 "$REMOTE_HOST" \
        "test -f '$REMOTE_REPO/.autotune/$TRIAL/DONE'" \
        2>/dev/null
    then
        # A network interruption and an unfinished run are both non-terminal.
        sleep "$POLL_INTERVAL_SECONDS"
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

    ATTEMPTS_FILE="$ATTEMPTS_DIR/$TRIAL"
    ATTEMPTS=0
    if [ -f "$ATTEMPTS_FILE" ]; then
        ATTEMPTS="$(cat "$ATTEMPTS_FILE")"
    fi
    ATTEMPTS=$((ATTEMPTS + 1))
    printf '%s\n' "$ATTEMPTS" > "$ATTEMPTS_FILE"

    PROMPT="$(
        cat tools/autotune_prompt.md

        printf '\n\nCompleted trial: %s\n' "$TRIAL"
        printf 'Remote exit code: %s\n' "$EXIT_CODE"

        printf '\nRemote run directory:\n'
        printf '%s/.autotune/%s\n' "$REMOTE_REPO" "$TRIAL"
    )"

    RESULT_FILE="$RESULTS_DIR/${TRIAL}.md"
    echo "[autotune] launching Codex attempt $ATTEMPTS/$MAX_CODEX_ATTEMPTS..."

    if codex exec \
        --approve-for-me \
        --ephemeral \
        -c 'model_reasoning_effort="medium"' \
        --cd "$PROJECT_ROOT" \
        --output-last-message "$RESULT_FILE" \
        "$PROMPT"
    then
        # Mark only after Codex exits successfully. This avoids losing a trial if
        # authentication, networking, or the agent process fails during analysis.
        touch "$HANDLED_DIR/$TRIAL"
        rm -f "$ATTEMPTS_FILE"
        echo "[autotune] Codex completed; final message: $RESULT_FILE"
        sleep "$POLL_INTERVAL_SECONDS"
        continue
    fi

    echo "[autotune] Codex failed for trial $TRIAL (attempt $ATTEMPTS)."
    if [ "$ATTEMPTS" -ge "$MAX_CODEX_ATTEMPTS" ]; then
        touch "$FAILED_DIR/$TRIAL"
        echo "[autotune] retry limit reached; manual review required."
        echo "[autotune] remove $FAILED_DIR/$TRIAL to retry after fixing the cause."
        sleep "$POLL_INTERVAL_SECONDS"
        continue
    fi

    sleep "$RETRY_INTERVAL_SECONDS"
done
