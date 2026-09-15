#!/usr/bin/env bash
set -uo pipefail

REMOTE_HOST="unitree-trainer"
REMOTE_REPO="/home/dev/unitree_rl_mjlab"
POLL_INTERVAL_SECONDS="${AUTOTUNE_POLL_INTERVAL_SECONDS:-60}"
RETRY_INTERVAL_SECONDS="${AUTOTUNE_RETRY_INTERVAL_SECONDS:-900}"
MAX_REPORT_ATTEMPTS="${AUTOTUNE_MAX_REPORT_ATTEMPTS:-3}"

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
STATE_DIR="$PROJECT_ROOT/.autotune"
HANDLED_DIR="$STATE_DIR/handled"
FAILED_DIR="$STATE_DIR/failed"
ATTEMPTS_DIR="$STATE_DIR/attempts"
RESULTS_DIR="$STATE_DIR/metric_reports"
PENDING_DIR="$STATE_DIR/pending"
STOP_FILE="$STATE_DIR/STOP_REQUESTED"
GROUP_DIR="$STATE_DIR/groups"

mkdir -p \
    "$HANDLED_DIR" "$FAILED_DIR" "$ATTEMPTS_DIR" "$RESULTS_DIR" \
    "$PENDING_DIR" "$GROUP_DIR"

notify_email() {
    local subject="$1"
    local body_file="$2"
    local attachment="${3:-}"
    local args=("$subject" "$body_file")
    if [ -n "$attachment" ] && [ -f "$attachment" ]; then
        args+=(--attachment "$attachment")
    fi
    python3 tools/autotune_notify.py "${args[@]}" \
        || echo "[autotune] email notification failed; see output above"
}

render_plot() {
    local trial="$1"
    shift
    local local_plot="$RESULTS_DIR/${trial}.png"
    local remote_plot="$REMOTE_REPO/.autotune/${trial}/training_metrics.png"
    if ssh -o ConnectTimeout=15 "$REMOTE_HOST" \
        "/home/dev/miniconda3/envs/unitree_rl_mjlab/bin/python $REMOTE_REPO/tools/plot_training_metrics.py $* --output '$remote_plot'" >/dev/null \
        && scp -q "$REMOTE_HOST:$remote_plot" "$local_plot"; then
        printf '%s\n' "$local_plot"
    else
        echo "[autotune] plot rendering failed; sending completion email without attachment" >&2
        return 1
    fi
}

# Only one metric watcher may process completions for this worktree.
exec 9>"$STATE_DIR/watcher.lock"
if ! flock -n 9; then
    echo "[autotune] another watcher already owns $STATE_DIR/watcher.lock"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "[autotune] metric watcher started; no Codex/model calls"
echo "[autotune] poll interval: ${POLL_INTERVAL_SECONDS}s"

while true; do
    TRIAL=""
    IS_GROUP=0
    GROUP_FILE=""
    for CANDIDATE_GROUP_FILE in "$GROUP_DIR"/*.json; do
        [ -e "$CANDIDATE_GROUP_FILE" ] || break
        CANDIDATE_GROUP="$(basename "$CANDIDATE_GROUP_FILE" .json)"
        CANDIDATE="group.$CANDIDATE_GROUP"
        if [ ! -f "$HANDLED_DIR/$CANDIDATE" ] \
            && [ ! -f "$FAILED_DIR/$CANDIDATE" ]; then
            TRIAL="$CANDIDATE"
            IS_GROUP=1
            GROUP_FILE="$CANDIDATE_GROUP_FILE"
            break
        fi
    done

    for PENDING_FILE in "$PENDING_DIR"/*; do
        [ -z "$TRIAL" ] || break
        [ -e "$PENDING_FILE" ] || break
        CANDIDATE="$(basename "$PENDING_FILE")"
        if [ ! -f "$HANDLED_DIR/$CANDIDATE" ] \
            && [ ! -f "$FAILED_DIR/$CANDIDATE" ]; then
            TRIAL="$CANDIDATE"
            break
        fi
    done

    # Backward compatibility for trials launched before the pending queue.
    if [ -z "$TRIAL" ] && [ -s "$STATE_DIR/current_trial" ]; then
        CANDIDATE="$(cat "$STATE_DIR/current_trial")"
        if [ ! -f "$HANDLED_DIR/$CANDIDATE" ] \
            && [ ! -f "$FAILED_DIR/$CANDIDATE" ]; then
            TRIAL="$CANDIDATE"
        fi
    fi

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

    if [ "$IS_GROUP" -eq 1 ]; then
        mapfile -t GROUP_SESSIONS < <(jq -r '.sessions[]' "$GROUP_FILE")
        GROUP_COMPLETE=1
        for GROUP_SESSION in "${GROUP_SESSIONS[@]}"; do
            if [[ ! "$GROUP_SESSION" =~ ^[A-Za-z0-9_.-]+$ ]]; then
                echo "[autotune] unsafe session in group: $GROUP_SESSION"
                GROUP_COMPLETE=0
                break
            fi
            if ! ssh -o ConnectTimeout=10 "$REMOTE_HOST" \
                "test -f '$REMOTE_REPO/.autotune/$GROUP_SESSION/DONE'" \
                2>/dev/null; then
                GROUP_COMPLETE=0
                break
            fi
        done
        if [ "$GROUP_COMPLETE" -ne 1 ]; then
            sleep "$POLL_INTERVAL_SECONDS"
            continue
        fi
    else
        if ! ssh -o ConnectTimeout=10 "$REMOTE_HOST" \
            "test -f '$REMOTE_REPO/.autotune/$TRIAL/DONE'" \
            2>/dev/null; then
            # A network interruption and an unfinished run are both non-terminal.
            sleep "$POLL_INTERVAL_SECONDS"
            continue
        fi
    fi

    echo
    echo "[autotune] completed trial: $TRIAL"

    ATTEMPTS_FILE="$ATTEMPTS_DIR/$TRIAL"
    ATTEMPTS=0
    if [ -f "$ATTEMPTS_FILE" ]; then
        ATTEMPTS="$(cat "$ATTEMPTS_FILE")"
    fi
    ATTEMPTS=$((ATTEMPTS + 1))
    printf '%s\n' "$ATTEMPTS" > "$ATTEMPTS_FILE"

    RESULT_FILE="$RESULTS_DIR/${TRIAL}.json"
    REPORT_ARGS=(--output "$RESULT_FILE" --window 500)
    if [ "$IS_GROUP" -eq 1 ]; then
        REPORT_ARGS+=(--group-file "$GROUP_FILE")
    else
        REPORT_ARGS+=(--session "$TRIAL")
    fi
    echo "[autotune] extracting numeric metrics attempt $ATTEMPTS/$MAX_REPORT_ATTEMPTS..."

    if python3 tools/autotune_metric_report.py "${REPORT_ARGS[@]}"
    then
        EMAIL_BODY="$RESULTS_DIR/${TRIAL}.email.txt"
        printf 'Training completed: %s\n\nAttached: TensorBoard training curves.\nNumeric diagnostics are saved locally for later inspection.\n' "$TRIAL" > "$EMAIL_BODY"
        PLOT_FILE=""
        if [ "$IS_GROUP" -eq 1 ]; then
            PLOT_FILE="$(render_plot "$TRIAL" "${GROUP_SESSIONS[@]}")" || true
        else
            PLOT_FILE="$(render_plot "$TRIAL" "$TRIAL")" || true
        fi
        touch "$HANDLED_DIR/$TRIAL"
        if [ "$IS_GROUP" -ne 1 ]; then
            rm -f "$PENDING_DIR/$TRIAL"
        fi
        rm -f "$ATTEMPTS_FILE"
        echo "[autotune] metric report complete: $RESULT_FILE"
        if [ -s "$STOP_FILE" ]; then
            echo "[autotune] stop requested:"
            sed 's/^/[autotune]   /' "$STOP_FILE"
            notify_email "G1 training complete: $TRIAL (stopped)" "$EMAIL_BODY" "$PLOT_FILE"
            echo "[autotune] watcher exiting cleanly"
            exit 0
        fi
        notify_email "G1 training complete: $TRIAL" "$EMAIL_BODY" "$PLOT_FILE"
        sleep "$POLL_INTERVAL_SECONDS"
        continue
    fi

    echo "[autotune] metric extraction failed for trial $TRIAL (attempt $ATTEMPTS)."
    if [ "$ATTEMPTS" -ge "$MAX_REPORT_ATTEMPTS" ]; then
        touch "$FAILED_DIR/$TRIAL"
        cat > "$STOP_FILE" <<EOF
Stopped after $TRIAL: numeric metric extraction failed $ATTEMPTS times.
Remove $FAILED_DIR/$TRIAL and restart the watcher after fixing SSH or TensorBoard access.
EOF
        notify_email "G1 metric watcher stopped: report failure" "$STOP_FILE"
        echo "[autotune] retry limit reached; watcher exiting"
        exit 1
    fi

    sleep "$RETRY_INTERVAL_SECONDS"
done
