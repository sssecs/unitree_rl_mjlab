#!/usr/bin/env bash
set -uo pipefail

REMOTE_HOST="unitree-trainer"
REMOTE_REPO="/home/dev/unitree_rl_mjlab"
POLL_INTERVAL_SECONDS="${AUTOTUNE_POLL_INTERVAL_SECONDS:-60}"
RETRY_INTERVAL_SECONDS="${AUTOTUNE_RETRY_INTERVAL_SECONDS:-900}"
MAX_CODEX_ATTEMPTS="${AUTOTUNE_MAX_CODEX_ATTEMPTS:-3}"
MAX_EVAL_ATTEMPTS="${AUTOTUNE_MAX_EVAL_ATTEMPTS:-10}"
CODEX_MODEL="${AUTOTUNE_CODEX_MODEL:-gpt-5.6-terra}"
CODEX_REASONING_EFFORT="${AUTOTUNE_REASONING_EFFORT:-medium}"

if [[ ! "$CODEX_REASONING_EFFORT" =~ ^(low|medium|high|xhigh|max|ultra)$ ]]; then
    echo "[autotune] invalid reasoning effort: $CODEX_REASONING_EFFORT"
    exit 1
fi

PROJECT_ROOT="$(git rev-parse --show-toplevel)"
STATE_DIR="$PROJECT_ROOT/.autotune"
HANDLED_DIR="$STATE_DIR/handled"
FAILED_DIR="$STATE_DIR/failed"
ATTEMPTS_DIR="$STATE_DIR/attempts"
RESULTS_DIR="$STATE_DIR/codex_results"
PENDING_DIR="$STATE_DIR/pending"
STOP_FILE="$STATE_DIR/STOP_REQUESTED"
GROUP_DIR="$STATE_DIR/groups"
EVAL_ATTEMPTS_DIR="$STATE_DIR/eval_attempts"

mkdir -p \
    "$HANDLED_DIR" "$FAILED_DIR" "$ATTEMPTS_DIR" "$RESULTS_DIR" \
    "$PENDING_DIR" "$GROUP_DIR" "$EVAL_ATTEMPTS_DIR"

notify_email() {
    local subject="$1"
    local body_file="$2"
    python3 tools/autotune_notify.py "$subject" "$body_file" \
        || echo "[autotune] email notification failed; see output above"
}

# Only one watcher may trigger Codex for this worktree.
exec 9>"$STATE_DIR/watcher.lock"
if ! flock -n 9; then
    echo "[autotune] another watcher already owns $STATE_DIR/watcher.lock"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "[autotune] watcher started; shell polling uses no model tokens"
echo "[autotune] poll interval: ${POLL_INTERVAL_SECONDS}s"
echo "[autotune] Codex model: $CODEX_MODEL ($CODEX_REASONING_EFFORT reasoning)"

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
        GROUP_DETAILS=""
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
            GROUP_EXIT_CODE="$(
                ssh "$REMOTE_HOST" \
                "cat '$REMOTE_REPO/.autotune/$GROUP_SESSION/exit_code'" \
                2>/dev/null || echo unknown
            )"
            GROUP_DETAILS+="Session: $GROUP_SESSION; exit code: $GROUP_EXIT_CODE; "
            GROUP_DETAILS+="run directory: $REMOTE_REPO/.autotune/$GROUP_SESSION"$'\n'
        done
        if [ "$GROUP_COMPLETE" -ne 1 ]; then
            sleep "$POLL_INTERVAL_SECONDS"
            continue
        fi
        PLAN_FILE="$(jq -r '.plan_file' "$GROUP_FILE")"
        if jq -e '.evaluation.automatic == true' "$PLAN_FILE" >/dev/null \
            && ! jq -e '.evaluation_complete == true' "$GROUP_FILE" >/dev/null;
        then
            EVAL_ATTEMPTS_FILE="$EVAL_ATTEMPTS_DIR/$CANDIDATE_GROUP"
            EVAL_ATTEMPTS=0
            if [ -f "$EVAL_ATTEMPTS_FILE" ]; then
                EVAL_ATTEMPTS="$(cat "$EVAL_ATTEMPTS_FILE")"
            fi
            EVAL_ATTEMPTS=$((EVAL_ATTEMPTS + 1))
            printf '%s\n' "$EVAL_ATTEMPTS" > "$EVAL_ATTEMPTS_FILE"
            echo "[autotune] materializing held-out matrix attempt $EVAL_ATTEMPTS/$MAX_EVAL_ATTEMPTS..."
            if python3 tools/autotune_evaluate_group.py "$GROUP_FILE"; then
                rm -f "$EVAL_ATTEMPTS_FILE"
                GROUP_DETAILS+="Fixed nominal/robust evaluation matrix is complete; paths are in the group manifest."$'\n'
            elif [ "$EVAL_ATTEMPTS" -ge "$MAX_EVAL_ATTEMPTS" ]; then
                cat > "$STOP_FILE" <<EOF
Stopped after $TRIAL: fixed held-out evaluation failed $EVAL_ATTEMPTS times.
Remote training artifacts are preserved. Repair evaluation before promotion.
EOF
                notify_email "G1 autotune stopped: evaluation failure" "$STOP_FILE"
                echo "[autotune] evaluator retry limit reached; watcher exiting"
                exit 1
            else
                echo "[autotune] group evaluation failed; will retry"
                sleep "$RETRY_INTERVAL_SECONDS"
                continue
            fi
        fi
        EXIT_CODE="group"
    else
        if ! ssh -o ConnectTimeout=10 "$REMOTE_HOST" \
            "test -f '$REMOTE_REPO/.autotune/$TRIAL/DONE'" \
            2>/dev/null; then
            # A network interruption and an unfinished run are both non-terminal.
            sleep "$POLL_INTERVAL_SECONDS"
            continue
        fi
        EXIT_CODE="$(
            ssh "$REMOTE_HOST" \
            "cat '$REMOTE_REPO/.autotune/$TRIAL/exit_code'" \
            2>/dev/null || echo unknown
        )"
    fi

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

        if [ "$IS_GROUP" -eq 1 ]; then
            printf '\n\nCompleted experiment group: %s\n' "$TRIAL"
            printf 'Analyze all siblings together; do not decide from only one run.\n'
            printf 'Group manifest: '
            jq -c . "$GROUP_FILE"
            printf '%s' "$GROUP_DETAILS"
        else
            printf '\n\nCompleted trial: %s\n' "$TRIAL"
            printf 'Remote exit code: %s\n' "$EXIT_CODE"
            printf '\nRemote run directory:\n'
            printf '%s/.autotune/%s\n' "$REMOTE_REPO" "$TRIAL"
        fi
    )"

    RESULT_FILE="$RESULTS_DIR/${TRIAL}.md"
    echo "[autotune] launching Codex attempt $ATTEMPTS/$MAX_CODEX_ATTEMPTS..."

    if codex exec \
        --approve-for-me \
        --ephemeral \
        --model "$CODEX_MODEL" \
        -c "model_reasoning_effort=\"$CODEX_REASONING_EFFORT\"" \
        --cd "$PROJECT_ROOT" \
        --output-last-message "$RESULT_FILE" \
        "$PROMPT"
    then
        # Mark only after Codex exits successfully. This avoids losing a trial if
        # authentication, networking, or the agent process fails during analysis.
        touch "$HANDLED_DIR/$TRIAL"
        if [ "$IS_GROUP" -ne 1 ]; then
            rm -f "$PENDING_DIR/$TRIAL"
        fi
        rm -f "$ATTEMPTS_FILE"
        echo "[autotune] Codex completed; final message: $RESULT_FILE"
        if [ -s "$STOP_FILE" ]; then
            echo "[autotune] stop requested by analysis:"
            sed 's/^/[autotune]   /' "$STOP_FILE"
            notify_email "G1 autotune summary: $TRIAL (stopped)" "$RESULT_FILE"
            echo "[autotune] watcher exiting cleanly"
            exit 0
        fi
        notify_email "G1 autotune summary: $TRIAL" "$RESULT_FILE"
        sleep "$POLL_INTERVAL_SECONDS"
        continue
    fi

    echo "[autotune] Codex failed for trial $TRIAL (attempt $ATTEMPTS)."
    if [ "$ATTEMPTS" -ge "$MAX_CODEX_ATTEMPTS" ]; then
        touch "$FAILED_DIR/$TRIAL"
        cat > "$STOP_FILE" <<EOF
Stopped after $TRIAL: Codex analysis failed $ATTEMPTS times.
Remove $FAILED_DIR/$TRIAL and restart the watcher after fixing authentication or connectivity.
EOF
        notify_email "G1 autotune stopped: Codex failure" "$STOP_FILE"
        echo "[autotune] retry limit reached; watcher exiting"
        exit 1
    fi

    sleep "$RETRY_INTERVAL_SECONDS"
done
