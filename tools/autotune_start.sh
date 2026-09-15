#!/usr/bin/env bash
set -euo pipefail

WATCHER_SESSION="${AUTOTUNE_WATCHER_SESSION:-g1-autotune-watcher}"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
STATE_DIR="$PROJECT_ROOT/.autotune"
LOG_FILE="$STATE_DIR/watcher.log"

mkdir -p "$STATE_DIR"

if [ -s "$STATE_DIR/STOP_REQUESTED" ]; then
    mkdir -p "$STATE_DIR/stop_history"
    STOP_ARCHIVE="$STATE_DIR/stop_history/$(date +%Y%m%d_%H%M%S).txt"
    mv "$STATE_DIR/STOP_REQUESTED" "$STOP_ARCHIVE"
    echo "Archived previous stop decision: $STOP_ARCHIVE"
fi

for command_name in flock jq python3 ssh tmux; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $command_name"
        exit 1
    fi
done

if tmux has-session -t "$WATCHER_SESSION" 2>/dev/null; then
    echo "Watcher is already running in tmux session: $WATCHER_SESSION"
    echo "Log: $LOG_FILE"
    exit 0
fi

printf -v WATCH_COMMAND \
    'cd %q && exec %q >> %q 2>&1' \
    "$PROJECT_ROOT" "$PROJECT_ROOT/tools/autotune_watch.sh" "$LOG_FILE"

tmux new-session -d -s "$WATCHER_SESSION" bash -lc "$WATCH_COMMAND"

sleep 1
if tmux has-session -t "$WATCHER_SESSION" 2>/dev/null; then
    echo "Started watcher in tmux session: $WATCHER_SESSION"
    echo "Log: $LOG_FILE"
else
    echo "ERROR: watcher exited during startup. Inspect: $LOG_FILE"
    exit 1
fi
