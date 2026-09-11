#!/usr/bin/env bash
set -euo pipefail

WATCHER_SESSION="${AUTOTUNE_WATCHER_SESSION:-g1-autotune-watcher}"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
STATE_DIR="$PROJECT_ROOT/.autotune"

if tmux has-session -t "$WATCHER_SESSION" 2>/dev/null; then
    echo "Watcher: running ($WATCHER_SESSION)"
else
    echo "Watcher: stopped ($WATCHER_SESSION)"
fi

if [ -s "$STATE_DIR/current_trial" ]; then
    printf 'Current trial: '
    cat "$STATE_DIR/current_trial"
else
    echo "Current trial: none"
fi

if [ -s "$STATE_DIR/STOP_REQUESTED" ]; then
    echo
    echo "Stop decision:"
    sed 's/^/  /' "$STATE_DIR/STOP_REQUESTED"
fi

if [ -d "$STATE_DIR/pending" ]; then
    PENDING_COUNT="$(find "$STATE_DIR/pending" -maxdepth 1 -type f | wc -l)"
    echo "Pending analyses: $PENDING_COUNT"
fi

if [ -d "$STATE_DIR/groups" ]; then
    GROUP_COUNT="$(find "$STATE_DIR/groups" -maxdepth 1 -name '*.json' -type f | wc -l)"
    echo "Registered groups: $GROUP_COUNT"
fi

if [ -f "$STATE_DIR/watcher.log" ]; then
    echo
    echo "Recent watcher log:"
    tail -n 20 "$STATE_DIR/watcher.log"
fi
