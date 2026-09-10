#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:?Usage: $0 <session-name>}"
LINES="${2:-200}"

ssh unitree-trainer \
  "tmux capture-pane -pt '$SESSION' -S -'$LINES'"
