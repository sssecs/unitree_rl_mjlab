#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:?Usage: $0 <session-name>}"

ssh unitree-trainer \
  "tmux send-keys -t '$SESSION' C-c"
