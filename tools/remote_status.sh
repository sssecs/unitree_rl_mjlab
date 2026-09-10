#!/usr/bin/env bash
set -euo pipefail

echo "=== tmux ==="
ssh unitree-trainer \
  'tmux ls 2>/dev/null || echo "no tmux sessions"'

echo
echo "=== GPU ==="
ssh unitree-trainer '
  export LD_LIBRARY_PATH="/home/dev/nvidia-550.127/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LD_PRELOAD="/home/dev/nvidia-550.127/lib/libnvidia-ml.so.1"
  exec /usr/bin/nvidia-smi
'