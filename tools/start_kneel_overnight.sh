#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

LOCAL_SESSION="g1_kneel_descriptor_overnight_v1"
VALIDATION_SESSION="g1_kneel_descriptor_overnight_v1_validate"
REMOTE_REPO="/home/dev/unitree_rl_mjlab"
REMOTE_DATA="/home/dev/EgoDex-PICO-kneel-synth"
FIRST_PLAN="experiments/g1_kneel_descriptor_seed4201_v1.json"
SECOND_PLAN="experiments/g1_kneel_descriptor_compare_v1.json"
LOG_FILE="$PROJECT_ROOT/.autotune/${LOCAL_SESSION}.log"

if [[ "${1:-}" == "--worker" ]]; then
  ./tools/remote_train.sh \
    "$VALIDATION_SESSION" Unitree-G1-Teleop-Kneel-TargetDescriptor \
    --gpu-ids '[0]' \
    --env.scene.num-envs=256 \
    --env.commands.teleop.command-dir="$REMOTE_DATA" \
    --agent.max-iterations=3 \
    --agent.logger=tensorboard \
    --agent.run-name="$VALIDATION_SESSION"

  VALIDATION_RUN_DIR="$REMOTE_REPO/.autotune/$VALIDATION_SESSION"
  for ((attempt=0; attempt<180; attempt++)); do
    state="$(ssh unitree-trainer "if test -f '$VALIDATION_RUN_DIR/DONE'; then cat '$VALIDATION_RUN_DIR/exit_code'; else echo RUNNING; fi")"
    if [[ "$state" != "RUNNING" ]]; then
      if [[ "$state" != "0" ]]; then
        ./tools/remote_log.sh "$VALIDATION_SESSION" 120
        echo "Validation failed with exit code $state; screening was not launched." >&2
        exit 1
      fi
      break
    fi
    sleep 10
  done
  if [[ "$state" == "RUNNING" ]]; then
    echo "Validation did not finish within 30 minutes; screening was not launched." >&2
    exit 1
  fi
  if ! ssh unitree-trainer "grep -q 'Iteration time:' '$VALIDATION_RUN_DIR/train.log'"; then
    ./tools/remote_log.sh "$VALIDATION_SESSION" 120
    echo "Validation did not complete a PPO iteration; screening was not launched." >&2
    exit 1
  fi
  if ssh unitree-trainer "grep -Eq 'Traceback|RuntimeError' '$VALIDATION_RUN_DIR/train.log'"; then
    ./tools/remote_log.sh "$VALIDATION_SESSION" 120
    echo "Validation log contains a Python error; screening was not launched." >&2
    exit 1
  fi
  echo "Validation passed: 256 environments on one GPU, 3 PPO iterations."

  python3 tools/remote_two_wave_sweep.py "$FIRST_PLAN" "$SECOND_PLAN"
  exit 0
fi

if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 2
fi

command -v tmux >/dev/null
ssh unitree-trainer "test -f '$REMOTE_DATA/SHA256SUMS'"
ssh unitree-trainer "test -w '$REMOTE_REPO'"
git diff --check

auto_sessions="$(ssh unitree-trainer "tmux list-sessions -F '#S' 2>/dev/null || true")"
if [[ -n "$auto_sessions" ]]; then
  echo "Remote tmux sessions are already active:" >&2
  echo "$auto_sessions" >&2
  exit 1
fi
if tmux has-session -t "$LOCAL_SESSION" 2>/dev/null; then
  echo "Local orchestration session already exists: $LOCAL_SESSION" >&2
  exit 1
fi

gpu_pids="$(ssh unitree-trainer   'LD_LIBRARY_PATH=/home/dev/nvidia-550.127/lib LD_PRELOAD=/home/dev/nvidia-550.127/lib/libnvidia-ml.so.1 /usr/bin/nvidia-smi --query-compute-apps=pid --format=csv,noheader')"
if [[ -n "$gpu_pids" ]]; then
  echo "Remote GPUs already have compute processes:" >&2
  echo "$gpu_pids" >&2
  exit 1
fi

./tools/remote_status.sh
./tools/remote_sync.sh
mkdir -p "$PROJECT_ROOT/.autotune"
tmux new-session -d -s "$LOCAL_SESSION" \
  "cd '$PROJECT_ROOT' && bash tools/start_kneel_overnight.sh --worker > '$LOG_FILE' 2>&1"

echo "Started local orchestration tmux: $LOCAL_SESSION"
echo "The validation and every training run use detached remote tmux sessions."
echo "Watch: tail -f '$LOG_FILE'"
echo "Remote status: ./tools/remote_status.sh"
