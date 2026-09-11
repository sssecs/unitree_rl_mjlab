#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${G1_REMOTE_HOST:-unitree-trainer}"
REMOTE_CHECKPOINT="${G1_REMOTE_CHECKPOINT:-/home/dev/unitree_rl_mjlab/logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_12-12-08/model_4999.pt}"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DEFAULT_DESTINATION="$PROJECT_ROOT/logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-10_12-12-08/model_4999.pt"
DESTINATION="${1:-$DEFAULT_DESTINATION}"

for command_name in cmp scp; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $command_name"
    exit 1
  fi
done

mkdir -p "$(dirname "$DESTINATION")"
TEMP_FILE="${DESTINATION}.download.$$"
trap 'rm -f "$TEMP_FILE"' EXIT

echo "Fetching best balanced G1 wrist model:"
echo "  $REMOTE_HOST:$REMOTE_CHECKPOINT"
echo "  -> $DESTINATION"
scp "$REMOTE_HOST:$REMOTE_CHECKPOINT" "$TEMP_FILE"

if [ -f "$DESTINATION" ]; then
  if cmp -s "$TEMP_FILE" "$DESTINATION"; then
    echo "Already up to date: $DESTINATION"
    exit 0
  fi

  BACKUP="${DESTINATION}.bak.$(date +%Y%m%d_%H%M%S)"
  mv "$DESTINATION" "$BACKUP"
  echo "Existing different checkpoint backed up to: $BACKUP"
fi

mv "$TEMP_FILE" "$DESTINATION"
trap - EXIT
echo "Saved: $DESTINATION"
