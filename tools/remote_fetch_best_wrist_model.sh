#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${G1_REMOTE_HOST:-unitree-trainer}"
REMOTE_CHECKPOINT="${G1_REMOTE_CHECKPOINT:-/home/dev/unitree_rl_mjlab/logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-12_20-17-07/model_4999.pt}"
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
DEFAULT_DESTINATION="$PROJECT_ROOT/models/best_wrist/model.pt"
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

echo "Fetching recommended G1 wrist candidate (not held-out validated):"
echo "  $REMOTE_HOST:$REMOTE_CHECKPOINT"
echo "  -> $DESTINATION"
scp "$REMOTE_HOST:$REMOTE_CHECKPOINT" "$TEMP_FILE"

if [ -f "$DESTINATION" ]; then
  if cmp -s "$TEMP_FILE" "$DESTINATION"; then
    echo "Already up to date: $DESTINATION"
    printf 'Source: %s:%s\nFetched: %s\n' "$REMOTE_HOST" "$REMOTE_CHECKPOINT" "$(date -Is)" > "${DESTINATION}.source.txt"
    exit 0
  fi

  BACKUP="${DESTINATION}.bak.$(date +%Y%m%d_%H%M%S)"
  mv "$DESTINATION" "$BACKUP"
  if [ -f "${DESTINATION}.source.txt" ]; then
    mv "${DESTINATION}.source.txt" "${BACKUP}.source.txt"
  fi
  echo "Existing different checkpoint backed up to: $BACKUP"
fi

mv "$TEMP_FILE" "$DESTINATION"
printf 'Source: %s:%s\nFetched: %s\n' "$REMOTE_HOST" "$REMOTE_CHECKPOINT" "$(date -Is)" > "${DESTINATION}.source.txt"
trap - EXIT
echo "Saved: $DESTINATION"
