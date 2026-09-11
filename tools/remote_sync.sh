#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="unitree-trainer"
REMOTE_REPO="/home/dev/unitree_rl_mjlab"

if ssh "$REMOTE_HOST" "command -v rsync >/dev/null 2>&1"; then
  rsync -av \
    --exclude '.git/' \
    --exclude '.autotune/' \
    --exclude 'logs/' \
    ./ \
    "$REMOTE_HOST:$REMOTE_REPO/"
else
  echo "Remote rsync is unavailable; using non-deleting tar stream fallback."
  git ls-files --cached --others --exclude-standard -z \
    | tar --no-recursion --null --files-from=- -cf - \
    | ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && tar -m -xf -"
fi
