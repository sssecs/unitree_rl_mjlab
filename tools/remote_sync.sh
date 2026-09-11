#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

REMOTE_HOST="unitree-trainer"
REMOTE_REPO="/home/dev/unitree_rl_mjlab"
REMOTE_STATE="$REMOTE_REPO/.autotune"
MANIFEST="$(mktemp)"
cleanup() {
  rm -f -- "$MANIFEST"
}
trap cleanup EXIT

# This is exactly the set copied by the tar fallback. Keeping it remotely lets
# us remove only files previously managed by this script, never logs or other
# remote-only artifacts.
git ls-files --cached --others --exclude-standard \
  | LC_ALL=C sort -u > "$MANIFEST"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_STATE'"
scp "$MANIFEST" "$REMOTE_HOST:$REMOTE_STATE/sync_manifest.next"

ssh "$REMOTE_HOST" "
set -euo pipefail
if [ -f '$REMOTE_STATE/sync_manifest.current' ]; then
  LC_ALL=C comm -23 \
    '$REMOTE_STATE/sync_manifest.current' \
    '$REMOTE_STATE/sync_manifest.next' |
  while IFS= read -r relative_path; do
    case \"\$relative_path\" in
      ''|/*|../*|*/../*|*/..)
        echo \"Unsafe stale path in sync manifest: \$relative_path\" >&2
        exit 1
        ;;
      .git/*|.autotune/*|logs/*|results/*|wandb/*)
        echo \"Refusing to prune protected path: \$relative_path\" >&2
        exit 1
        ;;
    esac
    target='$REMOTE_REPO/'\"\$relative_path\"
    if [ -f \"\$target\" ] || [ -L \"\$target\" ]; then
      rm -f -- \"\$target\"
      echo \"Pruned stale managed file: \$relative_path\"
    fi
  done
fi
"

if ssh "$REMOTE_HOST" "command -v rsync >/dev/null 2>&1"; then
  rsync -av \
    --files-from="$MANIFEST" \
    ./ \
    "$REMOTE_HOST:$REMOTE_REPO/"
else
  echo "Remote rsync is unavailable; using manifest-pruned tar fallback."
  git ls-files --cached --others --exclude-standard -z \
    | tar --no-recursion --null --files-from=- -cf - \
    | ssh "$REMOTE_HOST" "cd '$REMOTE_REPO' && tar -m -xf -"
fi

ssh "$REMOTE_HOST" \
  "mv '$REMOTE_STATE/sync_manifest.next' '$REMOTE_STATE/sync_manifest.current'"
