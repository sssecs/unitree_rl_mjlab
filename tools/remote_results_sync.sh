#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && $1 != --dry-run ) ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

cd "$(git rev-parse --show-toplevel)"

remote_dir="unitree-trainer:/home/dev/unitree_rl_mjlab/logs/rsl_rl/"
local_dir="logs/rsl_rl/"
mkdir -p "$local_dir"

options=(-rltv --omit-dir-times --ignore-existing)
if [[ $# -eq 1 ]]; then
    options+=(--dry-run)
fi

# A run may have a smoke/validate session name but a timestamp-only log path.
# The training launcher records the actual path in each session's run record.
smoke_dirs="$(ssh unitree-trainer '
  root=/home/dev/unitree_rl_mjlab/logs/rsl_rl/
  for run_dir in /home/dev/unitree_rl_mjlab/.autotune/*; do
    session=${run_dir##*/}
    case "$session" in
      *smoke*|*validate*|*validation*)
        test -f "$run_dir/training_log_dir" || continue
        log_dir=$(cat "$run_dir/training_log_dir")
        case "$log_dir" in
          "$root"*) printf "%s\n" "${log_dir#"$root"}" ;;
        esac
        ;;
    esac
  done
' | LC_ALL=C sort -u)"

excludes=(
    --exclude='*smoke*'
    --exclude='*validate*'
    --exclude='*validation*'
)
while IFS= read -r relative_dir; do
    [[ -z "$relative_dir" ]] && continue
    if [[ ! "$relative_dir" =~ ^[A-Za-z0-9_./-]+$ || "/$relative_dir/" == *"/../"* ]]; then
        echo "Unsafe remote smoke log path: $relative_dir" >&2
        exit 1
    fi
    excludes+=(--exclude="/$relative_dir/***")
done <<< "$smoke_dirs"

rsync "${options[@]}" "${excludes[@]}" "$remote_dir" "$local_dir"
