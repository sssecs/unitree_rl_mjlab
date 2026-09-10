#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

rsync -av \
  --exclude '.git/' \
  --exclude 'logs/' \
  ./ \
  unitree-trainer:/home/dev/unitree_rl_mjlab/