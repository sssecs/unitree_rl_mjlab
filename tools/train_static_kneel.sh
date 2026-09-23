#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/train_static_kneel.sh /data/synth left descriptor 2048
#   bash tools/train_static_kneel.sh /data/synth any baseline 2048

DATASET=${1:?dataset root required}
SIDE=${2:-left}          # any | left | right
MODE=${3:-descriptor}    # descriptor | baseline
NUM_ENVS=${4:-2048}

export UNITREE_TELEOP_COMMAND_DIR="${DATASET}"
export UNITREE_TELEOP_KNEEL_SIDE="${SIDE}"

if [[ "${MODE}" == "baseline" ]]; then
  TASK="Unitree-G1-Teleop-Kneel-Baseline"
elif [[ "${MODE}" == "descriptor" ]]; then
  TASK="Unitree-G1-Teleop-Kneel"
else
  echo "MODE must be descriptor or baseline" >&2
  exit 2
fi

python scripts/train.py "${TASK}" \
  --env.scene.num-envs="${NUM_ENVS}"
