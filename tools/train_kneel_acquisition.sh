#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash tools/train_kneel_acquisition.sh /data/synth left bent 2048
#   bash tools/train_kneel_acquisition.sh /data/synth left home 2048
#   bash tools/train_kneel_acquisition.sh /data/synth right bent 2048

DATASET=${1:?dataset root required}
SIDE=${2:-left}       # left | right; use one side per acquisition run
STAGE=${3:-bent}      # bent | home
NUM_ENVS=${4:-2048}

if [[ "${SIDE}" != "left" && "${SIDE}" != "right" ]]; then
  echo "SIDE must be left or right during K0 acquisition" >&2
  exit 2
fi

export UNITREE_TELEOP_COMMAND_DIR="${DATASET}"
export UNITREE_TELEOP_KNEEL_SIDE="${SIDE}"

case "${STAGE}" in
  bent)
    TASK="Unitree-G1-Teleop-Kneel-Acquire"
    ;;
  home)
    TASK="Unitree-G1-Teleop-Kneel-Acquire-Home"
    ;;
  *)
    echo "STAGE must be bent or home" >&2
    exit 2
    ;;
esac

python scripts/train.py "${TASK}" \
  --env.scene.num-envs="${NUM_ENVS}" \
  --agent.run-name="kneel_acquire_${SIDE}_${STAGE}" \
  --agent.logger=tensorboard
