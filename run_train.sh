#!/usr/bin/env bash
set -euo pipefail
umask 0002

source /home/dev/miniconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab

export LD_LIBRARY_PATH="$CONDA_PREFIX/cuda-compat:/home/dev/nvidia-550.127/lib:$CONDA_PREFIX/lib"
export LD_PRELOAD="/home/dev/nvidia-550.127/lib/libnvidia-ml.so.1"
export NCCL_CUMEM_HOST_ENABLE=0

cd /home/dev/unitree_rl_mjlab

exec python scripts/train.py "$@"