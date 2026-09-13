#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export G1_REMOTE_CHECKPOINT="${G1_REMOTE_GROUND_CHECKPOINT:-/home/dev/unitree_rl_mjlab/logs/rsl_rl/g1_wrist_recovery_teacher/2026-09-13_03-22-39/model_4999.pt}"
exec "$PROJECT_ROOT/tools/remote_fetch_best_wrist_model.sh" "${1:-$PROJECT_ROOT/models/best_ground/model.pt}"
