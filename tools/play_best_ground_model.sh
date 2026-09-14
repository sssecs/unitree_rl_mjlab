#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
export G1_WRIST_CHECKPOINT="${G1_WRIST_CHECKPOINT:-$PROJECT_ROOT/models/best_ground/model.pt}"
export G1_WRIST_PROFILE="${G1_WRIST_PROFILE:-operation-capability}"
exec "$PROJECT_ROOT/tools/play_best_wrist_model.sh" "$@"
