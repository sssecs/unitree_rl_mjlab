#!/usr/bin/env bash
set -euo pipefail

# ===== 本地项目 =====
LOCAL_PATH="/mnt/hdd/humanoid_locomotion/whole_body_policy/unitree_rl_mjlab"

# ===== 远程服务器 / 容器 SSH =====
REMOTE_USER="dev"
REMOTE_HOST="172.52.0.250"
REMOTE_PORT="31001"

# 修改成你希望同步到容器里的目录
REMOTE_PATH="/home/dev/unitree_rl_mjlab"

echo "Syncing:"
echo "  ${LOCAL_PATH}/"
echo "-> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"
echo

rsync -avz \
  --info=progress2 \
  --filter=':- .gitignore' \
  --exclude='.git/' \
  -e "ssh -p ${REMOTE_PORT}" \
  "${LOCAL_PATH}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

echo
echo "Done."