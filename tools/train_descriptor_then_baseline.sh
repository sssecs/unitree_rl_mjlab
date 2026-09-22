#!/usr/bin/env bash
set -e

python scripts/train.py Unitree-G1-Teleop \
  --env.scene.num-envs=4096 \
  --agent.logger=tensorboard \
  --agent.seed=42 \
  --env.commands.teleop.style-mode=descriptor \
  --env.commands.teleop.balance-mode=reward \
  --env.rewards.com-balance.weight=0.30 \
  --agent.run-name=teleop_com_a_reward_only

python scripts/train.py Unitree-G1-Teleop \
  --env.scene.num-envs=4096 \
  --agent.logger=tensorboard \
  --agent.seed=42 \
  --env.commands.teleop.style-mode=descriptor \
  --env.commands.teleop.balance-mode=reward_gate \
  --env.rewards.com-balance.weight=0.30 \
  --agent.run-name=teleop_com_b_reward_gate

python scripts/train.py Unitree-G1-Teleop \
  --env.scene.num-envs=4096 \
  --agent.logger=tensorboard \
  --agent.seed=42 \
  --env.commands.teleop.style-mode=descriptor \
  --env.commands.teleop.balance-mode=reward_gate \
  --env.rewards.com-balance.weight=0.15 \
  --agent.run-name=teleop_com_c_soft_reward_gate
