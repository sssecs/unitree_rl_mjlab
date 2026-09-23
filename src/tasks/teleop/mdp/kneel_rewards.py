from __future__ import annotations

import torch

from . import rewards as _base


def _weight(env, command_name: str) -> torch.Tensor:
  command = _base._command(env, command_name)
  return command.wrist_task_weight_scale


def kneel_wrist_position_coarse_exp(env, command_name: str, std: float):
  return _weight(env, command_name) * _base.wrist_position_coarse_exp(
    env, command_name, std
  )


def kneel_wrist_position_fine_tanh(env, command_name: str, std: float):
  return _weight(env, command_name) * _base.wrist_position_fine_tanh(
    env, command_name, std
  )


def kneel_inter_wrist_position_tracking_exp(
  env, command_name: str, std: float
):
  return _weight(env, command_name) * _base.inter_wrist_position_tracking_exp(
    env, command_name, std
  )


def kneel_wrist_linear_velocity_tracking_l2(env, command_name: str):
  return _weight(env, command_name) * _base.wrist_linear_velocity_tracking_l2(
    env, command_name
  )


def kneel_wrist_orientation_tracking_exp(env, command_name: str, std: float):
  return _weight(env, command_name) * _base.wrist_orientation_tracking_exp(
    env, command_name, std
  )


def kneel_human_style_reward(
  env, command_name: str, shape_gate_floor: float = 0.25
) -> torch.Tensor:
  """Static-kneel descriptor reward with exploration floor and safe gating.

  Compared with the existing descriptor reward:

  * the wrist-tracking gate has a 0.20 floor, so the policy can discover a
    kneeling strategy before it already tracks low wrist targets accurately;
  * in ``reward_gate`` balance mode, balance only attenuates the shape part of
    style and never deletes the semantic torso/knee signal entirely;
  * the selected PICO candidate remains reward-only and is not an actor input.
  """
  command = _base._command(env, command_name)
  result = torch.zeros(env.num_envs, device=env.device)
  if command.cfg.style_mode == "baseline":
    return result

  metric_names = (
    "style_active_fraction",
    "style_pelvis_reward",
    "style_leg_reward",
    "style_torso_reward",
    "style_shoulder_pelvis_reward",
    "style_knee_reward",
    "style_task_gate",
    "style_stability_gate",
    "style_robot_pelvis_height",
    "style_human_pelvis_height",
    "style_task_gate_raw",
    "style_shape_balance_gate",
  )
  for name in metric_names:
    if name in command.metrics:
      command.metrics[name].zero_()

  env_ids = command.motion_tracking_active.nonzero().flatten()
  if env_ids.numel() == 0:
    return result

  human, valid = command.human_style_target(env_ids)
  robot = command.robot_style_descriptor()[env_ids]

  pelvis_reward = torch.exp(-torch.square((robot[:, 0] - human[:, 0]) / 0.08))
  leg_error_sq = 0.5 * torch.sum(
    torch.square(robot[:, 1:3] - human[:, 1:3]), dim=-1
  )
  leg_reward = torch.exp(-leg_error_sq / 0.10**2)

  torso_error = torch.atan2(
    torch.sin(robot[:, 5] - human[:, 5]),
    torch.cos(robot[:, 5] - human[:, 5]),
  )
  torso_reward = torch.exp(-torch.square(torso_error / 0.25))

  shoulder_error = torch.linalg.vector_norm(
    robot[:, 6:8] - human[:, 6:8], dim=-1
  )
  shoulder_reward = torch.exp(-torch.square(shoulder_error / 0.12))

  knee_activation = torch.exp(-torch.square(human[:, 3:5] / 0.12))
  knee_match = torch.exp(
    -torch.square((robot[:, 3:5] - human[:, 3:5]) / 0.09)
  )
  knee_reward = torch.sum(knee_activation * knee_match, dim=-1) / (
    torch.sum(knee_activation, dim=-1) + 1.0e-6
  )

  # Preserve the old total component weights when balance is disabled.
  shape_reward = (
    0.30 * pelvis_reward
    + 0.35 * leg_reward
    + 0.20 * shoulder_reward
  )
  semantic_reward = 0.15 * torso_reward + 0.15 * knee_reward

  wrist_error = _base._wrist_position_rms_error(command)[env_ids]
  raw_task_gate = torch.sigmoid((0.10 - wrist_error) / 0.025)
  task_gate = 0.20 + 0.80 * raw_task_gate
  stability_gate = torch.ones_like(task_gate)

  balance_gate = command.balance_style_gate[env_ids]
  if command.cfg.balance_mode == "reward_gate":
    shape_gate_floor = float(max(0.0, min(1.0, shape_gate_floor)))
    shape_balance_gate = shape_gate_floor + (1.0 - shape_gate_floor) * balance_gate
  else:
    shape_balance_gate = torch.ones_like(balance_gate)

  style_reward = semantic_reward + shape_balance_gate * shape_reward
  active = valid.to(style_reward.dtype)
  result[env_ids] = active * task_gate * stability_gate * style_reward

  values = {
    "style_active_fraction": active,
    "style_pelvis_reward": active * pelvis_reward,
    "style_leg_reward": active * leg_reward,
    "style_torso_reward": active * torso_reward,
    "style_shoulder_pelvis_reward": active * shoulder_reward,
    "style_knee_reward": active * knee_reward,
    "style_task_gate": active * task_gate,
    "style_stability_gate": active * stability_gate,
    "style_robot_pelvis_height": active * robot[:, 0],
    "style_human_pelvis_height": active * human[:, 0],
    "style_task_gate_raw": active * raw_task_gate,
    "style_shape_balance_gate": active * shape_balance_gate,
  }
  for name, value in values.items():
    if name in command.metrics:
      command.metrics[name][env_ids] = value
  return result


def kneel_quiet_feet_when_task_stable(
  env,
  command_name: str,
  wrist_error_threshold: float,
  command_wrist_speed_threshold: float,
  command_shoulder_speed_threshold: float,
  command_heading_rate_threshold: float,
  base_speed_threshold: float,
  base_ang_speed_threshold: float,
  asset_cfg,
  balance_gate_floor: float = 0.0,
):
  penalty = _base.quiet_feet_when_task_stable(
    env,
    command_name,
    wrist_error_threshold,
    command_wrist_speed_threshold,
    command_shoulder_speed_threshold,
    command_heading_rate_threshold,
    base_speed_threshold,
    base_ang_speed_threshold,
    asset_cfg,
    balance_gate_floor,
  )
  return penalty
