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


def descriptor_knee_contact_match_reward(
  env,
  command_name: str,
  activation_height: float = 0.20,
  activation_std: float = 0.03,
  side_temperature: float = 0.05,
  approach_std: float = 0.10,
  contact_weight: float = 0.75,
) -> torch.Tensor:
  """Match descriptor-selected knee support with dense approach and contact.

  The descriptor knee-ground distances select the support side. The term is
  inactive until either target knee approaches the ground, so standing and
  high-workspace motions receive no knee-contact incentive.  Height matching
  gives a dense path toward contact; actual sensor contact supplies the final
  support bonus.
  """
  command = _base._command(env, command_name)
  result = torch.zeros(env.num_envs, device=env.device)
  env_ids = command.motion_tracking_active.nonzero().flatten()
  if env_ids.numel() == 0:
    return result

  human, valid = command.human_style_target(env_ids)
  knee_height = human[:, 3:5]
  nearest_height = torch.min(knee_height, dim=-1).values
  activation = torch.sigmoid(
    (activation_height - nearest_height) / activation_std
  )
  target_side = torch.softmax(-knee_height / side_temperature, dim=-1)
  left_contact, right_contact = command.knee_contact_masks()
  actual_contact = torch.stack(
    (left_contact[env_ids], right_contact[env_ids]), dim=-1
  ).to(human.dtype)
  matched_contact = torch.sum(target_side * actual_contact, dim=-1)
  robot_knee_height = command.robot_style_descriptor()[env_ids, 3:5]
  approach = torch.sum(
    target_side
    * torch.exp(-torch.square((robot_knee_height - knee_height) / approach_std)),
    dim=-1,
  )
  contact_weight = float(max(0.0, min(1.0, contact_weight)))
  support_reward = (
    (1.0 - contact_weight) * approach + contact_weight * matched_contact
  )
  result[env_ids] = valid.to(human.dtype) * activation * support_reward

  if "descriptor_knee_contact_target" in command.metrics:
    command.metrics["descriptor_knee_contact_target"][env_ids] = activation
  if "descriptor_knee_contact_match" in command.metrics:
    command.metrics["descriptor_knee_contact_match"][env_ids] = (
      activation * matched_contact
    )
  if "descriptor_knee_support_approach" in command.metrics:
    command.metrics["descriptor_knee_support_approach"][env_ids] = (
      activation * approach
    )
  if "style_knee_reward" in command.metrics:
    command.metrics["style_knee_reward"][env_ids] = (
      activation * support_reward
    )
  return result


def kneel_human_style_reward(
  env,
  command_name: str,
  shape_gate_floor: float = 0.25,
  torso_huber_weight: float = 1.0,
  torso_huber_beta: float = 0.25,
) -> torch.Tensor:
  """Static-kneel descriptor reward with exploration floor and safe gating.

  Compared with the existing descriptor reward:

  * the wrist-tracking gate has a 0.20 floor, so the policy can discover a
    kneeling strategy before it already tracks low wrist targets accurately;
  * in ``reward_gate`` balance mode, balance only attenuates the shape part of
    style and never deletes the semantic torso/knee signal entirely;
  * torso pitch uses a Huber penalty that still distinguishes the correct bend
    direction when the posture is far from the target; this part is not gated
    by wrist tracking.
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
  torso_loss = torch.nn.functional.smooth_l1_loss(
    torso_error, torch.zeros_like(torso_error),
    beta=torso_huber_beta, reduction="none",
  )

  shoulder_error = torch.linalg.vector_norm(
    robot[:, 6:8] - human[:, 6:8], dim=-1
  )
  shoulder_reward = torch.exp(-torch.square(shoulder_error / 0.12))

  # Keep shape components separate from the ungated torso correction.
  shape_reward = (
    0.30 * pelvis_reward
    + 0.35 * leg_reward
    + 0.20 * shoulder_reward
  )
  # The torso loss is separate from the wrist gate: the previous narrow
  # exponential had almost no signal for a policy bending backward.
  semantic_reward = torch.zeros_like(shape_reward)

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

  gated_style_reward = semantic_reward + shape_balance_gate * shape_reward
  active = valid.to(gated_style_reward.dtype)
  result[env_ids] = active * (
    task_gate * stability_gate * gated_style_reward
    - torso_huber_weight * torso_loss
  )

  values = {
    "style_active_fraction": active,
    "style_pelvis_reward": active * pelvis_reward,
    "style_leg_reward": active * leg_reward,
    # Historical metric name; now reports the negative Huber torso contribution.
    "style_torso_reward": -active * torso_huber_weight * torso_loss,
    "style_shoulder_pelvis_reward": active * shoulder_reward,
    "style_knee_reward": torch.zeros_like(active),
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
