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


def _descriptor_support_side(
  human: torch.Tensor,
  side_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return desired L/R knee support weights and a smooth activation.

  Prefer the explicit knee pseudo-contact semantics (descriptor dims 15:17).
  Before pseudo-contact becomes confident, fall back to the lower human knee.
  """
  pseudo = torch.clamp(human[:, 15:17], 0.0, 1.0)
  pseudo_sum = torch.sum(pseudo, dim=-1, keepdim=True)
  fallback = torch.softmax(-human[:, 3:5] / side_temperature, dim=-1)
  desired = torch.where(
    pseudo_sum > 0.10,
    pseudo / torch.clamp(pseudo_sum, min=1.0e-6),
    fallback,
  )
  nearest_human_knee = torch.min(human[:, 3:5], dim=-1).values
  activation = torch.sigmoid((0.45 - nearest_human_knee) / 0.08)
  return desired, activation


def descriptor_knee_contact_match_reward(
  env,
  command_name: str,
  activation_height: float = 0.45,
  activation_std: float = 0.08,
  side_temperature: float = 0.05,
  approach_std: float = 0.35,
  contact_weight: float = 3.0,
  approach_weight: float = 1.25,
  asymmetry_weight: float = 0.75,
  wrong_contact_weight: float = 1.0,
  asymmetry_margin: float = 0.04,
  asymmetry_std: float = 0.06,
) -> torch.Tensor:
  """Dense single-knee support acquisition reward.

  Human knee height is not regressed directly to G1 knee-link height. Human
  descriptor dims 15:17 identify support topology; G1 knee-link height is only
  a monotonic approach proxy; actual shin/linkage contacts provide the terminal
  support reward.
  """
  command = _base._command(env, command_name)
  result = torch.zeros(env.num_envs, device=env.device)
  env_ids = command.motion_tracking_active.nonzero().flatten()
  if env_ids.numel() == 0:
    return result

  human, valid = command.human_style_target(env_ids)
  desired, default_activation = _descriptor_support_side(
    human, max(float(side_temperature), 1.0e-4)
  )

  nearest_human_knee = torch.min(human[:, 3:5], dim=-1).values
  activation = torch.sigmoid(
    (float(activation_height) - nearest_human_knee)
    / max(float(activation_std), 1.0e-4)
  )
  activation = torch.maximum(activation, 0.5 * default_activation)

  robot = command.robot_style_descriptor()[env_ids]
  robot_knee_height = torch.clamp(robot[:, 3:5], min=0.0)

  # Dense monotonic descent signal: still non-zero from a standing posture.
  approach = torch.sum(
    desired
    * torch.exp(
      -robot_knee_height / max(float(approach_std), 1.0e-4)
    ),
    dim=-1,
  )

  # Prefer a one-knee topology rather than the easy symmetric squat attractor.
  left_lower = torch.sigmoid(
    (
      robot_knee_height[:, 1]
      - robot_knee_height[:, 0]
      - float(asymmetry_margin)
    ) / max(float(asymmetry_std), 1.0e-4)
  )
  right_lower = torch.sigmoid(
    (
      robot_knee_height[:, 0]
      - robot_knee_height[:, 1]
      - float(asymmetry_margin)
    ) / max(float(asymmetry_std), 1.0e-4)
  )
  asymmetry = torch.sum(
    desired * torch.stack((left_lower, right_lower), dim=-1), dim=-1
  )

  left_contact, right_contact = command.knee_contact_masks()
  actual_contact = torch.stack(
    (left_contact[env_ids], right_contact[env_ids]), dim=-1
  ).to(human.dtype)
  matched_contact = torch.sum(desired * actual_contact, dim=-1)
  wrong_contact = torch.sum((1.0 - desired) * actual_contact, dim=-1)

  support_reward = (
    float(approach_weight) * approach
    + float(contact_weight) * matched_contact
    + float(asymmetry_weight) * asymmetry
    - float(wrong_contact_weight) * wrong_contact
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
) -> torch.Tensor:
  """Descriptor reward without a large negative torso basin.

  Torso is a bounded positive cosine reward. Knee support topology is handled
  separately by ``descriptor_knee_contact_match_reward``.
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

  pelvis_reward = torch.exp(
    -torch.square((robot[:, 0] - human[:, 0]) / 0.10)
  )
  leg_error_sq = 0.5 * torch.sum(
    torch.square(robot[:, 1:3] - human[:, 1:3]), dim=-1
  )
  leg_reward = torch.exp(-leg_error_sq / 0.14**2)

  torso_error = torch.atan2(
    torch.sin(robot[:, 5] - human[:, 5]),
    torch.cos(robot[:, 5] - human[:, 5]),
  )
  torso_reward = 0.5 * (1.0 + torch.cos(torso_error))

  shoulder_error = torch.linalg.vector_norm(
    robot[:, 6:8] - human[:, 6:8], dim=-1
  )
  shoulder_reward = torch.exp(-torch.square(shoulder_error / 0.16))

  shape_reward = (
    0.30 * pelvis_reward
    + 0.35 * leg_reward
    + 0.20 * shoulder_reward
  )
  semantic_reward = 0.15 * torso_reward

  wrist_error = _base._wrist_position_rms_error(command)[env_ids]
  raw_task_gate = torch.sigmoid((0.10 - wrist_error) / 0.025)
  task_gate = 0.20 + 0.80 * raw_task_gate
  stability_gate = torch.ones_like(task_gate)

  balance_gate = command.balance_style_gate[env_ids]
  if command.cfg.balance_mode == "reward_gate":
    shape_gate_floor = float(max(0.0, min(1.0, shape_gate_floor)))
    shape_balance_gate = (
      shape_gate_floor + (1.0 - shape_gate_floor) * balance_gate
    )
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


def kneel_acquisition_style_reward(
  env,
  command_name: str,
) -> torch.Tensor:
  """Broad positive-only posture shaping for the K0 acquisition stage."""
  command = _base._command(env, command_name)
  result = torch.zeros(env.num_envs, device=env.device)
  if command.cfg.style_mode == "baseline":
    return result

  env_ids = command.motion_tracking_active.nonzero().flatten()
  if env_ids.numel() == 0:
    return result

  human, valid = command.human_style_target(env_ids)
  robot = command.robot_style_descriptor()[env_ids]

  pelvis_reward = torch.exp(
    -torch.square((robot[:, 0] - human[:, 0]) / 0.18)
  )
  leg_error_sq = 0.5 * torch.sum(
    torch.square(robot[:, 1:3] - human[:, 1:3]), dim=-1
  )
  leg_reward = torch.exp(-leg_error_sq / 0.22**2)

  torso_error = torch.atan2(
    torch.sin(robot[:, 5] - human[:, 5]),
    torch.cos(robot[:, 5] - human[:, 5]),
  )
  torso_reward = 0.5 * (1.0 + torch.cos(torso_error))

  shoulder_error = torch.linalg.vector_norm(
    robot[:, 6:8] - human[:, 6:8], dim=-1
  )
  shoulder_reward = torch.exp(-torch.square(shoulder_error / 0.24))

  reward = (
    0.40 * pelvis_reward
    + 0.35 * leg_reward
    + 0.15 * torso_reward
    + 0.10 * shoulder_reward
  )
  active = valid.to(reward.dtype)
  result[env_ids] = active * reward

  values = {
    "style_active_fraction": active,
    "style_pelvis_reward": active * pelvis_reward,
    "style_leg_reward": active * leg_reward,
    "style_torso_reward": active * torso_reward,
    "style_shoulder_pelvis_reward": active * shoulder_reward,
    "style_task_gate": active,
    "style_stability_gate": active,
    "style_robot_pelvis_height": active * robot[:, 0],
    "style_human_pelvis_height": active * human[:, 0],
    "style_task_gate_raw": active,
    "style_shape_balance_gate": active,
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
  return _base.quiet_feet_when_task_stable(
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
