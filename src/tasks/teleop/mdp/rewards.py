from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import SparseWholeBodyCommand, _yaw_quat_tensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> SparseWholeBodyCommand:
  return cast(
    SparseWholeBodyCommand,
    env.command_manager.get_term(command_name),
  )


def wrist_position_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Strong task reward: absolute world-space wrist positions."""
  command = _command(env, command_name)

  left_error2 = torch.sum(
    torch.square(
      command.cmd_left_wrist_pos_w
      - command.sim_left_wrist_pos_w
    ),
    dim=-1,
  )
  right_error2 = torch.sum(
    torch.square(
      command.cmd_right_wrist_pos_w
      - command.sim_right_wrist_pos_w
    ),
    dim=-1,
  )
  return torch.exp(
    -0.5 * (left_error2 + right_error2) / std**2
  )


def _wrist_position_rms_error(
  command: SparseWholeBodyCommand,
) -> torch.Tensor:
  """RMS of the two scalar wrist-position errors."""
  left = torch.linalg.vector_norm(
    command.cmd_left_wrist_pos_w - command.sim_left_wrist_pos_w,
    dim=-1,
  )
  right = torch.linalg.vector_norm(
    command.cmd_right_wrist_pos_w - command.sim_right_wrist_pos_w,
    dim=-1,
  )
  return torch.sqrt(0.5 * (left.square() + right.square()) + 1.0e-12)


def wrist_position_coarse_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Long-range wrist tracking signal with slowly decaying gradient."""
  command = _command(env, command_name)
  return torch.exp(-_wrist_position_rms_error(command) / std)


def wrist_position_fine_tanh(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Near-target wrist precision reward."""
  command = _command(env, command_name)
  return 1.0 - torch.tanh(_wrist_position_rms_error(command) / std)


def inter_wrist_position_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Track relative left-right wrist displacement."""
  command = _command(env, command_name)
  target = command.cmd_left_wrist_pos_w - command.cmd_right_wrist_pos_w
  actual = command.sim_left_wrist_pos_w - command.sim_right_wrist_pos_w
  error = torch.linalg.vector_norm(target - actual, dim=-1)
  return torch.exp(-error / std)


def wrist_linear_velocity_tracking_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Mean bilateral squared wrist linear-velocity error."""
  command = _command(env, command_name)
  left = command.cmd_left_wrist_lin_vel_w - command.sim_left_wrist_lin_vel_w
  right = command.cmd_right_wrist_lin_vel_w - command.sim_right_wrist_lin_vel_w
  return 0.5 * (
    torch.sum(torch.square(left), dim=-1)
    + torch.sum(torch.square(right), dim=-1)
  )


def wrist_orientation_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Strong task reward: absolute world-space wrist orientations."""
  command = _command(env, command_name)

  left = quat_error_magnitude(
    command.cmd_left_wrist_quat_w,
    command.sim_left_wrist_quat_w,
  )
  right = quat_error_magnitude(
    command.cmd_right_wrist_quat_w,
    command.sim_right_wrist_quat_w,
  )
  return torch.exp(
    -0.5 * (left.square() + right.square()) / std**2
  )


def shoulder_mid_xy_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  tolerance: float,
) -> torch.Tensor:
  """Soft absolute shoulder-mid XY placement reward."""
  command = _command(env, command_name)

  error = torch.linalg.vector_norm(
    command.cmd_shoulder_mid_xy_w
    - command.sim_shoulder_mid_pos_w[:, :2],
    dim=-1,
  )
  excess = torch.relu(error - tolerance)
  return torch.exp(-excess.square() / std**2)


def shoulder_height_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  tolerance: float,
) -> torch.Tensor:
  """Soft bilateral shoulder-height preference."""
  command = _command(env, command_name)

  left = torch.abs(
    command.cmd_left_shoulder_height
    - command.sim_left_shoulder_height
  )
  right = torch.abs(
    command.cmd_right_shoulder_height
    - command.sim_right_shoulder_height
  )

  left_excess = torch.relu(left - tolerance)
  right_excess = torch.relu(right - tolerance)
  error2 = 0.5 * (
    left_excess.square()
    + right_excess.square()
  )
  return torch.exp(-error2 / std**2)


def shoulder_heading_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  tolerance: float,
) -> torch.Tensor:
  """Soft absolute shoulder-derived heading reward."""
  command = _command(env, command_name)

  error = quat_error_magnitude(
    _yaw_quat_tensor(command.cmd_shoulder_heading_w),
    command.sim_shoulder_yaw_quat_w,
  )
  excess = torch.relu(error - tolerance)
  return torch.exp(-excess.square() / std**2)


def recovery_joint_posture_l2(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Nominal-joint posture penalty active only during post-motion recovery.

  The trajectory itself remains free to use whole-body redundancy.  Once the
  recorded motion ends, this term helps the sparse return command converge to
  an ordinary standing configuration rather than a kinematically contorted
  pose that happens to satisfy wrist/shoulder targets.
  """
  command = _command(env, command_name)
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None

  joint_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos[:, joint_ids]
  default = default_joint_pos[:, joint_ids]
  error = torch.mean(torch.square(joint_pos - default), dim=-1)
  return error * command.recovery_started.to(error.dtype)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collision impulses."""
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data

  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()

  assert data.found is not None
  return data.found.squeeze(-1)


def _contact_found_mask(sensor: ContactSensor) -> torch.Tensor:
  """Collapse contact sensor match/slot dimensions to per-primary masks."""
  assert sensor.data.found is not None
  mask = sensor.data.found > 0
  while mask.ndim > 2:
    mask = torch.any(mask, dim=-1)
  return mask.to(torch.float32)


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize horizontal foot speed only while that foot contacts ground."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  in_contact = _contact_found_mask(sensor)
  speed_sq = torch.sum(
    torch.square(asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]),
    dim=-1,
  )
  if in_contact.shape != speed_sq.shape:
    in_contact = in_contact.reshape_as(speed_sq)
  return torch.sum(speed_sq * in_contact, dim=-1)


def quiet_feet_when_task_stable(
  env: ManagerBasedRlEnv,
  command_name: str,
  wrist_error_threshold: float,
  command_wrist_speed_threshold: float,
  command_shoulder_speed_threshold: float,
  command_heading_rate_threshold: float,
  base_speed_threshold: float,
  base_ang_speed_threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize needless foot motion only for a quasi-static, already-tracked task."""
  command = _command(env, command_name)
  asset: Entity = env.scene[asset_cfg.name]

  left_error = torch.linalg.vector_norm(
    command.cmd_left_wrist_pos_w - command.sim_left_wrist_pos_w, dim=-1
  )
  right_error = torch.linalg.vector_norm(
    command.cmd_right_wrist_pos_w - command.sim_right_wrist_pos_w, dim=-1
  )
  wrist_error = 0.5 * (left_error + right_error)

  wrist_cmd_speed = torch.maximum(
    torch.linalg.vector_norm(command.cmd_left_wrist_lin_vel_w, dim=-1),
    torch.linalg.vector_norm(command.cmd_right_wrist_lin_vel_w, dim=-1),
  )
  shoulder_cmd_speed = torch.linalg.vector_norm(
    command.cmd_shoulder_mid_lin_vel_ew, dim=-1
  )
  heading_rate = torch.abs(command.cmd_shoulder_heading_rate_w)

  base_speed = torch.linalg.vector_norm(
    asset.data.root_link_lin_vel_b[:, :2], dim=-1
  )
  base_ang_speed = torch.linalg.vector_norm(
    asset.data.root_link_ang_vel_b, dim=-1
  )

  stable = (
    (~command.warmup_active)
    & (wrist_error < wrist_error_threshold)
    & (wrist_cmd_speed < command_wrist_speed_threshold)
    & (shoulder_cmd_speed < command_shoulder_speed_threshold)
    & (heading_rate < command_heading_rate_threshold)
    & (base_speed < base_speed_threshold)
    & (base_ang_speed < base_ang_speed_threshold)
  )

  foot_speed_sq = torch.sum(
    torch.square(asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]),
    dim=(-2, -1),
  )
  return foot_speed_sq * stable.to(foot_speed_sq.dtype)


def undesired_ground_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float,
) -> torch.Tensor:
  """Count prohibited robot-ground contacts; allowed geoms are sensor-defined."""
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.force_history is not None:
    magnitude = torch.linalg.vector_norm(sensor.data.force_history, dim=-1)
    return (magnitude > force_threshold).any(dim=-1).sum(dim=-1).float()

  assert sensor.data.found is not None
  found = sensor.data.found
  while found.ndim > 1:
    found = found.any(dim=-1)
  return found.float()
