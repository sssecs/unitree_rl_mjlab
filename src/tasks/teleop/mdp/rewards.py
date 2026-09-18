from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_error_magnitude,
)

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


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize physical torso/root tilt while leaving yaw free."""
  asset: Entity = env.scene[asset_cfg.name]

  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    body_quat_w = body_quat_w.squeeze(1)
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

  return torch.sum(
    torch.square(asset.data.projected_gravity_b[:, :2]),
    dim=1,
  )


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
