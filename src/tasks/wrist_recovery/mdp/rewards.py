from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import BimanualWristCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, name: str) -> BimanualWristCommand:
  return cast(BimanualWristCommand, env.command_manager.get_term(name))


def wrist_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.linalg.norm(
    command.desired_wrist_pos_w - command.robot_wrist_pos_w, dim=-1
  ).mean(-1)
  return torch.exp(-error / std)


def wrist_position_error_tanh(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = torch.linalg.norm(
    command.desired_wrist_pos_w - command.robot_wrist_pos_w, dim=-1
  ).mean(-1)
  return 1.0 - torch.tanh(error / std)


def wrist_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  error = quat_error_magnitude(
    command.desired_wrist_quat_w, command.robot_wrist_quat_w
  ).mean(-1)
  return torch.exp(-error / std)


def inter_wrist_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  target = command.desired_wrist_pos_w[:, 0] - command.desired_wrist_pos_w[:, 1]
  actual = command.robot_wrist_pos_w[:, 0] - command.robot_wrist_pos_w[:, 1]
  return torch.exp(-torch.linalg.norm(target - actual, dim=-1) / std)


def wrist_velocity_error_l2(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = _command(env, command_name)
  actual = command.robot.data.body_link_lin_vel_w[:, command.wrist_body_ids]
  return torch.sum(
    torch.square(command.target_lin_vel_w - actual), dim=(-2, -1)
  )


def base_height_l2(
  env: ManagerBasedRlEnv, target_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)


def base_horizontal_velocity_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=-1)


def base_yaw_rate_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_ang_vel_b[:, 2])


def vertical_velocity_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def roll_pitch_velocity_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=-1)


def joint_deviation_l1(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.abs(
      asset.data.joint_pos[:, asset_cfg.joint_ids]
      - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=-1,
  )


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  speed_sq = torch.sum(
    torch.square(asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=-1
  )
  return torch.sum(speed_sq * in_contact, dim=-1)


def quiet_feet_when_task_stable(
  env: ManagerBasedRlEnv,
  command_name: str,
  velocity_command_name: str,
  error_threshold: float,
  base_speed_threshold: float,
  base_ang_speed_threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  command = _command(env, command_name)
  asset: Entity = env.scene[asset_cfg.name]
  wrist_error = torch.linalg.norm(
    command.desired_wrist_pos_w - command.robot_wrist_pos_w, dim=-1
  ).mean(-1)
  base_command = torch.linalg.norm(
    env.command_manager.get_command(velocity_command_name), dim=-1
  )
  foot_speed_sq = torch.sum(
    torch.square(asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]),
    dim=(-2, -1),
  )
  stable = (
    (wrist_error < error_threshold)
    & (base_command < 0.08)
    & (command.scenario < 0.5)
    & (
      torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=-1)
      < base_speed_threshold
    )
    & (
      torch.linalg.norm(asset.data.root_link_ang_vel_b, dim=-1)
      < base_ang_speed_threshold
    )
  )
  return foot_speed_sq * stable.float()


def undesired_contacts(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  if sensor.data.force_history is not None:
    magnitude = torch.linalg.norm(sensor.data.force_history, dim=-1)
    return (magnitude > force_threshold).any(dim=-1).sum(dim=-1).float()
  assert sensor.data.found is not None
  return sensor.data.found.any(dim=-1).float()
