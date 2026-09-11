from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, subtract_frame_transforms

from .commands import BimanualWristCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def wrist_pose_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(BimanualWristCommand, env.command_manager.get_term(command_name))
  return torch.cat(
    (
      command.wrist_position_error_b.flatten(1),
      command.wrist_orientation_error_b.flatten(1),
    ),
    dim=-1,
  )


def wrist_velocity_error(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = cast(BimanualWristCommand, env.command_manager.get_term(command_name))
  current = command.robot.data.body_link_lin_vel_w[:, command.wrist_body_ids]
  error_w = command.target_lin_vel_w - current
  root_quat = command.robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
  return quat_apply_inverse(root_quat, error_w).flatten(1)


def feet_state(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  command = cast(BimanualWristCommand, env.command_manager.get_term(command_name))
  asset: Entity = command.robot
  sensor: ContactSensor = env.scene[sensor_name]
  root_pos = asset.data.root_link_pos_w[:, None, :].expand(-1, 2, -1)
  root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
  foot_pos_b, _ = subtract_frame_transforms(
    root_pos,
    root_quat,
    asset.data.body_link_pos_w[:, command.foot_body_ids],
    asset.data.body_link_quat_w[:, command.foot_body_ids],
  )
  foot_vel_b = quat_apply_inverse(
    root_quat, asset.data.body_link_lin_vel_w[:, command.foot_body_ids]
  )
  assert sensor.data.force is not None
  contact_force_b = quat_apply_inverse(root_quat, sensor.data.force)
  return torch.cat(
    (foot_pos_b.flatten(1), foot_vel_b.flatten(1), contact_force_b.flatten(1)),
    dim=-1,
  )


def teacher_disturbance(env: ManagerBasedRlEnv) -> torch.Tensor:
  wrench = getattr(env, "_teacher_hand_wrench_w", None)
  push = getattr(env, "_teacher_push_delta_w", None)
  if wrench is None:
    wrench = torch.zeros(env.num_envs, 2, 6, device=env.device)
  if push is None:
    push = torch.zeros(env.num_envs, 6, device=env.device)
  return torch.cat((wrench.flatten(1), push), dim=-1)
