from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import (
  SparseWholeBodyCommand,
  _rotation_6d,
  _yaw_quat_tensor,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def sim_task_state_absolute(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Privileged 24-D simulated task state for the teacher actor.

  It mirrors SparseWholeBodyCommand.command exactly, but contains the actual
  MuJoCo task-space state. Positions are expressed in the fixed per-environment
  world (_ew), not relative to the current robot.
  """
  command = cast(
    SparseWholeBodyCommand,
    env.command_manager.get_term(command_name),
  )

  return torch.cat(
    (
      command.sim_left_wrist_pos_ew,                     # 3
      _rotation_6d(command.sim_left_wrist_quat_w),       # 6
      command.sim_right_wrist_pos_ew,                    # 3
      _rotation_6d(command.sim_right_wrist_quat_w),      # 6
      command.sim_shoulder_mid_pos_ew[:, :2],            # 2
      command.sim_left_shoulder_height[:, None],         # 1
      command.sim_right_shoulder_height[:, None],        # 1
      command.sim_shoulder_heading_vec_w,                # 2
    ),
    dim=-1,
  )


def teleop_tracking_errors(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Privileged critic observation: 8 scalar cmd-vs-sim task errors."""
  command = cast(
    SparseWholeBodyCommand,
    env.command_manager.get_term(command_name),
  )

  left_pos = torch.linalg.vector_norm(
    command.cmd_left_wrist_pos_w - command.sim_left_wrist_pos_w,
    dim=-1,
  )
  right_pos = torch.linalg.vector_norm(
    command.cmd_right_wrist_pos_w - command.sim_right_wrist_pos_w,
    dim=-1,
  )
  left_ori = quat_error_magnitude(
    command.cmd_left_wrist_quat_w,
    command.sim_left_wrist_quat_w,
  )
  right_ori = quat_error_magnitude(
    command.cmd_right_wrist_quat_w,
    command.sim_right_wrist_quat_w,
  )
  shoulder_mid_xy = torch.linalg.vector_norm(
    command.cmd_shoulder_mid_xy_w
    - command.sim_shoulder_mid_pos_w[:, :2],
    dim=-1,
  )
  shoulder_heading = quat_error_magnitude(
    _yaw_quat_tensor(command.cmd_shoulder_heading_w),
    command.sim_shoulder_yaw_quat_w,
  )
  left_shoulder = torch.abs(
    command.cmd_left_shoulder_height
    - command.sim_left_shoulder_height
  )
  right_shoulder = torch.abs(
    command.cmd_right_shoulder_height
    - command.sim_right_shoulder_height
  )

  return torch.stack(
    (
      left_pos,
      right_pos,
      left_ori,
      right_ori,
      shoulder_mid_xy,
      shoulder_heading,
      left_shoulder,
      right_shoulder,
    ),
    dim=-1,
  )


def command_wrist_linear_velocity_absolute(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """6-D emitted wrist-target linear velocity in simulator world."""
  command = cast(
    SparseWholeBodyCommand,
    env.command_manager.get_term(command_name),
  )
  return torch.cat(
    (
      command.cmd_left_wrist_lin_vel_w,
      command.cmd_right_wrist_lin_vel_w,
    ),
    dim=-1,
  )


def sim_wrist_linear_velocity_absolute(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Privileged 6-D actual wrist linear velocity in simulator world."""
  command = cast(
    SparseWholeBodyCommand,
    env.command_manager.get_term(command_name),
  )
  return torch.cat(
    (
      command.sim_left_wrist_lin_vel_w,
      command.sim_right_wrist_lin_vel_w,
    ),
    dim=-1,
  )
