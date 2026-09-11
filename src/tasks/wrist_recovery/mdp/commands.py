from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class BimanualWristCommand(CommandTerm):
  """World-fixed wrist targets with an optional smooth forward reach."""

  cfg: BimanualWristCommandCfg

  def __init__(self, cfg: BimanualWristCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    wrist_ids, names = self.robot.find_bodies(
      cfg.wrist_body_names, preserve_order=True
    )
    if len(wrist_ids) != 2:
      raise ValueError(f"Expected two wrist bodies, found {names}.")
    self.wrist_body_ids = wrist_ids
    foot_ids, foot_names = self.robot.find_bodies(
      cfg.foot_body_names, preserve_order=True
    )
    if len(foot_ids) != 2:
      raise ValueError(f"Expected two foot bodies, found {foot_names}.")
    self.foot_body_ids = foot_ids

    shape = (self.num_envs, 2, 3)
    self.start_pos_w = torch.zeros(shape, device=self.device)
    self.target_pos_w = torch.zeros(shape, device=self.device)
    self.previous_target_pos_w = torch.zeros(shape, device=self.device)
    self.target_lin_vel_w = torch.zeros(shape, device=self.device)
    self.target_quat_w = torch.zeros(self.num_envs, 2, 4, device=self.device)
    self.target_quat_w[..., 0] = 1.0
    self.forward_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.scenario = torch.zeros(self.num_envs, device=self.device)
    self.extension = torch.zeros(self.num_envs, device=self.device)
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.needs_initialization = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    for key in (
      "wrist_pos_error_mean",
      "wrist_pos_error_peak",
      "wrist_rot_error_mean",
      "inter_wrist_error",
      "foot_stagger",
    ):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

  @property
  def robot_wrist_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.wrist_body_ids]

  @property
  def robot_wrist_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.wrist_body_ids]

  @property
  def desired_wrist_pos_w(self) -> torch.Tensor:
    return torch.where(
      self.needs_initialization[:, None, None],
      self.robot_wrist_pos_w,
      self.target_pos_w,
    )

  @property
  def desired_wrist_quat_w(self) -> torch.Tensor:
    return torch.where(
      self.needs_initialization[:, None, None],
      self.robot_wrist_quat_w,
      self.target_quat_w,
    )

  @property
  def target_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
    root_pos = self.robot.data.root_link_pos_w[:, None, :].expand(-1, 2, -1)
    root_quat = self.robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
    return subtract_frame_transforms(
      root_pos,
      root_quat,
      self.desired_wrist_pos_w,
      self.desired_wrist_quat_w,
    )

  @property
  def wrist_position_error_b(self) -> torch.Tensor:
    error_w = self.desired_wrist_pos_w - self.robot_wrist_pos_w
    root_quat = self.robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
    return quat_apply_inverse(root_quat, error_w)

  @property
  def wrist_orientation_error_b(self) -> torch.Tensor:
    error_q_w = quat_mul(
      self.desired_wrist_quat_w, quat_inv(self.robot_wrist_quat_w)
    )
    error_aa_w = axis_angle_from_quat(error_q_w)
    root_quat = self.robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
    return quat_apply_inverse(root_quat, error_aa_w)

  @property
  def command(self) -> torch.Tensor:
    pos_b, quat_b = self.target_pose_b
    rot6d_b = matrix_from_quat(quat_b)[..., :2].reshape(self.num_envs, -1)
    task = torch.stack((self.scenario, self.phase, self.extension), dim=-1)
    return torch.cat((pos_b.flatten(1), rot6d_b, task), dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    curriculum = (
      self._env.common_step_counter - self.cfg.curriculum_warmup_steps
    ) / self.cfg.curriculum_ramp_steps
    curriculum = float(max(0.0, min(1.0, curriculum)))
    reach_probability = self.cfg.reach_probability * curriculum
    self.scenario[env_ids] = (
      torch.rand(len(env_ids), device=self.device) < reach_probability
    ).float()
    self.extension[env_ids].uniform_(*self.cfg.extension_range)
    self.extension[env_ids] *= self.scenario[env_ids] * curriculum
    self.elapsed[env_ids] = 0.0
    self.phase[env_ids] = 0.0
    self.target_lin_vel_w[env_ids] = 0.0
    self.needs_initialization[env_ids] = True

  def _update_command(self) -> None:
    if hasattr(self._env, "_teacher_push_delta_w"):
      self._env._teacher_push_delta_w.zero_()

    pending = self.needs_initialization.nonzero().flatten()
    if len(pending) > 0:
      wrist_pos = self.robot_wrist_pos_w[pending]
      self.start_pos_w[pending] = wrist_pos
      self.target_pos_w[pending] = wrist_pos
      self.previous_target_pos_w[pending] = wrist_pos
      self.target_quat_w[pending] = self.robot_wrist_quat_w[pending]
      forward_b = torch.zeros(len(pending), 3, device=self.device)
      forward_b[:, 0] = 1.0
      self.forward_w[pending] = quat_apply(
        self.robot.data.root_link_quat_w[pending], forward_b
      )
      self.forward_w[pending, 2] = 0.0
      self.forward_w[pending] /= torch.linalg.norm(
        self.forward_w[pending], dim=-1, keepdim=True
      ).clamp_min(1.0e-6)
      self.elapsed[pending] = 0.0
      self.needs_initialization[pending] = False

    self.elapsed += self._env.step_dt
    raw_phase = (
      (self.elapsed - self.cfg.reach_delay_s) / self.cfg.reach_duration_s
    ).clamp(0.0, 1.0)
    self.phase = (
      raw_phase * raw_phase * (3.0 - 2.0 * raw_phase) * self.scenario
    )
    self.previous_target_pos_w.copy_(self.target_pos_w)
    offset_w = self.forward_w[:, None, :] * (
      self.extension * self.phase
    )[:, None, None]
    self.target_pos_w.copy_(self.start_pos_w + offset_w)
    self.target_lin_vel_w.copy_(
      (self.target_pos_w - self.previous_target_pos_w) / self._env.step_dt
    )

  def _update_metrics(self) -> None:
    pos_error = torch.linalg.norm(
      self.desired_wrist_pos_w - self.robot_wrist_pos_w, dim=-1
    )
    rot_error = quat_error_magnitude(
      self.desired_wrist_quat_w, self.robot_wrist_quat_w
    )
    self.metrics["wrist_pos_error_mean"] = pos_error.mean(-1)
    self.metrics["wrist_pos_error_peak"] = torch.maximum(
      self.metrics["wrist_pos_error_peak"], pos_error.max(-1).values
    )
    self.metrics["wrist_rot_error_mean"] = rot_error.mean(-1)
    desired_rel = self.desired_wrist_pos_w[:, 0] - self.desired_wrist_pos_w[:, 1]
    actual_rel = self.robot_wrist_pos_w[:, 0] - self.robot_wrist_pos_w[:, 1]
    self.metrics["inter_wrist_error"] = torch.linalg.norm(
      desired_rel - actual_rel, dim=-1
    )
    feet = self.robot.data.body_link_pos_w[:, self.foot_body_ids]
    self.metrics["foot_stagger"] = torch.abs(feet[:, 0, 0] - feet[:, 1, 0])


@dataclass(kw_only=True)
class BimanualWristCommandCfg(CommandTermCfg):
  entity_name: str
  wrist_body_names: tuple[str, str]
  foot_body_names: tuple[str, str]
  reach_probability: float = 0.5
  extension_range: tuple[float, float] = (0.12, 0.28)
  reach_delay_s: float = 1.0
  reach_duration_s: float = 2.0
  curriculum_warmup_steps: int = 30_000
  curriculum_ramp_steps: int = 60_000

  def build(self, env: ManagerBasedRlEnv) -> BimanualWristCommand:
    return BimanualWristCommand(self, env)
