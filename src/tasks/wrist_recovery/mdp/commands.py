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
  quat_from_angle_axis,
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
    if not 0.0 <= cfg.reach_probability <= 1.0:
      raise ValueError("reach_probability must be in [0, 1].")
    if not 0.0 <= cfg.asymmetric_probability <= 1.0:
      raise ValueError("asymmetric_probability must be in [0, 1].")
    if not 0.0 <= cfg.height_probability <= 1.0:
      raise ValueError("height_probability must be in [0, 1].")
    for name, bounds in (
      ("shoulder_height_range", cfg.shoulder_height_range),
      ("wrist_height_offset_range", cfg.wrist_height_offset_range),
      ("low_reach_extension_range", cfg.low_reach_extension_range),
    ):
      if bounds[0] > bounds[1]:
        raise ValueError(f"{name} lower bound exceeds upper bound: {bounds}")
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
    shoulder_ids, shoulder_names = self.robot.find_bodies(
      cfg.shoulder_body_names, preserve_order=True
    )
    if len(shoulder_ids) != 2:
      raise ValueError(f"Expected two shoulder bodies, found {shoulder_names}.")
    self.shoulder_body_ids = shoulder_ids
    torso_ids, torso_names = self.robot.find_bodies((cfg.torso_body_name,))
    if len(torso_ids) != 1:
      raise ValueError(f"Expected one torso body, found {torso_names}.")
    self.torso_body_id = torso_ids[0]

    shape = (self.num_envs, 2, 3)
    self.start_pos_w = torch.zeros(shape, device=self.device)
    self.target_pos_w = torch.zeros(shape, device=self.device)
    self.previous_target_pos_w = torch.zeros(shape, device=self.device)
    self.target_lin_vel_w = torch.zeros(shape, device=self.device)
    self.target_quat_w = torch.zeros(self.num_envs, 2, 4, device=self.device)
    self.target_quat_w[..., 0] = 1.0
    self.sampled_offset_b = torch.zeros(shape, device=self.device)
    self.sampled_axis_angle_b = torch.zeros(shape, device=self.device)
    self.scenario = torch.zeros(self.num_envs, device=self.device)
    self.is_asymmetric = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.height_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.height_difficulty = torch.zeros(self.num_envs, device=self.device)
    self.sampled_shoulder_height = torch.zeros(
      self.num_envs, device=self.device
    )
    self.sampled_wrist_height_offset = torch.zeros(
      self.num_envs, device=self.device
    )
    self.start_shoulder_height = torch.zeros(self.num_envs, device=self.device)
    self.target_shoulder_height = torch.zeros(self.num_envs, device=self.device)
    self.final_shoulder_height = torch.zeros(self.num_envs, device=self.device)
    self.extension = torch.zeros(self.num_envs, device=self.device)
    self.elapsed = torch.zeros(self.num_envs, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.needs_initialization = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.scripted = False
    self.final_pos_w = torch.zeros(shape, device=self.device)
    self.start_quat_w = self.target_quat_w.clone()
    self.final_quat_w = self.target_quat_w.clone()
    self.transport_reference_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.transport_reference_yaw = torch.zeros(self.num_envs, device=self.device)
    self.scripted_delay_s = cfg.reach_delay_s
    self.scripted_duration_s = cfg.reach_duration_s

    for key in (
      "wrist_pos_error_mean",
      "wrist_pos_error_peak",
      "wrist_rot_error_mean",
      "inter_wrist_error",
      "foot_stagger",
      "asymmetric_fraction",
      "height_command_fraction",
      "height_wrist_pos_error_masked",
      "nonheight_wrist_pos_error_masked",
      "shoulder_height_error",
      "height_shoulder_error_masked",
      "height_target_vertical_gap_masked",
      "height_final_shoulder_target_masked",
      "height_final_wrist_target_masked",
      "shoulder_height_difference",
      "torso_backward_lean",
      "torso_forward_bend",
      "shoulder_height",
      "target_shoulder_height",
      "target_wrist_height",
    ):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
    if cfg.clutch_enabled:
      self.moving_statistics = torch.zeros(self.num_envs, 8, device=self.device)
      for mode in ("balance", "transport", "adjust"):
        for suffix in ("fraction", "wrist_error_masked", "shoulder_error_masked",
                       "velocity_xy_error_masked", "velocity_yaw_error_masked",
                       "moving_fraction", "moving_velocity_xy_error_masked",
                       "moving_velocity_yaw_error_masked", "moving_wrist_error_masked",
                       "moving_shoulder_error_masked", "moving_command_xy_masked",
                       "moving_command_yaw_masked"):
          self.metrics[f"{mode}_{suffix}"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def robot_wrist_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.wrist_body_ids]

  @property
  def robot_wrist_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.wrist_body_ids]

  @property
  def shoulder_heights_w(self) -> torch.Tensor:
    # These body origins coincide with the shoulder joint anchors, so rotating
    # an arm does not move its shoulder-height reference point.
    return self.robot.data.body_link_pos_w[:, self.shoulder_body_ids, 2]

  @property
  def shoulder_height(self) -> torch.Tensor:
    return self.shoulder_heights_w.mean(-1)

  @property
  def shoulder_height_difference(self) -> torch.Tensor:
    heights = self.shoulder_heights_w
    return torch.abs(heights[:, 0] - heights[:, 1])

  @property
  def torso_forward_axis_z(self) -> torch.Tensor:
    torso_quat = self.robot.data.body_link_quat_w[:, self.torso_body_id]
    forward = torch.zeros(self.num_envs, 3, device=self.device)
    forward[:, 0] = 1.0
    return quat_apply(torso_quat, forward)[:, 2]

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
    task = torch.stack(
      (
        self.scenario,
        self.phase,
        self.extension,
        self.target_shoulder_height,
        self.target_shoulder_height - self.shoulder_height,
        self.height_active.float(),
      ),
      dim=-1,
    )
    command = torch.cat((pos_b.flatten(1), rot6d_b, task), dim=-1)
    if self.cfg.clutch_enabled:
      twist = self._env.command_manager.get_term("twist")
      command = torch.cat((command, (twist.mode == 1).float()[:, None]), dim=-1)
    return command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.scripted:
      return
    if self.cfg.clutch_enabled:
      self.moving_statistics[env_ids] = 0
    curriculum = (
      self._env.common_step_counter - self.cfg.curriculum_warmup_steps
    ) / self.cfg.curriculum_ramp_steps
    curriculum = float(max(0.0, min(1.0, curriculum)))
    reach_probability = self.cfg.reach_probability * curriculum
    reach_active = (
      torch.rand(len(env_ids), device=self.device) < reach_probability
    )
    self.scenario[env_ids] = reach_active.float()
    # Advanced indexing returns a copy: sample first, then assign back.
    self.extension[env_ids] = torch.empty(
      len(env_ids), device=self.device
    ).uniform_(*self.cfg.extension_range)
    self.extension[env_ids] *= self.scenario[env_ids] * curriculum
    offsets = torch.zeros(len(env_ids), 2, 3, device=self.device)
    offsets[..., 0] = self.extension[env_ids, None]
    axis_angles = torch.zeros_like(offsets)

    height_probability = self.cfg.height_probability * curriculum
    height_active = (
      torch.rand(len(env_ids), device=self.device) < height_probability
    )
    self.height_active[env_ids] = height_active
    self.height_difficulty[env_ids] = curriculum
    self.sampled_shoulder_height[env_ids] = torch.empty(
      len(env_ids), device=self.device
    ).uniform_(*self.cfg.shoulder_height_range)
    self.sampled_wrist_height_offset[env_ids] = torch.empty(
      len(env_ids), device=self.device
    ).uniform_(*self.cfg.wrist_height_offset_range)
    num_height = int(height_active.sum().item())
    if num_height > 0:
      height_extension = torch.empty(num_height, device=self.device)
      height_extension.uniform_(*self.cfg.low_reach_extension_range)
      offsets[height_active, :, 0] = height_extension[:, None] * curriculum
    self.scenario[env_ids] = torch.maximum(
      self.scenario[env_ids], height_active.float()
    )

    asymmetric = (
      torch.rand(len(env_ids), device=self.device)
      < self.cfg.asymmetric_probability
    ) & reach_active & ~height_active
    num_asymmetric = int(asymmetric.sum().item())
    if num_asymmetric > 0:
      asymmetric_offsets = torch.empty(
        num_asymmetric, 2, 3, device=self.device
      )
      asymmetric_offsets[..., 0].uniform_(*self.cfg.extension_range)
      asymmetric_offsets[..., 1].uniform_(*self.cfg.lateral_offset_range)
      asymmetric_offsets[..., 2].uniform_(*self.cfg.vertical_offset_range)
      offsets[asymmetric] = asymmetric_offsets * curriculum

      random_axes = torch.randn(num_asymmetric, 2, 3, device=self.device)
      random_axes /= torch.linalg.norm(
        random_axes, dim=-1, keepdim=True
      ).clamp_min(1.0e-6)
      angles = torch.empty(num_asymmetric, 2, 1, device=self.device)
      angles.uniform_(*self.cfg.orientation_angle_range)
      axis_angles[asymmetric] = random_axes * angles * curriculum
    self.sampled_offset_b[env_ids] = offsets
    self.sampled_axis_angle_b[env_ids] = axis_angles
    self.is_asymmetric[env_ids] = asymmetric
    self.extension[env_ids] = torch.linalg.norm(offsets, dim=-1).mean(-1)
    self.elapsed[env_ids] = 0.0
    self.phase[env_ids] = 0.0
    self.target_lin_vel_w[env_ids] = 0.0
    self.needs_initialization[env_ids] = True

  def _update_command(self) -> None:
    if hasattr(self._env, "_teacher_push_delta_w"):
      self._env._teacher_push_delta_w.zero_()

    pending = self.needs_initialization.nonzero().flatten()
    if len(pending) > 0:
      if self.cfg.clutch_enabled:
        self.transport_reference_pos[pending] = self.robot.data.root_link_pos_w[pending]
        forward = torch.zeros(len(pending), 3, device=self.device)
        forward[:, 0] = 1
        axis = quat_apply(self.robot.data.root_link_quat_w[pending], forward)
        self.transport_reference_yaw[pending] = torch.atan2(axis[:, 1], axis[:, 0])
      wrist_pos = self.robot_wrist_pos_w[pending]
      self.start_pos_w[pending] = wrist_pos
      self.target_pos_w[pending] = wrist_pos
      self.previous_target_pos_w[pending] = wrist_pos
      wrist_quat = self.robot_wrist_quat_w[pending]
      self.target_quat_w[pending] = wrist_quat
      self.start_quat_w[pending] = wrist_quat
      root_quat = self.robot.data.root_link_quat_w[pending, None, :].expand(
        -1, 2, -1
      )
      offset_w = quat_apply(root_quat, self.sampled_offset_b[pending])
      self.final_pos_w[pending] = wrist_pos + offset_w
      shoulder_height = self.shoulder_height[pending]
      height_pending = self.height_active[pending]
      if height_pending.any():
        height_ids = pending[height_pending]
        # Translate the wrists by the commanded shoulder-height change instead
        # of sampling an unrelated absolute wrist height. This preserves the
        # initially feasible shoulder--wrist vertical separation; the small
        # offset remains available for grasp-height variation.
        shoulder_delta = self.height_difficulty[height_ids] * (
          self.sampled_shoulder_height[height_ids]
          - shoulder_height[height_pending]
        )
        self.final_pos_w[height_ids, :, 2] = (
          wrist_pos[height_pending, :, 2]
          + shoulder_delta[:, None]
          + self.height_difficulty[height_ids, None]
          * self.sampled_wrist_height_offset[height_ids, None]
        )
      self.start_shoulder_height[pending] = shoulder_height
      self.target_shoulder_height[pending] = shoulder_height
      self.final_shoulder_height[pending] = torch.where(
        height_pending,
        shoulder_height
        + self.height_difficulty[pending]
        * (self.sampled_shoulder_height[pending] - shoulder_height),
        shoulder_height,
      )
      aa_w = quat_apply(root_quat, self.sampled_axis_angle_b[pending])
      angle = torch.linalg.norm(aa_w, dim=-1)
      axis = aa_w / angle[..., None].clamp_min(1.0e-6)
      final_quat = quat_mul(quat_from_angle_axis(angle, axis), wrist_quat)
      self.final_quat_w[pending] = torch.where(
        (angle > 1.0e-6)[..., None], final_quat, wrist_quat
      )
      self.elapsed[pending] = 0.0
      self.needs_initialization[pending] = False

    if self.cfg.clutch_enabled and not self.scripted:
      self._advance_transport_reference()
    self.elapsed += self._env.step_dt
    raw_phase = (
      (self.elapsed - self.scripted_delay_s) / self.scripted_duration_s
      if self.scripted
      else (self.elapsed - self.cfg.reach_delay_s) / self.cfg.reach_duration_s
    ).clamp(0.0, 1.0)
    self.phase = (
      raw_phase * raw_phase * (3.0 - 2.0 * raw_phase) * self.scenario
    )
    self.previous_target_pos_w.copy_(self.target_pos_w)
    blend = self.phase[:, None, None]
    self.target_pos_w.copy_(
      self.start_pos_w + blend * (self.final_pos_w - self.start_pos_w)
    )
    final_quat = torch.where(
      (self.start_quat_w * self.final_quat_w).sum(-1, keepdim=True) < 0,
      -self.final_quat_w,
      self.final_quat_w,
    )
    interpolated = (1.0 - blend) * self.start_quat_w + blend * final_quat
    self.target_quat_w.copy_(
      interpolated
      / torch.linalg.norm(interpolated, dim=-1, keepdim=True).clamp_min(1e-6)
    )
    self.target_lin_vel_w.copy_(
      (self.target_pos_w - self.previous_target_pos_w) / self._env.step_dt
    )
    height_blend = raw_phase * raw_phase * (3.0 - 2.0 * raw_phase)
    height_blend *= self.height_active.float()
    self.target_shoulder_height.copy_(
      self.start_shoulder_height
      + height_blend * (self.final_shoulder_height - self.start_shoulder_height)
    )

  def _advance_transport_reference(self) -> None:
    twist = self._env.command_manager.get_term("twist")
    if not twist.cfg.clutch_enabled:
      raise ValueError("Both wrist and twist clutch_enabled must be true")
    dt = self._env.step_dt
    cmd = twist.command * (twist.mode == 1)[:, None]
    yaw = self.transport_reference_yaw
    c, s = torch.cos(yaw), torch.sin(yaw)
    delta = torch.zeros_like(self.transport_reference_pos)
    delta[:, 0] = (c * cmd[:, 0] - s * cmd[:, 1]) * dt
    delta[:, 1] = (s * cmd[:, 0] + c * cmd[:, 1]) * dt
    angle = cmd[:, 2] * dt
    axis = torch.zeros_like(delta)
    axis[:, 2] = 1
    rotation = quat_from_angle_axis(angle, axis)[:, None, :].expand(-1, 2, -1)
    center = self.transport_reference_pos[:, None, :]
    for positions in (self.start_pos_w, self.final_pos_w):
      positions.copy_(center + quat_apply(rotation, positions - center) + delta[:, None, :])
    for orientations in (self.start_quat_w, self.final_quat_w):
      orientations.copy_(quat_mul(rotation, orientations))
    self.transport_reference_pos += delta
    self.transport_reference_yaw += angle

  def set_scripted_trajectory(
    self,
    position_offsets_b: torch.Tensor,
    orientation_axis_angle_b: torch.Tensor | None = None,
    *,
    delay_s: float = 1.0,
    duration_s: float = 2.0,
    scenario_code: float = 1.0,
  ) -> None:
    """Install deterministic base-frame targets for held-out evaluation."""
    expected = (self.num_envs, 2, 3)
    if position_offsets_b.shape != expected:
      raise ValueError(
        f"Expected position offsets shaped {expected}, got {position_offsets_b.shape}."
      )
    start_pos = self.robot_wrist_pos_w.clone()
    start_quat = self.robot_wrist_quat_w.clone()
    root_quat = self.robot.data.root_link_quat_w[:, None, :].expand(-1, 2, -1)
    offset_w = quat_apply(root_quat, position_offsets_b.to(self.device))

    self.scripted = True
    self.scripted_delay_s = delay_s
    self.scripted_duration_s = duration_s
    self.start_pos_w.copy_(start_pos)
    self.target_pos_w.copy_(start_pos)
    self.previous_target_pos_w.copy_(start_pos)
    self.target_quat_w.copy_(start_quat)
    self.start_quat_w.copy_(start_quat)
    self.final_pos_w.copy_(start_pos + offset_w)
    self.final_quat_w.copy_(start_quat)
    if orientation_axis_angle_b is not None:
      aa_w = quat_apply(root_quat, orientation_axis_angle_b.to(self.device))
      angle = torch.linalg.norm(aa_w, dim=-1)
      axis = aa_w / angle[..., None].clamp_min(1e-6)
      delta = quat_from_angle_axis(angle, axis)
      final = quat_mul(delta, start_quat)
      self.final_quat_w.copy_(
        torch.where((angle > 1e-6)[..., None], final, start_quat)
      )
    self.scenario.fill_(scenario_code)
    self.height_active.zero_()
    self.height_difficulty.zero_()
    shoulder_height = self.shoulder_height
    self.start_shoulder_height.copy_(shoulder_height)
    self.target_shoulder_height.copy_(shoulder_height)
    self.final_shoulder_height.copy_(shoulder_height)
    self.is_asymmetric.copy_(
      (position_offsets_b[:, 0] - position_offsets_b[:, 1])
      .abs()
      .amax(dim=-1)
      > 1.0e-6
    )
    self.extension.copy_(torch.linalg.norm(position_offsets_b, dim=-1).mean(-1))
    self.elapsed.zero_()
    self.phase.zero_()
    self.target_lin_vel_w.zero_()
    self.needs_initialization.zero_()

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
    self.metrics["asymmetric_fraction"] = self.is_asymmetric.float()
    self.metrics["height_command_fraction"] = self.height_active.float()
    height_mask = self.height_active.float()
    nonheight_mask = (~self.height_active).float()
    mean_pos_error = pos_error.mean(-1)
    # These are numerators, not conditional means. Divide their logged mean
    # by the matching logged command fraction (see summarize_wrist_training.py).
    self.metrics["height_wrist_pos_error_masked"] = mean_pos_error * height_mask
    self.metrics["nonheight_wrist_pos_error_masked"] = (
      mean_pos_error * nonheight_mask
    )
    self.metrics["shoulder_height_error"] = torch.abs(
      self.target_shoulder_height - self.shoulder_height
    )
    self.metrics["height_shoulder_error_masked"] = (
      self.metrics["shoulder_height_error"] * height_mask
    )
    self.metrics["height_target_vertical_gap_masked"] = (
      torch.abs(
        self.target_shoulder_height
        - self.desired_wrist_pos_w[..., 2].mean(-1)
      )
      * height_mask
    )
    self.metrics["height_final_shoulder_target_masked"] = (
      self.final_shoulder_height * height_mask
    )
    self.metrics["height_final_wrist_target_masked"] = (
      self.final_pos_w[..., 2].mean(-1) * height_mask
    )
    self.metrics["shoulder_height_difference"] = self.shoulder_height_difference
    self.metrics["torso_backward_lean"] = torch.clamp_min(
      self.torso_forward_axis_z, 0.0
    )
    self.metrics["torso_forward_bend"] = torch.clamp_min(
      -self.torso_forward_axis_z, 0.0
    )
    self.metrics["shoulder_height"] = self.shoulder_height
    self.metrics["target_shoulder_height"] = self.target_shoulder_height
    self.metrics["target_wrist_height"] = self.desired_wrist_pos_w[..., 2].mean(-1)
    if self.cfg.clutch_enabled:
      twist = self._env.command_manager.get_term("twist")
      xy_error = torch.linalg.vector_norm(twist.command[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1)
      yaw_error = (twist.command[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]).abs()
      moving = torch.linalg.vector_norm(twist.command, dim=-1) > 1e-5
      # Accumulate moving-only diagnostics over the episode: adjustment has
      # already stopped at most terminal/reset snapshots, so instantaneous
      # moving masks would lose precisely the cases we need to inspect.
      self.moving_statistics[:, 0] += 1
      self.moving_statistics[:, 1] += moving.float()
      self.moving_statistics[:, 2] += xy_error * moving
      self.moving_statistics[:, 3] += yaw_error * moving
      self.moving_statistics[:, 4] += mean_pos_error * moving
      self.moving_statistics[:, 5] += self.metrics["shoulder_height_error"] * moving
      self.moving_statistics[:, 6] += torch.linalg.vector_norm(twist.command[:, :2], dim=-1) * moving
      self.moving_statistics[:, 7] += twist.command[:, 2].abs() * moving
      denominator = self.moving_statistics[:, 0].clamp_min(1)
      for index, name in enumerate(("balance", "transport", "adjust")):
        mask = (twist.mode == index).float()
        active = mask * self.moving_statistics[:, 1] / denominator
        self.metrics[f"{name}_fraction"] = mask
        self.metrics[f"{name}_wrist_error_masked"] = mean_pos_error * mask
        self.metrics[f"{name}_shoulder_error_masked"] = self.metrics["shoulder_height_error"] * mask
        self.metrics[f"{name}_velocity_xy_error_masked"] = xy_error * mask
        self.metrics[f"{name}_velocity_yaw_error_masked"] = yaw_error * mask
        self.metrics[f"{name}_moving_fraction"] = active
        self.metrics[f"{name}_moving_velocity_xy_error_masked"] = mask * self.moving_statistics[:, 2] / denominator
        self.metrics[f"{name}_moving_velocity_yaw_error_masked"] = mask * self.moving_statistics[:, 3] / denominator
        for column, suffix in ((4,"wrist_error"), (5,"shoulder_error"), (6,"command_xy"), (7,"command_yaw")):
          self.metrics[f"{name}_moving_{suffix}_masked"] = mask * self.moving_statistics[:, column] / denominator


@dataclass(kw_only=True)
class BimanualWristCommandCfg(CommandTermCfg):
  entity_name: str
  wrist_body_names: tuple[str, str]
  foot_body_names: tuple[str, str]
  shoulder_body_names: tuple[str, str]
  torso_body_name: str
  reach_probability: float = 0.5
  asymmetric_probability: float = 0.0
  extension_range: tuple[float, float] = (0.12, 0.28)
  lateral_offset_range: tuple[float, float] = (-0.10, 0.10)
  vertical_offset_range: tuple[float, float] = (-0.06, 0.06)
  orientation_angle_range: tuple[float, float] = (-0.25, 0.25)
  height_probability: float = 0.0
  clutch_enabled: bool = False
  shoulder_height_range: tuple[float, float] = (0.72, 1.02)
  wrist_height_offset_range: tuple[float, float] = (-0.02, 0.02)
  low_reach_extension_range: tuple[float, float] = (0.08, 0.22)
  reach_delay_s: float = 1.0
  reach_duration_s: float = 2.0
  curriculum_warmup_steps: int = 30_000
  curriculum_ramp_steps: int = 60_000

  def build(self, env: ManagerBasedRlEnv) -> BimanualWristCommand:
    return BimanualWristCommand(self, env)
