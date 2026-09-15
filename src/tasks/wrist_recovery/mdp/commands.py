from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  matrix_from_quat,
  quat_from_matrix,
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
    if not 0 <= cfg.persistent_probability <= 1 or (cfg.persistent_probability > 0 and not cfg.capability_pack):
      raise ValueError("Persistent trajectories require capability_pack and probability in [0,1]")
    if not 0 < cfg.diagnostics_trace_window_steps <= cfg.diagnostics_trace_period_steps:
      raise ValueError("Diagnostic trace window must be positive and no larger than its period")
    if not 0.0 <= cfg.reach_probability <= 1.0:
      raise ValueError("reach_probability must be in [0, 1].")
    if not 0.0 <= cfg.asymmetric_probability <= 1.0:
      raise ValueError("asymmetric_probability must be in [0, 1].")
    if not 0.0 <= cfg.height_probability <= 1.0:
      raise ValueError("height_probability must be in [0, 1].")
    if not 0.0 <= cfg.ground_probability <= 1.0:
      raise ValueError("ground_probability must be in [0, 1].")
    if not 0.0 <= cfg.bilateral_ground_probability <= 1.0:
      raise ValueError("bilateral_ground_probability must be in [0, 1].")
    if not 0 <= cfg.continuous_probability <= 1 or cfg.continuous_period_range[0] <= 0 or cfg.continuous_period_range[0] > cfg.continuous_period_range[1] or cfg.continuous_displacement <= 0:
      raise ValueError("Invalid continuous trajectory probability/period/displacement")
    for name, bounds in (
      ("shoulder_height_range", cfg.shoulder_height_range),
      ("wrist_height_offset_range", cfg.wrist_height_offset_range),
      ("low_reach_extension_range", cfg.low_reach_extension_range),
      ("ground_wrist_height_range", cfg.ground_wrist_height_range),
      ("ground_other_wrist_raise_range", cfg.ground_other_wrist_raise_range),
      ("height_lateral_offset_range", cfg.height_lateral_offset_range),
    ):
      if bounds[0] > bounds[1]:
        raise ValueError(f"{name} lower bound exceeds upper bound: {bounds}")
    self.robot: Entity = env.scene[cfg.entity_name]
    self.leg_joint_ids, leg_names = self.robot.find_joints(
      (r".*_hip_.*_joint", r".*_knee_joint", r".*_ankle_.*_joint")
    )
    if len(self.leg_joint_ids) != 12:
      raise ValueError(f"Expected twelve leg joints, found {leg_names}.")
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
    self.ground_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.bilateral_ground_active = torch.zeros_like(self.ground_active)
    self.continuous_active = torch.zeros_like(self.ground_active)
    self.continuous_delta_w = torch.zeros(self.num_envs, 2, 3, device=self.device)
    self.continuous_period = torch.ones(self.num_envs, 2, device=self.device)
    self.continuous_statistics = torch.zeros(self.num_envs, 8, device=self.device)
    self.pack_delta = torch.zeros(self.num_envs, 4, 2, 3, device=self.device)
    self.pack_quat = torch.zeros(self.num_envs, 4, 2, 4, device=self.device)
    self.pack_duration = torch.ones(self.num_envs, 3, 2, device=self.device)
    self.pack_stage = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
    self.pack_statistics = torch.zeros(self.num_envs, 5, device=self.device)
    self.persistent_active = torch.zeros_like(self.continuous_active)
    self.egodex_active = torch.zeros_like(self.continuous_active)
    self.pack_start_time = torch.zeros(self.num_envs, 2, device=self.device)
    self.pack_continuations = torch.zeros_like(self.pack_start_time)
    self.persistent_statistics = torch.zeros(self.num_envs, 4, device=self.device)
    self.pack_reference_quat = torch.zeros(self.num_envs, 4, device=self.device)
    self.pack_reference_quat[:, 0] = 1
    self.egodex_positions = None
    if cfg.egodex_probability > 0:
      corpus_path = Path(cfg.egodex_data_path)
      if not corpus_path.is_absolute(): corpus_path = Path(__file__).resolve().parents[4] / corpus_path
      with np.load(corpus_path, allow_pickle=False) as corpus:
        positions, orientations = corpus["positions"], corpus["orientations"]
        offsets, lengths = corpus["offsets"], corpus["lengths"]
      if positions.ndim != 3 or positions.shape[1:] != (3, 3) or len(offsets) != len(lengths)+1:
        raise ValueError(f"Invalid EgoDex corpus: {corpus_path}")
      if orientations.shape != (len(positions), 2, 4) or not np.isfinite(orientations).all():
        raise ValueError(f"Invalid EgoDex wrist orientations: {corpus_path}")
      self.egodex_positions = torch.as_tensor(positions, device=self.device)
      self.egodex_orientations = torch.as_tensor(orientations, device=self.device)
      self.egodex_offsets = torch.as_tensor(offsets[:-1], device=self.device)
      self.egodex_lengths = torch.as_tensor(lengths, device=self.device)
      self.egodex_clip = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
      self.egodex_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
      self.egodex_origin_w = torch.zeros(self.num_envs, 3, device=self.device)
      self.egodex_forward_w = torch.zeros(self.num_envs, 3, device=self.device); self.egodex_forward_w[:, 0] = 1
      self.egodex_right_w = torch.zeros(self.num_envs, 3, device=self.device); self.egodex_right_w[:, 1] = -1
    if cfg.diagnostics_enabled:
      from .operation_diagnostics import OperationDiagnostics
      self.operation_diagnostics = OperationDiagnostics(self)
    self.sampled_bilateral_ground_height = torch.zeros(self.num_envs, 2, device=self.device)
    self.ground_side = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.sampled_ground_wrist_height = torch.zeros(self.num_envs, device=self.device)
    self.sampled_other_wrist_raise = torch.zeros(self.num_envs, device=self.device)
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
      "leg_joint_acc_rms_snapshot",
      "leg_joint_vel_rms_snapshot",
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
      self.bilateral_transport_statistics = torch.zeros(self.num_envs, 6, device=self.device)
      for suffix in ("fraction", "wrist_error_masked", "shoulder_error_masked",
                     "command_xy_masked", "projected_speed_masked", "velocity_xy_error_masked"):
        self.metrics[f"bilateral_transport_moving_{suffix}"] = torch.zeros(self.num_envs, device=self.device)
      for mode in ("balance", "transport", "adjust"):
        for suffix in ("fraction", "wrist_error_masked", "shoulder_error_masked",
                       "velocity_xy_error_masked", "velocity_yaw_error_masked",
                       "moving_fraction", "moving_velocity_xy_error_masked",
                       "moving_velocity_yaw_error_masked", "moving_wrist_error_masked",
                       "moving_shoulder_error_masked", "moving_command_xy_masked",
                       "moving_command_yaw_masked"):
          self.metrics[f"{mode}_{suffix}"] = torch.zeros(self.num_envs, device=self.device)
    if cfg.ground_probability > 0:
      for key in ("ground_fraction", "ground_wrist_error_masked", "ground_shoulder_error_masked", "ground_low_target_masked"):
        self.metrics[key] = torch.zeros(self.num_envs,device=self.device)
    if cfg.bilateral_ground_probability > 0:
      for key in ("bilateral_ground_fraction", "bilateral_ground_wrist_error_masked",
                  "bilateral_ground_shoulder_error_masked", "bilateral_ground_target_min_masked",
                  "bilateral_ground_target_max_masked"):
        self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

    if cfg.capability_pack:
      for key in ("pack_shoulder_error_masked", "pack_case_fraction", "pack_peak_wrist_error_masked", "pack_lift_fraction", "pack_place_fraction"):
        self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
      for key in ("persistent_fraction", "persistent_continuations_masked"):
        self.metrics[key] = torch.zeros(self.num_envs, device=self.device)
      for key in ("steady_fraction", "wrist_error_masked", "rotation_error_masked", "shoulder_error_masked"):
        self.metrics['persistent_'+key] = torch.zeros(self.num_envs, device=self.device)
    if cfg.continuous_probability > 0:
      for key in ("continuous_fraction", "continuous_wrist_error_masked", "continuous_rotation_error_masked",
                  "continuous_moving_fraction", "continuous_command_speed_masked",
                  "continuous_projected_speed_masked", "continuous_velocity_error_masked"):
        self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

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

  def _egodex_targets(self, env_ids: torch.Tensor, age_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate mapped [left hand, right hand, shoulder midpoint] samples."""
    frame = self.egodex_start[env_ids].float() + age_s * self.cfg.egodex_source_fps
    length = self.egodex_lengths[self.egodex_clip[env_ids]]
    i0 = frame.floor().long().clamp_min(0).minimum(length - 1)
    i1 = (i0 + 1).minimum(length - 1)
    alpha = (frame - i0.float()).clamp(0, 1)
    base = self.egodex_offsets[self.egodex_clip[env_ids]]
    local = self.egodex_positions[base + i0] * (1-alpha[:, None, None]) + self.egodex_positions[base + i1] * alpha[:, None, None]
    q0, q1 = self.egodex_orientations[base + i0], self.egodex_orientations[base + i1]
    q1 = torch.where((q0*q1).sum(-1, keepdim=True) < 0, -q1, q1)
    local_quat = q0 * (1-alpha[:, None, None]) + q1 * alpha[:, None, None]
    local_quat /= torch.linalg.vector_norm(local_quat, dim=-1, keepdim=True).clamp_min(1e-6)
    world = (self.egodex_origin_w[env_ids, None, :]
             + local[..., 0, None] * self.egodex_forward_w[env_ids, None, :]
             + local[..., 1, None] * self.egodex_right_w[env_ids, None, :])
    world[..., 2] += local[..., 2]
    frame_rotation = torch.stack((self.egodex_forward_w[env_ids], self.egodex_right_w[env_ids],
                                  torch.tensor((0., 0., 1.), device=self.device).expand(len(env_ids), -1)), dim=-1)
    source_quat = quat_from_matrix(frame_rotation[:, None] @ matrix_from_quat(local_quat))
    return world[:, :2], world[:, 2, 2], source_quat

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.scripted:
      return
    if self.cfg.clutch_enabled:
      self.moving_statistics[env_ids] = 0
      self.bilateral_transport_statistics[env_ids] = 0
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
    if self.cfg.ground_probability > 0:
      self.ground_active[env_ids] = height_active & (torch.rand(len(env_ids), device=self.device) < self.cfg.ground_probability)
      self.ground_side[env_ids] = torch.randint(0,2,(len(env_ids),),device=self.device)
      self.sampled_ground_wrist_height[env_ids] = torch.empty(len(env_ids),device=self.device).uniform_(*self.cfg.ground_wrist_height_range)
      self.sampled_other_wrist_raise[env_ids] = torch.empty(len(env_ids),device=self.device).uniform_(*self.cfg.ground_other_wrist_raise_range)
    self.bilateral_ground_active[env_ids] = False
    if self.cfg.bilateral_ground_probability > 0:
      self.bilateral_ground_active[env_ids] = self.ground_active[env_ids] & (
        torch.rand(len(env_ids), device=self.device) < self.cfg.bilateral_ground_probability
      )
      self.sampled_bilateral_ground_height[env_ids] = torch.empty(
        len(env_ids), 2, device=self.device
      ).uniform_(*self.cfg.ground_wrist_height_range)
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
      if self.cfg.height_spatial_sampling:
        offsets[height_active, :, 0] = torch.empty(num_height,2,device=self.device).uniform_(*self.cfg.low_reach_extension_range) * curriculum
        offsets[height_active, :, 1] = torch.empty(num_height,2,device=self.device).uniform_(*self.cfg.height_lateral_offset_range) * curriculum
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
    self.continuous_active[env_ids] = False
    self.continuous_statistics[env_ids] = 0
    self.pack_statistics[env_ids] = 0
    self.pack_start_time[env_ids] = 0
    self.pack_continuations[env_ids] = 0
    self.persistent_statistics[env_ids] = 0
    self.persistent_active[env_ids] = False
    self.egodex_active[env_ids] = False
    if self.cfg.diagnostics_enabled:
      self.operation_diagnostics.reset(env_ids)
    self.continuous_delta_w[env_ids] = 0
    if self.cfg.continuous_probability > 0:
      self.continuous_active[env_ids] = (self.scenario[env_ids] > 0) & (
        torch.rand(len(env_ids), device=self.device) < self.cfg.continuous_probability
      )
      if self.cfg.capability_pack:
        self.persistent_active[env_ids] = self.continuous_active[env_ids] & (
          torch.rand(len(env_ids), device=self.device) < self.cfg.persistent_probability
        )
      self.continuous_period[env_ids] = torch.empty(len(env_ids), 2, device=self.device).uniform_(*self.cfg.continuous_period_range)
      delta = torch.randn(len(env_ids), 2, 2, device=self.device)
      delta = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-6) * self.cfg.continuous_displacement
      candidate = offsets[..., :2] + delta
      for row_mask, x_range, y_range in ((height_active, self.cfg.low_reach_extension_range, self.cfg.height_lateral_offset_range),
                                        (~height_active, self.cfg.extension_range, self.cfg.lateral_offset_range)):
        candidate[row_mask, :, 0] = candidate[row_mask, :, 0].clamp(x_range[0]*curriculum, x_range[1]*curriculum)
        candidate[row_mask, :, 1] = candidate[row_mask, :, 1].clamp(y_range[0]*curriculum, y_range[1]*curriculum)
      delta_b = torch.zeros_like(offsets)
      difference = candidate - offsets[..., :2]
      scale = (self.cfg.continuous_displacement / torch.linalg.vector_norm(difference, dim=-1).clamp_min(1e-6)).clamp(max=1.)
      delta_b[..., :2] = difference * scale[..., None] * self.continuous_active[env_ids, None, None]
      root_q = self.robot.data.root_link_quat_w[env_ids, None, :].expand(-1, 2, -1)
      self.continuous_delta_w[env_ids] = quat_apply(root_q, delta_b)
      self.continuous_delta_w[env_ids, :, 2] = 0  # Keep both low heights unchanged.
    self.sampled_axis_angle_b[env_ids] = axis_angles
    self.is_asymmetric[env_ids] = asymmetric
    self.extension[env_ids] = torch.linalg.norm(offsets, dim=-1).mean(-1)
    self.elapsed[env_ids] = 0.0
    self.phase[env_ids] = 0.0
    self.target_lin_vel_w[env_ids] = 0.0
    if self.egodex_positions is not None:
      ego = torch.rand(len(env_ids), device=self.device) < self.cfg.egodex_probability
      ego_ids = env_ids[ego]
      if len(ego_ids):
        self.egodex_active[ego_ids] = True
        self.egodex_clip[ego_ids] = torch.randint(len(self.egodex_lengths), (len(ego_ids),), device=self.device)
        lengths = self.egodex_lengths[self.egodex_clip[ego_ids]]
        # Random subclips retain native source speed; clamp at endpoint rather than wrap.
        self.egodex_start[ego_ids] = (torch.rand(len(ego_ids), device=self.device) * lengths.float()).long()
        self.scenario[ego_ids] = 1
        self.height_active[ego_ids] = True
        self.ground_active[ego_ids] = False
        self.bilateral_ground_active[ego_ids] = False
        self.continuous_active[ego_ids] = True
        self.persistent_active[ego_ids] = False
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
      ego_ids = pending[self.egodex_active[pending]]
      if len(ego_ids):
        shoulder_xy = self.robot.data.body_link_pos_w[ego_ids][:, self.shoulder_body_ids, :2].mean(1)
        self.egodex_origin_w[ego_ids] = 0
        self.egodex_origin_w[ego_ids, :2] = shoulder_xy
        local_forward = torch.zeros(len(ego_ids), 3, device=self.device)
        local_forward[:, 0] = 1
        forward = quat_apply(self.robot.data.root_link_quat_w[ego_ids], local_forward)
        forward[:, 2] = 0
        forward /= torch.linalg.vector_norm(forward, dim=-1, keepdim=True).clamp_min(1e-6)
        self.egodex_forward_w[ego_ids] = forward
        self.egodex_right_w[ego_ids] = torch.stack((forward[:, 1], -forward[:, 0], torch.zeros_like(forward[:, 0])), -1)
        ego_pos, ego_shoulder, ego_quat = self._egodex_targets(ego_ids, torch.zeros(len(ego_ids), device=self.device))
        self.final_pos_w[ego_ids] = ego_pos
        self.final_shoulder_height[ego_ids] = ego_shoulder
        # Unlike the earlier relative-orientation prototype, this is an
        # absolute target in the fixed shoulder-ground frame.  The initial
        # reach interpolates from the actual wrist pose to the first command;
        # subsequent clip samples are never reset or re-aligned to the robot.
        self.final_quat_w[ego_ids] = ego_quat
      if self.cfg.ground_probability > 0:
        ground = pending[self.ground_active[pending]]
        if len(ground):
          # Derive shoulder height from the selected wrist's initial vertical
          # gap, rather than sampling two independent absolute heights.
          side = self.ground_side[ground]
          rows = torch.arange(len(ground), device=self.device)
          initial = self.start_pos_w[ground]
          initial_height = initial[rows,side,2]
          residual = self.sampled_wrist_height_offset[ground]
          desired_shoulder = self.sampled_ground_wrist_height[ground] + self.start_shoulder_height[ground] - initial_height - residual
          delta = self.height_difficulty[ground] * (desired_shoulder-self.start_shoulder_height[ground])
          self.final_shoulder_height[ground] = self.start_shoulder_height[ground] + delta
          self.final_pos_w[ground,:,2] = initial[:,:,2] + delta[:,None] + self.height_difficulty[ground,None]*residual[:,None]
          # The other hand remains above the low hand, but is not forced to
          # stay at its initial standing height (which can be unreachable).
          other = 1-side
          self.final_pos_w[ground,other,2] += self.height_difficulty[ground]*self.sampled_other_wrist_raise[ground]
          self.is_asymmetric[ground] = True
          bilateral = ground[self.bilateral_ground_active[ground]]
          if len(bilateral):
            initial = self.start_pos_w[bilateral]
            target = self.sampled_bilateral_ground_height[bilateral]
            # Shared shoulder height accommodates both initial vertical gaps;
            # this coupling is NOT a full-pose IK feasibility proof.
            residual = self.sampled_wrist_height_offset[bilateral]
            shoulder_candidates = target + self.start_shoulder_height[bilateral, None] - initial[..., 2] - residual[:, None]
            difficulty = self.height_difficulty[bilateral]
            self.final_shoulder_height[bilateral] = self.start_shoulder_height[bilateral] + difficulty * (
              shoulder_candidates.min(-1).values - self.start_shoulder_height[bilateral]
            )
            self.final_pos_w[bilateral, :, 2] = initial[..., 2] + difficulty[:, None] * (target - initial[..., 2])
      aa_w = quat_apply(root_quat, self.sampled_axis_angle_b[pending])
      angle = torch.linalg.norm(aa_w, dim=-1)
      axis = aa_w / angle[..., None].clamp_min(1.0e-6)
      final_quat = quat_mul(quat_from_angle_axis(angle, axis), wrist_quat)
      self.final_quat_w[pending] = torch.where(
        (angle > 1.0e-6)[..., None], final_quat, wrist_quat
      )
      self.elapsed[pending] = 0.0
      if self.cfg.capability_pack:
        from .operation_clip import sample_clip
        self.pack_reference_quat[pending] = self.robot.data.root_link_quat_w[pending]
        self.pack_delta[pending], self.pack_quat[pending], self.pack_duration[pending] = sample_clip(
          self.sampled_offset_b[pending], self.robot.data.root_link_quat_w[pending], self.final_quat_w[pending],
          self.height_active[pending], self.height_difficulty[pending], self.cfg
        )
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
    if self.cfg.continuous_probability > 0:
      quintic = raw_phase.pow(3) * (10 - 15 * raw_phase + 6 * raw_phase.square())
      blend = torch.where(self.continuous_active[:, None, None], quintic[:, None, None], blend)
    self.target_pos_w.copy_(
      self.start_pos_w + blend * (self.final_pos_w - self.start_pos_w)
    )
    if self.cfg.continuous_probability > 0 and not self.scripted and not self.cfg.capability_pack:
      age = (self.elapsed - self.cfg.reach_delay_s - self.cfg.reach_duration_s).clamp_min(0)
      cycle = torch.remainder(age[:, None] / self.continuous_period, 1.)
      triangular = 1 - (2 * cycle - 1).abs()
      wave = triangular.pow(3) * (10 - 15 * triangular + 6 * triangular.square())
      self.target_pos_w += self.continuous_delta_w * wave[..., None] * self.continuous_active[:, None, None]
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
    pack_offset = None
    if self.cfg.capability_pack and not self.scripted:
      from .operation_clip import evaluate_clip, continue_clip
      age = (self.elapsed-self.cfg.reach_delay_s-self.cfg.reach_duration_s).clamp_min(0)
      finished = (age[:, None]-self.pack_start_time >= self.pack_duration.sum(1)) & self.persistent_active[:, None]
      ids = finished.any(-1).nonzero().flatten()
      if len(ids):
        self.pack_start_time[ids] += self.pack_duration[ids].sum(1)*finished[ids]
        self.pack_continuations[ids] += finished[ids].float()
        delta, quats, durations = self.pack_delta[ids], self.pack_quat[ids], self.pack_duration[ids]
        continue_clip(delta, quats, durations, finished[ids], self.sampled_offset_b[ids],
                      self.pack_reference_quat[ids], self.final_quat_w[ids], self.height_active[ids],
                      self.height_difficulty[ids], self.cfg)
        self.pack_delta[ids], self.pack_quat[ids], self.pack_duration[ids] = delta, quats, durations
      pack_offset, pack_orientation, self.pack_stage = evaluate_clip(self.pack_delta, self.pack_quat, self.pack_duration, age[:, None]-self.pack_start_time)
      active = self.continuous_active & ~self.egodex_active & (self.elapsed >= self.cfg.reach_delay_s+self.cfg.reach_duration_s)
      self.target_pos_w.copy_(torch.where(active[:, None, None], self.final_pos_w+pack_offset, self.target_pos_w))
      self.target_quat_w.copy_(torch.where(active[:, None, None], pack_orientation, self.target_quat_w))
    if self.egodex_positions is not None:
      active = self.egodex_active & (self.elapsed >= self.cfg.reach_delay_s+self.cfg.reach_duration_s)
      ids = active.nonzero().flatten()
      if len(ids):
        ego_pos, ego_shoulder, ego_quat = self._egodex_targets(ids, self.elapsed[ids]-self.cfg.reach_delay_s-self.cfg.reach_duration_s)
        self.target_pos_w[ids] = ego_pos
        self.target_shoulder_height[ids] = ego_shoulder
        self.target_quat_w[ids] = ego_quat
    self.target_lin_vel_w.copy_(
      (self.target_pos_w - self.previous_target_pos_w) / self._env.step_dt
    )
    height_blend = raw_phase * raw_phase * (3.0 - 2.0 * raw_phase)
    if self.cfg.continuous_probability > 0:
      height_blend = torch.where(self.continuous_active, quintic, height_blend)
    height_blend *= self.height_active.float()
    self.target_shoulder_height.copy_(
      self.start_shoulder_height
      + height_blend * (self.final_shoulder_height - self.start_shoulder_height)
    )
    if pack_offset is not None:
      z = pack_offset[..., 2]
      # Smooth shared lift, <= either hand's lift; avoid the velocity kink of min.
      shared_lift = z[:, 0]*z[:, 1]/(z.sum(-1)+.02)
      self.target_shoulder_height += shared_lift * active * self.height_active
    if self.egodex_positions is not None:
      active = self.egodex_active & (self.elapsed >= self.cfg.reach_delay_s+self.cfg.reach_duration_s)
      ids = active.nonzero().flatten()
      if len(ids):
        ego_pos, ego_shoulder, ego_quat = self._egodex_targets(ids, self.elapsed[ids]-self.cfg.reach_delay_s-self.cfg.reach_duration_s)
        self.target_pos_w[ids] = ego_pos
        self.target_shoulder_height[ids] = ego_shoulder
        self.target_quat_w[ids] = ego_quat
      self.target_lin_vel_w.copy_((self.target_pos_w - self.previous_target_pos_w) / self._env.step_dt)

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
    if self.cfg.capability_pack:
      rotation4 = rotation[:, None, :, :].expand(-1, 4, -1, -1)
      self.pack_delta.copy_(quat_apply(rotation4, self.pack_delta))
      self.pack_quat.copy_(quat_mul(rotation4, self.pack_quat))
      self.pack_reference_quat.copy_(quat_mul(rotation[:, 0], self.pack_reference_quat))
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
    self.ground_active.zero_()
    self.bilateral_ground_active.zero_()
    self.continuous_active.zero_()
    self.persistent_active.zero_()
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
    # Command-reset snapshots: useful screening proxies, not a frequency spectrum.
    self.metrics["leg_joint_acc_rms_snapshot"] = torch.sqrt(
      torch.square(self.robot.data.joint_acc[:, self.leg_joint_ids]).mean(-1)
    )
    self.metrics["leg_joint_vel_rms_snapshot"] = torch.sqrt(
      torch.square(self.robot.data.joint_vel[:, self.leg_joint_ids]).mean(-1)
    )
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
    if self.cfg.ground_probability > 0:
      mask = self.ground_active.float()
      self.metrics["ground_fraction"] = mask
      self.metrics["ground_wrist_error_masked"] = mean_pos_error*mask
      self.metrics["ground_shoulder_error_masked"] = self.metrics["shoulder_height_error"]*mask
      self.metrics["ground_low_target_masked"] = self.final_pos_w[...,2].min(-1).values*mask
    if self.cfg.bilateral_ground_probability > 0:
      mask = self.bilateral_ground_active.float()
      self.metrics["bilateral_ground_fraction"] = mask
      self.metrics["bilateral_ground_wrist_error_masked"] = pos_error.max(-1).values * mask
      self.metrics["bilateral_ground_shoulder_error_masked"] = self.metrics["shoulder_height_error"] * mask
      self.metrics["bilateral_ground_target_min_masked"] = self.final_pos_w[..., 2].min(-1).values * mask
      self.metrics["bilateral_ground_target_max_masked"] = self.final_pos_w[..., 2].max(-1).values * mask
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
      # Measure the actual intersection, not separate marginal success rates.
      # Only evaluate low targets after their reach transition is complete.
      speed = torch.linalg.vector_norm(twist.command[:, :2], dim=-1)
      active_bilateral = self.bilateral_ground_active & (twist.mode == 1) & (speed > 1e-3) & (self.phase >= .99)
      active_float = active_bilateral.float()
      projected_speed = (self.robot.data.root_link_lin_vel_b[:, :2] * twist.command[:, :2]).sum(-1) / speed.clamp_min(1e-6)
      stats = self.bilateral_transport_statistics
      for column, value in enumerate((torch.ones_like(speed), pos_error.max(-1).values,
                                     self.metrics["shoulder_height_error"], speed,
                                     projected_speed, xy_error)):
        stats[:, column] += value * active_float
      self.metrics["bilateral_transport_moving_fraction"] = stats[:, 0] / denominator
      for column, suffix in enumerate(("wrist_error", "shoulder_error", "command_xy",
                                       "projected_speed", "velocity_xy_error"), start=1):
        self.metrics[f"bilateral_transport_moving_{suffix}_masked"] = stats[:, column] / denominator
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


    if self.cfg.continuous_probability > 0:
      steady = self.continuous_active & (self.elapsed >= self.cfg.reach_delay_s + self.cfg.reach_duration_s)
      speed = torch.linalg.vector_norm(self.target_lin_vel_w, dim=-1)
      moving = steady & (speed.mean(-1) > .001)
      actual = self.robot.data.body_link_lin_vel_w[:, self.wrist_body_ids]
      projected = (actual * self.target_lin_vel_w).sum(-1) / speed.clamp_min(1e-6)
      stats = self.continuous_statistics
      stats[:, 0] += 1
      stats[:, 1] += steady.float()
      stats[:, 2] += pos_error.max(-1).values * steady
      stats[:, 3] += quat_error_magnitude(self.desired_wrist_quat_w, self.robot_wrist_quat_w).max(-1).values * steady
      stats[:, 4] += moving.float()
      stats[:, 5] += speed.mean(-1) * moving
      stats[:, 6] += projected.mean(-1) * moving
      stats[:, 7] += torch.linalg.vector_norm(actual - self.target_lin_vel_w, dim=-1).mean(-1) * moving
      if self.cfg.capability_pack:
        pack = self.pack_statistics
        pack[:, 0] += steady.float()
        pack[:, 1] += self.metrics["shoulder_height_error"]*steady
        pack[:, 2] = torch.maximum(pack[:, 2], pos_error.max(-1).values*steady)
        pack[:, 3] += steady*(self.pack_stage == 1).any(-1)
        pack[:, 4] += steady*(self.pack_stage == 2).any(-1)
        total = stats[:, 0].clamp_min(1)
        self.metrics["pack_case_fraction"] = self.continuous_active.float()
        self.metrics["pack_shoulder_error_masked"] = pack[:, 1]/total
        self.metrics["pack_peak_wrist_error_masked"] = pack[:, 2]*self.continuous_active
        self.metrics["pack_lift_fraction"] = pack[:, 3]/total
        self.metrics["pack_place_fraction"] = pack[:, 4]/total
        self.metrics["persistent_fraction"] = self.persistent_active.float()
        self.metrics["persistent_continuations_masked"] = self.pack_continuations.mean(-1)*self.persistent_active
        mask = steady & self.persistent_active
        ps = self.persistent_statistics
        for column,value in enumerate((torch.ones_like(total),pos_error.max(-1).values,
                                       quat_error_magnitude(self.desired_wrist_quat_w,self.robot_wrist_quat_w).max(-1).values,
                                       self.metrics['shoulder_height_error'])):
          ps[:,column] += value*mask
        for column,key in enumerate(('steady_fraction','wrist_error_masked','rotation_error_masked','shoulder_error_masked')):
          self.metrics['persistent_'+key] = ps[:,column]/total
      denominator = stats[:, 0].clamp_min(1)
      for column, key in enumerate(("continuous_fraction", "continuous_wrist_error_masked", "continuous_rotation_error_masked",
                                    "continuous_moving_fraction", "continuous_command_speed_masked",
                                    "continuous_projected_speed_masked", "continuous_velocity_error_masked"), start=1):
        self.metrics[key] = stats[:, column] / denominator
    if self.cfg.diagnostics_enabled:
      self.operation_diagnostics.update(self)


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
  height_spatial_sampling: bool = False
  height_lateral_offset_range: tuple[float, float] = (-0.04, 0.04)
  ground_probability: float = 0.0
  """Fraction of height episodes assigned near-ground targets."""
  bilateral_ground_probability: float = 0.0
  """Conditional fraction of ground episodes with BOTH wrists sampled low."""
  ground_wrist_height_range: tuple[float, float] = (0.08, 0.18)
  ground_other_wrist_raise_range: tuple[float, float] = (0.10, 0.20)
  clutch_enabled: bool = False
  shoulder_height_range: tuple[float, float] = (0.72, 1.02)
  wrist_height_offset_range: tuple[float, float] = (-0.02, 0.02)
  low_reach_extension_range: tuple[float, float] = (0.08, 0.22)
  reach_delay_s: float = 1.0
  reach_duration_s: float = 2.0
  continuous_probability: float = 0.0
  egodex_probability: float = 0.0
  """Fraction of resets that use a mapped EgoDex continuous command."""
  egodex_data_path: str = "data/egodex_test_wrist_commands.npz"
  egodex_source_fps: float = 30.0
  capability_pack: bool = False
  persistent_probability: float = 0.0
  diagnostics_enabled: bool = False
  diagnostics_trace_period_steps: int = 10000
  diagnostics_trace_window_steps: int = 128
  continuous_displacement: float = .03
  continuous_period_range: tuple[float, float] = (4., 6.)
  curriculum_warmup_steps: int = 30_000
  curriculum_ramp_steps: int = 60_000

  def build(self, env: ManagerBasedRlEnv) -> BimanualWristCommand:
    return BimanualWristCommand(self, env)
