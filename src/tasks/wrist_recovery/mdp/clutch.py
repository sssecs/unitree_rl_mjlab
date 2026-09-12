"""Episode-level transport/anchored-adjustment/balance command mixture."""

from dataclasses import dataclass

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg


class ClutchedVelocityCommand(UniformVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    if cfg.curriculum_ramp_steps <= 0:
      raise ValueError("curriculum_ramp_steps must be positive")
    if cfg.clutch_enabled and (cfg.heading_command or cfg.init_velocity_prob != 0):
      raise ValueError("Clutch mixture requires direct twist and zero initial velocity probability")
    if cfg.clutch_enabled and cfg.resampling_time_range[0] < env.cfg.episode_length_s:
      raise ValueError("Clutch resampling interval must cover the entire episode")
    if min(cfg.transport_probability, cfg.adjust_probability) < 0 or (
      cfg.transport_probability + cfg.adjust_probability > 1
    ):
      raise ValueError("Clutch probabilities must be nonnegative and sum to <=1")
    if min(cfg.adjust_distance_limit, cfg.adjust_yaw_limit, cfg.adjust_duration_s) <= 0:
      raise ValueError("Adjustment limits must be positive")
    self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.sampled_twist = torch.zeros_like(self.vel_command_b)
    self.adjust_distance = torch.zeros(self.num_envs, device=self.device)
    self.adjust_yaw = torch.zeros_like(self.adjust_distance)
    self.age = torch.zeros_like(self.adjust_distance)

  def _resample_command(self, env_ids):
    if not self.cfg.clutch_enabled:
      return super()._resample_command(env_ids)
    # The long resampling interval makes this an episode-level mode. Reset
    # resamples it; limits are not renewed halfway through an anchored episode.
    draw = torch.rand(len(env_ids), device=self.device)
    curriculum = max(0., min(1., (self._env.common_step_counter - self.cfg.curriculum_warmup_steps) / self.cfg.curriculum_ramp_steps))
    p_transport = self.cfg.transport_probability * curriculum
    p_adjust = self.cfg.adjust_probability * curriculum
    self.mode[env_ids] = torch.where(draw < p_transport, 1,
      torch.where(draw < p_transport + p_adjust, 2, 0))
    bounds = (self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y,
              self.cfg.ranges.ang_vel_z)
    for axis, limits in enumerate(bounds):
      self.sampled_twist[env_ids, axis] = torch.empty(len(env_ids), device=self.device).uniform_(*limits) * curriculum
    adjust = self.mode[env_ids] == 2
    limits = (self.cfg.adjust_linear_speed, self.cfg.adjust_linear_speed,
              self.cfg.adjust_angular_speed)
    for axis, limit in enumerate(limits):
      values = torch.empty(len(env_ids), device=self.device).uniform_(-limit, limit) * curriculum
      self.sampled_twist[env_ids, axis] = torch.where(adjust, values, self.sampled_twist[env_ids, axis])
    self.sampled_twist[env_ids] *= (self.mode[env_ids] != 0)[:, None]
    self.vel_command_b[env_ids] = self.sampled_twist[env_ids]
    self.is_standing_env[env_ids] = self.mode[env_ids] == 0
    self.adjust_distance[env_ids] = 0
    self.adjust_yaw[env_ids] = 0
    self.age[env_ids] = 0

  def _update_command(self):
    if not self.cfg.clutch_enabled:
      return super()._update_command()
    dt = self._env.step_dt
    self.age += dt
    self.vel_command_b.copy_(self.sampled_twist)
    adjusting = self.mode == 2
    self.vel_command_b[adjusting & (self.age < self.cfg.adjust_delay_s)] = 0
    self.vel_command_b[adjusting & (self.age >= self.cfg.adjust_delay_s + self.cfg.adjust_duration_s)] = 0
    # Clip the last command step to the remaining path-length/yaw budget.
    speed = torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=-1)
    remaining = (self.cfg.adjust_distance_limit - self.adjust_distance).clamp_min(0)
    scale = (remaining / (speed * dt).clamp_min(1e-9)).clamp(max=1)
    self.vel_command_b[:, :2] *= torch.where(adjusting, scale, 1.)[:, None]
    yaw = self.vel_command_b[:, 2].abs()
    remaining_yaw = (self.cfg.adjust_yaw_limit - self.adjust_yaw).clamp_min(0)
    scale_yaw = (remaining_yaw / (yaw * dt).clamp_min(1e-9)).clamp(max=1)
    self.vel_command_b[:, 2] *= torch.where(adjusting, scale_yaw, 1.)
    self.adjust_distance += torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=-1) * dt * adjusting
    self.adjust_yaw += self.vel_command_b[:, 2].abs() * dt * adjusting


@dataclass(kw_only=True)
class ClutchedVelocityCommandCfg(UniformVelocityCommandCfg):
  clutch_enabled: bool = False
  curriculum_warmup_steps: int = 30000
  curriculum_ramp_steps: int = 60000
  transport_probability: float = 0.25
  adjust_probability: float = 0.25
  adjust_linear_speed: float = 0.04
  adjust_angular_speed: float = 0.06
  adjust_distance_limit: float = 0.08
  adjust_yaw_limit: float = 0.12
  adjust_delay_s: float = 3.0
  adjust_duration_s: float = 2.0

  def build(self, env):
    return ClutchedVelocityCommand(self, env)
