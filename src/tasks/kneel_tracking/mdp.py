"""TWIST-inspired teacher inputs with soft tracking and start-only resets."""

from dataclasses import dataclass

import numpy as np
import torch

from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply_inverse


class KneelCommand(MotionCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    with np.load(cfg.motion_file, allow_pickle=False) as data:
      if not np.isclose(float(data["fps"]), 1 / env.step_dt):
        raise ValueError("Kneel reference fps must match the environment control rate")
      if data["joint_names"].tolist() != list(self.robot.joint_names):
        raise ValueError("Kneel reference joint order differs from the robot")
      if data["body_names"].tolist() != list(self.robot.body_names):
        raise ValueError("Kneel reference body order differs from the robot")
      for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                  "body_lin_vel_w", "body_ang_vel_w", "contact_targets"):
        if not np.isfinite(data[key]).all():
          raise ValueError(f"Nonfinite reference array: {key}")
      self.contact_targets = torch.as_tensor(
        data["contact_targets"], device=self.device, dtype=torch.float32
      )
    self._just_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    for name in ("phase", "joint_rmse", "root_lin_vel_rmse", "root_height_error",
                 "left_knee_contact", "right_knee_contact"):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids):
    # Always initialize from the clip's first frame. No random kneel RSI.
    super()._resample_command(env_ids)
    # Reset has no pose perturbation, so aligned targets equal the reference.
    # Avoid retaining the previous episode's aligned targets until compute().
    self.body_pos_relative_w[env_ids] = self.body_pos_w[env_ids]
    self.body_quat_relative_w[env_ids] = self.body_quat_w[env_ids]
    self._just_reset[env_ids] = True

  def _update_command(self):
    # Parent increments by one. Hold the last frame instead of teleporting when
    # the clip ends, and preserve frame zero on the first post-reset update.
    self.time_steps.clamp_(max=self.motion.time_step_total - 2)
    self.time_steps[self._just_reset] = -1
    super()._update_command()
    self._just_reset.zero_()

  def _update_metrics(self):
    super()._update_metrics()
    self.metrics["phase"] = self.time_steps.float() / (self.motion.time_step_total - 1)
    self.metrics["joint_rmse"] = (self.joint_pos - self.robot_joint_pos).square().mean(-1).sqrt()
    target, measured = root_velocities(self)
    self.metrics["root_lin_vel_rmse"] = (target - measured).square().mean(-1).sqrt()
    self.metrics["root_height_error"] = (
      self.body_pos_w[:, 0, 2] - self.robot_body_pos_w[:, 0, 2]
    ).abs()
    for side in ("left", "right"):
      found = self._env.scene[f"{side}_knee_ground"].data.found
      self.metrics[f"{side}_knee_contact"] = found.reshape(self.num_envs, -1).any(-1).float()


@dataclass(kw_only=True)
class KneelCommandCfg(MotionCommandCfg):
  future_offsets: tuple[int, ...] = (0, 5, 10, 25, 50, 100)
  include_contact_targets: bool = True

  def build(self, env):
    if self.sampling_mode != "start":
      raise ValueError("Kneel task currently supports standing/start resets only")
    if any(offset < 0 for offset in self.future_offsets):
      raise ValueError("Future offsets must be nonnegative")
    return KneelCommand(self, env)


def get_command(env, command_name="kneel"):
  return env.command_manager.get_term(command_name)


def future_reference(env, command_name="kneel"):
  command = get_command(env, command_name)
  offsets = torch.tensor(command.cfg.future_offsets, device=env.device)
  frames = (command.time_steps[:, None] + offsets).clamp(max=command.motion.time_step_total - 1)
  motion = command.motion
  # Root is first in the tracked body list. Velocities in the reference root
  # frame avoid demanding a globally exact XY path from imperfect references.
  root_quat = motion.body_quat_w[frames, 0]
  root_lin = quat_apply_inverse(root_quat, motion.body_lin_vel_w[frames, 0])
  root_ang = quat_apply_inverse(root_quat, motion.body_ang_vel_w[frames, 0])
  gravity = torch.zeros_like(root_lin)
  gravity[..., 2] = -1
  terms = [
    motion.joint_pos[frames], motion.joint_vel[frames], root_lin, root_ang,
    quat_apply_inverse(root_quat, gravity), motion.body_pos_w[frames, 0, 2:3],
  ]
  if command.cfg.include_contact_targets:
    terms.append(command.contact_targets[frames])
  return torch.cat(terms, dim=-1).flatten(1)


def contact_phase(env):
  command = get_command(env)
  return torch.cat([
    command.contact_targets[command.time_steps],
    command.time_steps[:, None].float() / (command.motion.time_step_total - 1),
  ], dim=-1)


def root_velocities(command):
  target = quat_apply_inverse(command.body_quat_w[:, 0], command.body_lin_vel_w[:, 0])
  measured = quat_apply_inverse(command.robot_body_quat_w[:, 0], command.robot_body_lin_vel_w[:, 0])
  return target, measured


def joint_tracking(env, velocity=False, std=0.5, command_name="kneel"):
  command = get_command(env, command_name)
  target = command.joint_vel if velocity else command.joint_pos
  measured = command.robot_joint_vel if velocity else command.robot_joint_pos
  return torch.exp(-(target - measured).square().mean(-1) / std**2)


def root_velocity_tracking(env, angular=False, std=1.0, command_name="kneel"):
  command = get_command(env, command_name)
  if angular:
    target = quat_apply_inverse(command.body_quat_w[:, 0], command.body_ang_vel_w[:, 0])
    measured = quat_apply_inverse(command.robot_body_quat_w[:, 0], command.robot_body_ang_vel_w[:, 0])
  else:
    target, measured = root_velocities(command)
  return torch.exp(-(target - measured).square().mean(-1) / std**2)
