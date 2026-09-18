from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_mul,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from .motion_library import SparseWholeBodyMotionLibrary

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


EXPECTED_INTERFACE_VERSION = "3.2"

# Desired command frames use pastel axes; simulated robot frames use normal RGB.
_CMD_FRAME_COLORS = (
  (1.0, 0.5, 0.5),
  (0.5, 1.0, 0.5),
  (0.5, 0.5, 1.0),
)

_LEFT_ERROR_COLOR = (1.0, 0.55, 0.10, 0.95)
_RIGHT_ERROR_COLOR = (0.80, 0.25, 1.00, 0.95)
_SHOULDER_MID_ERROR_COLOR = (1.0, 0.85, 0.10, 0.95)
_SHOULDER_HEIGHT_ERROR_COLOR = (0.20, 0.85, 0.85, 0.95)
_SHOULDER_HEIGHT_TARGET_COLOR = (0.20, 0.85, 0.85, 0.75)



def _yaw_quat_tensor(yaw: torch.Tensor) -> torch.Tensor:
  """Construct scalar-first yaw quaternions."""
  half = 0.5 * yaw
  q = torch.zeros((*yaw.shape, 4), dtype=yaw.dtype, device=yaw.device)
  q[..., 0] = torch.cos(half)
  q[..., 3] = torch.sin(half)
  return q



def _rotation_6d(q: torch.Tensor) -> torch.Tensor:
  """First two rotation-matrix columns, concatenated column-by-column."""
  matrix = matrix_from_quat(q)
  return matrix[..., :, :2].transpose(-1, -2).reshape(q.shape[0], 6)


def _quat_nlerp_shortest(
  q0: torch.Tensor,
  q1: torch.Tensor,
  alpha: torch.Tensor,
) -> torch.Tensor:
  """Shortest-path normalized quaternion interpolation.

  ``alpha`` is shape (N,); quaternions are scalar-first [w, x, y, z].
  """
  dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
  q1_short = torch.where(dot < 0.0, -q1, q1)

  a = alpha
  while a.ndim < q0.ndim:
    a = a.unsqueeze(-1)

  q = q0 + a * (q1_short - q0)
  return torch.nn.functional.normalize(q, dim=-1)


def _rotate_xy(xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """Rotate batched XY vectors by a batched yaw."""
  c = torch.cos(yaw)
  s = torch.sin(yaw)
  x = xy[..., 0]
  y = xy[..., 1]
  return torch.stack((c * x - s * y, s * x + c * y), dim=-1)




class SparseWholeBodyCommand(CommandTerm):
  """Absolute teacher command with one fixed episode-level SE(2) alignment.

  Naming:
    npz_* : arrays stored by the source NPZ
    cmd_* : current command target
    sim_* : state measured from MuJoCo

  Frames:
    *_cw  : NPZ command world
    *_ew  : fixed per-environment world (sim world minus static env_origin)
    *_w   : MuJoCo simulator world

  ``*_ew`` is still an absolute/fixed frame: it never follows the robot.
  It only removes the arbitrary vectorized-environment grid translation.
  """

  cfg: SparseWholeBodyCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(
    self,
    cfg: SparseWholeBodyCommandCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    self.stabilization_body_index = self.robot.body_names.index(
      cfg.stabilization_body_name
    )
    self.left_wrist_index = self.robot.body_names.index(
      cfg.left_wrist_body_name
    )
    self.right_wrist_index = self.robot.body_names.index(
      cfg.right_wrist_body_name
    )
    self.left_shoulder_index = self.robot.body_names.index(
      cfg.left_shoulder_body_name
    )
    self.right_shoulder_index = self.robot.body_names.index(
      cfg.right_shoulder_body_name
    )

    command_dir = cfg.command_dir or os.environ.get(
      cfg.command_dir_env_var, ""
    )
    command_file = cfg.command_file or os.environ.get(
      cfg.command_file_env_var, ""
    )

    # Directory mode takes precedence if both are provided.
    command_source = command_dir or command_file
    if not command_source:
      raise RuntimeError(
        "No teleop command source configured. Set "
        f"{cfg.command_dir_env_var}=<directory> for recursive multi-motion "
        f"training, or {cfg.command_file_env_var}=<episode.npz> for "
        "single-motion debugging."
      )

    command_source = os.path.abspath(os.path.expanduser(command_source))
    self.motion = SparseWholeBodyMotionLibrary(
      command_source=command_source,
      device=self.device,
      canonicalize_heading=cfg.canonicalize_heading,
      recursive=cfg.recursive_scan,
      skip_invalid_files=cfg.skip_invalid_files,
      sampling_weight_mode=cfg.motion_sampling_weight_mode,
    )

    # Every vectorized environment owns an independent selected motion id.
    self.motion_ids = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    self.command_time = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )

    # Current interpolated raw NPZ values in command world.
    self._npz_left_wrist_pos_cw = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self._npz_left_wrist_quat_cw = torch.zeros(
      self.num_envs, 4, device=self.device
    )
    self._npz_right_wrist_pos_cw = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self._npz_right_wrist_quat_cw = torch.zeros(
      self.num_envs, 4, device=self.device
    )
    self._npz_shoulder_mid_xy_cw = torch.zeros(
      self.num_envs, 2, device=self.device
    )
    self._npz_left_shoulder_height = torch.zeros(
      self.num_envs, device=self.device
    )
    self._npz_right_shoulder_height = torch.zeros(
      self.num_envs, device=self.device
    )
    self._npz_shoulder_heading_cw = torch.zeros(
      self.num_envs, device=self.device
    )

    # Fixed alignment for each environment/episode:
    #
    #   p_ew = Rz(align_yaw) * p_cw + [tx, ty, 0]
    #
    # z is not shoulder-aligned: the command ground stays z=0 in _ew.
    self.episode_align_yaw = torch.zeros(
      self.num_envs, device=self.device
    )
    self.episode_align_quat_w = _yaw_quat_tensor(self.episode_align_yaw)
    self.episode_align_translation_xy_ew = torch.zeros(
      self.num_envs, 2, device=self.device
    )
    self._alignment_pending = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    # Episode pre-roll / command warm-up.
    #
    # At reset we snapshot the post-reset simulated task pose. During warm-up,
    # the externally visible cmd_* target moves smoothly from this snapshot to
    # the fixed-aligned NPZ reference while command_time remains frozen.
    self.warmup_time = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self._warmup_start_left_wrist_pos_ew = torch.zeros(
      self.num_envs, 3, dtype=torch.float32, device=self.device
    )
    self._warmup_start_left_wrist_quat_w = torch.zeros(
      self.num_envs, 4, dtype=torch.float32, device=self.device
    )
    self._warmup_start_right_wrist_pos_ew = torch.zeros(
      self.num_envs, 3, dtype=torch.float32, device=self.device
    )
    self._warmup_start_right_wrist_quat_w = torch.zeros(
      self.num_envs, 4, dtype=torch.float32, device=self.device
    )
    self._warmup_start_shoulder_mid_xy_ew = torch.zeros(
      self.num_envs, 2, dtype=torch.float32, device=self.device
    )
    self._warmup_start_left_shoulder_height = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self._warmup_start_right_shoulder_height = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self._warmup_start_shoulder_heading_w = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )

    metric_names = (
      "cmd_vs_sim_left_wrist_pos_error",
      "cmd_vs_sim_right_wrist_pos_error",
      "cmd_vs_sim_left_wrist_ori_error",
      "cmd_vs_sim_right_wrist_ori_error",
      "cmd_vs_sim_shoulder_mid_xy_error",
      "cmd_vs_sim_shoulder_heading_error",
      "cmd_vs_sim_left_shoulder_height_error",
      "cmd_vs_sim_right_shoulder_height_error",
    )
    for name in metric_names:
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    all_envs = torch.arange(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._update_npz_targets(all_envs)

  @property
  def current_motion_duration(self) -> torch.Tensor:
    """Selected clip duration for every vectorized environment."""
    return self.motion.get_motion_length(self.motion_ids)

  def get_motion_name(self, env_idx: int) -> str:
    """Relative path/name of the clip currently assigned to one environment."""
    return self.motion.get_motion_name(int(self.motion_ids[env_idx].item()))

  # -----------------------------------------------------------------------
  # Simulation measurements.
  # -----------------------------------------------------------------------

  @property
  def sim_stabilization_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.stabilization_body_index]

  @property
  def sim_stabilization_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.stabilization_body_index]

  @property
  def sim_left_wrist_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.left_wrist_index]

  @property
  def sim_left_wrist_pos_ew(self) -> torch.Tensor:
    return self.sim_left_wrist_pos_w - self._env.scene.env_origins

  @property
  def sim_left_wrist_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.left_wrist_index]

  @property
  def sim_right_wrist_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.right_wrist_index]

  @property
  def sim_right_wrist_pos_ew(self) -> torch.Tensor:
    return self.sim_right_wrist_pos_w - self._env.scene.env_origins

  @property
  def sim_right_wrist_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.right_wrist_index]

  @property
  def sim_left_shoulder_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.left_shoulder_index]

  @property
  def sim_right_shoulder_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.right_shoulder_index]

  @property
  def sim_left_shoulder_height(self) -> torch.Tensor:
    return (
      self.sim_left_shoulder_pos_w[:, 2]
      - self._env.scene.env_origins[:, 2]
    )

  @property
  def sim_right_shoulder_height(self) -> torch.Tensor:
    return (
      self.sim_right_shoulder_pos_w[:, 2]
      - self._env.scene.env_origins[:, 2]
    )

  @property
  def sim_shoulder_mid_pos_w(self) -> torch.Tensor:
    return 0.5 * (
      self.sim_left_shoulder_pos_w
      + self.sim_right_shoulder_pos_w
    )

  @property
  def sim_shoulder_mid_pos_ew(self) -> torch.Tensor:
    return self.sim_shoulder_mid_pos_w - self._env.scene.env_origins

  @property
  def sim_shoulder_heading_w(self) -> torch.Tensor:
    """Heading defined exactly as in the human converter."""
    left_xy = self.sim_left_shoulder_pos_w[:, :2]
    right_xy = self.sim_right_shoulder_pos_w[:, :2]

    y_left = left_xy - right_xy
    y_left = y_left / torch.clamp(
      torch.linalg.vector_norm(y_left, dim=-1, keepdim=True),
      min=1.0e-6,
    )

    x_forward = torch.stack(
      (
        y_left[:, 1],
        -y_left[:, 0],
      ),
      dim=-1,
    )
    return torch.atan2(x_forward[:, 1], x_forward[:, 0])

  @property
  def sim_shoulder_yaw_quat_w(self) -> torch.Tensor:
    return _yaw_quat_tensor(self.sim_shoulder_heading_w)

  @property
  def sim_shoulder_heading_vec_w(self) -> torch.Tensor:
    return torch.stack(
      (
        torch.cos(self.sim_shoulder_heading_w),
        torch.sin(self.sim_shoulder_heading_w),
      ),
      dim=-1,
    )

  # -----------------------------------------------------------------------
  # Episode alignment.
  # -----------------------------------------------------------------------

  def ensure_episode_alignment(self) -> None:
    """Align any pending environments using current post-reset simulation state.

    mjlab v1.2 resets command terms before the reset state is forwarded through
    MuJoCo kinematics. Therefore _resample_command() only marks the alignment
    as pending. The alignment is resolved after forward(), either here when the
    command observation is requested, or in _update_command() after auto-reset.
    """
    env_ids = self._alignment_pending.nonzero(as_tuple=False).flatten()
    if len(env_ids) == 0:
      return
    self._compute_episode_alignment(env_ids)

  def _compute_episode_alignment(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return

    # Align heading first.
    source_heading = self._npz_shoulder_heading_cw[env_ids]
    sim_heading = self.sim_shoulder_heading_w[env_ids]
    delta = sim_heading - source_heading
    # Keep the stored calibration angle in a compact range.
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))

    # Then translate the rotated command shoulder midpoint onto the simulated
    # shoulder midpoint in the fixed per-environment world.
    source_mid_xy = self._npz_shoulder_mid_xy_cw[env_ids]
    rotated_mid_xy = _rotate_xy(source_mid_xy, delta)
    sim_mid_xy_ew = self.sim_shoulder_mid_pos_ew[env_ids, :2]
    translation_xy = sim_mid_xy_ew - rotated_mid_xy

    self.episode_align_yaw[env_ids] = delta
    self.episode_align_quat_w[env_ids] = _yaw_quat_tensor(delta)
    self.episode_align_translation_xy_ew[env_ids] = translation_xy

    # Snapshot the actual POST-RESET task pose.  Warm-up starts exactly here,
    # so the first command target has essentially zero task-space error.
    self._warmup_start_left_wrist_pos_ew[env_ids] = (
      self.sim_left_wrist_pos_ew[env_ids]
    )
    self._warmup_start_left_wrist_quat_w[env_ids] = (
      self.sim_left_wrist_quat_w[env_ids]
    )
    self._warmup_start_right_wrist_pos_ew[env_ids] = (
      self.sim_right_wrist_pos_ew[env_ids]
    )
    self._warmup_start_right_wrist_quat_w[env_ids] = (
      self.sim_right_wrist_quat_w[env_ids]
    )
    self._warmup_start_shoulder_mid_xy_ew[env_ids] = (
      self.sim_shoulder_mid_pos_ew[env_ids, :2]
    )
    self._warmup_start_left_shoulder_height[env_ids] = (
      self.sim_left_shoulder_height[env_ids]
    )
    self._warmup_start_right_shoulder_height[env_ids] = (
      self.sim_right_shoulder_height[env_ids]
    )
    self._warmup_start_shoulder_heading_w[env_ids] = (
      self.sim_shoulder_heading_w[env_ids]
    )
    self.warmup_time[env_ids] = 0.0

    self._alignment_pending[env_ids] = False

  # -----------------------------------------------------------------------
  # Fixed-aligned NPZ reference and emitted command target.
  #
  # Layering is intentionally explicit:
  #   _npz_*_cw          = sampled source data
  #   aligned_npz_*      = one fixed episode SE(2) transform
  #   cmd_*              = target actually seen by policy/reward
  #                         (warm-up blend, then aligned_npz exactly)
  #   sim_*              = measured MuJoCo state
  # -----------------------------------------------------------------------

  def _npz_pos_cw_to_aligned_ew(
    self,
    pos_cw: torch.Tensor,
  ) -> torch.Tensor:
    rotated = quat_apply(self.episode_align_quat_w, pos_cw)
    translation = torch.zeros_like(rotated)
    translation[:, :2] = self.episode_align_translation_xy_ew
    return rotated + translation

  @property
  def warmup_alpha(self) -> torch.Tensor:
    """Per-env blend factor from reset pose (0) to aligned NPZ target (1)."""
    duration = float(self.cfg.warmup_duration_s)
    if duration <= 0.0:
      return torch.ones(
        self.num_envs, dtype=torch.float32, device=self.device
      )

    u = torch.clamp(self.warmup_time / duration, min=0.0, max=1.0)
    if self.cfg.warmup_profile == "linear":
      return u
    if self.cfg.warmup_profile == "smoothstep":
      return u * u * (3.0 - 2.0 * u)
    raise ValueError(
      f"Unknown warmup_profile={self.cfg.warmup_profile!r}; "
      "expected 'linear' or 'smoothstep'."
    )

  @property
  def warmup_active(self) -> torch.Tensor:
    """Whether each env is still in its pre-roll interval."""
    duration = float(self.cfg.warmup_duration_s)
    if duration <= 0.0:
      return torch.zeros(
        self.num_envs, dtype=torch.bool, device=self.device
      )
    return self.warmup_time < (duration - 1.0e-8)

  def _warmup_lerp(
    self,
    start: torch.Tensor,
    target: torch.Tensor,
  ) -> torch.Tensor:
    alpha = self.warmup_alpha
    while alpha.ndim < target.ndim:
      alpha = alpha.unsqueeze(-1)
    return start + alpha * (target - start)

  # ---------------------- fixed-aligned NPZ reference ------------------

  @property
  def aligned_npz_left_wrist_pos_ew(self) -> torch.Tensor:
    return self._npz_pos_cw_to_aligned_ew(self._npz_left_wrist_pos_cw)

  @property
  def aligned_npz_left_wrist_quat_w(self) -> torch.Tensor:
    return quat_mul(
      self.episode_align_quat_w,
      self._npz_left_wrist_quat_cw,
    )

  @property
  def aligned_npz_right_wrist_pos_ew(self) -> torch.Tensor:
    return self._npz_pos_cw_to_aligned_ew(self._npz_right_wrist_pos_cw)

  @property
  def aligned_npz_right_wrist_quat_w(self) -> torch.Tensor:
    return quat_mul(
      self.episode_align_quat_w,
      self._npz_right_wrist_quat_cw,
    )

  @property
  def aligned_npz_shoulder_mid_xy_ew(self) -> torch.Tensor:
    return (
      _rotate_xy(
        self._npz_shoulder_mid_xy_cw,
        self.episode_align_yaw,
      )
      + self.episode_align_translation_xy_ew
    )

  @property
  def aligned_npz_left_shoulder_height(self) -> torch.Tensor:
    return self._npz_left_shoulder_height

  @property
  def aligned_npz_right_shoulder_height(self) -> torch.Tensor:
    return self._npz_right_shoulder_height

  @property
  def aligned_npz_shoulder_heading_w(self) -> torch.Tensor:
    return self._npz_shoulder_heading_cw + self.episode_align_yaw

  # ----------------------- emitted command target ----------------------

  @property
  def cmd_left_wrist_pos_ew(self) -> torch.Tensor:
    return self._warmup_lerp(
      self._warmup_start_left_wrist_pos_ew,
      self.aligned_npz_left_wrist_pos_ew,
    )

  @property
  def cmd_left_wrist_pos_w(self) -> torch.Tensor:
    return self.cmd_left_wrist_pos_ew + self._env.scene.env_origins

  @property
  def cmd_left_wrist_quat_w(self) -> torch.Tensor:
    return _quat_nlerp_shortest(
      self._warmup_start_left_wrist_quat_w,
      self.aligned_npz_left_wrist_quat_w,
      self.warmup_alpha,
    )

  @property
  def cmd_right_wrist_pos_ew(self) -> torch.Tensor:
    return self._warmup_lerp(
      self._warmup_start_right_wrist_pos_ew,
      self.aligned_npz_right_wrist_pos_ew,
    )

  @property
  def cmd_right_wrist_pos_w(self) -> torch.Tensor:
    return self.cmd_right_wrist_pos_ew + self._env.scene.env_origins

  @property
  def cmd_right_wrist_quat_w(self) -> torch.Tensor:
    return _quat_nlerp_shortest(
      self._warmup_start_right_wrist_quat_w,
      self.aligned_npz_right_wrist_quat_w,
      self.warmup_alpha,
    )

  @property
  def cmd_shoulder_mid_xy_ew(self) -> torch.Tensor:
    return self._warmup_lerp(
      self._warmup_start_shoulder_mid_xy_ew,
      self.aligned_npz_shoulder_mid_xy_ew,
    )

  @property
  def cmd_shoulder_mid_xy_w(self) -> torch.Tensor:
    return (
      self.cmd_shoulder_mid_xy_ew
      + self._env.scene.env_origins[:, :2]
    )

  @property
  def cmd_left_shoulder_height(self) -> torch.Tensor:
    return self._warmup_lerp(
      self._warmup_start_left_shoulder_height,
      self.aligned_npz_left_shoulder_height,
    )

  @property
  def cmd_right_shoulder_height(self) -> torch.Tensor:
    return self._warmup_lerp(
      self._warmup_start_right_shoulder_height,
      self.aligned_npz_right_shoulder_height,
    )

  @property
  def cmd_shoulder_heading_w(self) -> torch.Tensor:
    # Interpolate the shortest wrapped yaw displacement.
    start = self._warmup_start_shoulder_heading_w
    target = self.aligned_npz_shoulder_heading_w
    delta = torch.atan2(
      torch.sin(target - start),
      torch.cos(target - start),
    )
    return start + self.warmup_alpha * delta

  @property
  def cmd_shoulder_heading_vec_w(self) -> torch.Tensor:
    return torch.stack(
      (
        torch.cos(self.cmd_shoulder_heading_w),
        torch.sin(self.cmd_shoulder_heading_w),
      ),
      dim=-1,
    )

  @property
  def command(self) -> torch.Tensor:
    """24-D ABSOLUTE teacher command in a fixed per-environment world.

    No part of this vector is expressed relative to the current robot pose.
    The only frame normalization is subtraction of the static env_origin for
    positions, which prevents vectorized-environment grid placement from
    becoming a learning signal.
    """
    self.ensure_episode_alignment()

    return torch.cat(
      (
        self.cmd_left_wrist_pos_ew,                    # 3
        _rotation_6d(self.cmd_left_wrist_quat_w),      # 6
        self.cmd_right_wrist_pos_ew,                   # 3
        _rotation_6d(self.cmd_right_wrist_quat_w),     # 6
        self.cmd_shoulder_mid_xy_ew,                   # 2
        self.cmd_left_shoulder_height[:, None],        # 1
        self.cmd_right_shoulder_height[:, None],       # 1
        self.cmd_shoulder_heading_vec_w,               # 2
      ),
      dim=-1,
    )

  # -----------------------------------------------------------------------
  # CommandTerm lifecycle.
  # -----------------------------------------------------------------------

  def _resample_command(self, env_ids: torch.Tensor):
    """Sample one motion per environment, then request fixed SE(2) alignment.

    This mirrors TWIST's reset-time motion-id sampling. Sampling is with
    replacement, so different vectorized environments can train on different
    NPZ clips simultaneously.
    """
    if len(env_ids) == 0:
      return

    if self.cfg.fixed_motion_id is None:
      sampled_motion_ids = self.motion.sample_motion_ids(len(env_ids))
    else:
      fixed = int(self.cfg.fixed_motion_id)
      if fixed < 0 or fixed >= self.motion.num_motions:
        raise ValueError(
          f"fixed_motion_id={fixed} is outside "
          f"[0, {self.motion.num_motions - 1}]"
        )
      sampled_motion_ids = torch.full(
        (len(env_ids),),
        fixed,
        dtype=torch.long,
        device=self.device,
      )

    self.motion_ids[env_ids] = sampled_motion_ids

    if self.cfg.sampling_mode == "start":
      self.command_time[env_ids] = 0.0
    elif self.cfg.sampling_mode == "uniform":
      self.command_time[env_ids] = self.motion.sample_time(
        sampled_motion_ids
      )
    else:
      raise ValueError(f"Unknown sampling_mode: {self.cfg.sampling_mode}")

    self._update_npz_targets(env_ids)
    self.warmup_time[env_ids] = 0.0
    self._alignment_pending[env_ids] = True

  def _update_command(self):
    """Advance warm-up first, then advance the selected NPZ motion clock.

    The reference motion itself is frozen throughout pre-roll.  This means the
    robot is not asked to "catch a moving train": it first reaches the selected
    clip's starting reference, and only then does command_time progress.
    """
    pending_before = self._alignment_pending.clone()

    # On mjlab v1.2 auto-reset, command compute runs after reset forward().
    # Resolve alignment now, but preserve alpha=0 for this first observation.
    if torch.any(pending_before):
      self._compute_episode_alignment(
        pending_before.nonzero(as_tuple=False).flatten()
      )

    dt = float(self._env.step_dt)
    duration = float(self.cfg.warmup_duration_s)

    if duration > 0.0:
      # Decide based on the PRE-update warm-up state.  If this step reaches
      # alpha=1, command_time still remains frozen at the NPZ start frame.
      # The following step begins normal motion playback.
      warmup_before = self.warmup_time < (duration - 1.0e-8)
      advance_warmup = (~pending_before) & warmup_before

      self.warmup_time[advance_warmup] = torch.clamp(
        self.warmup_time[advance_warmup] + dt,
        max=duration,
      )

      advance_motion = (~pending_before) & (~warmup_before)
    else:
      advance_motion = ~pending_before

    self.command_time[advance_motion] += dt

    motion_durations = self.current_motion_duration
    if self.cfg.loop:
      positive = advance_motion & (motion_durations > 0.0)
      self.command_time[positive] = torch.remainder(
        self.command_time[positive],
        motion_durations[positive],
      )
    else:
      self.command_time[advance_motion] = torch.minimum(
        self.command_time[advance_motion],
        motion_durations[advance_motion],
      )

    env_ids = torch.arange(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._update_npz_targets(env_ids)

  def _update_npz_targets(self, env_ids: torch.Tensor):
    if len(env_ids) == 0:
      return

    (
      left_pos,
      left_quat,
      right_pos,
      right_quat,
      shoulder_mid_xy,
      left_h,
      right_h,
      heading,
    ) = self.motion.sample(
      self.motion_ids[env_ids],
      self.command_time[env_ids],
    )

    self._npz_left_wrist_pos_cw[env_ids] = left_pos
    self._npz_left_wrist_quat_cw[env_ids] = left_quat
    self._npz_right_wrist_pos_cw[env_ids] = right_pos
    self._npz_right_wrist_quat_cw[env_ids] = right_quat
    self._npz_shoulder_mid_xy_cw[env_ids] = shoulder_mid_xy
    self._npz_left_shoulder_height[env_ids] = left_h
    self._npz_right_shoulder_height[env_ids] = right_h
    self._npz_shoulder_heading_cw[env_ids] = heading

  def _update_metrics(self):
    pending = self._alignment_pending

    metric_values = {
      "cmd_vs_sim_left_wrist_pos_error": torch.linalg.vector_norm(
        self.cmd_left_wrist_pos_w - self.sim_left_wrist_pos_w,
        dim=-1,
      ),
      "cmd_vs_sim_right_wrist_pos_error": torch.linalg.vector_norm(
        self.cmd_right_wrist_pos_w - self.sim_right_wrist_pos_w,
        dim=-1,
      ),
      "cmd_vs_sim_left_wrist_ori_error": quat_error_magnitude(
        self.cmd_left_wrist_quat_w,
        self.sim_left_wrist_quat_w,
      ),
      "cmd_vs_sim_right_wrist_ori_error": quat_error_magnitude(
        self.cmd_right_wrist_quat_w,
        self.sim_right_wrist_quat_w,
      ),
      "cmd_vs_sim_shoulder_mid_xy_error": torch.linalg.vector_norm(
        self.cmd_shoulder_mid_xy_w
        - self.sim_shoulder_mid_pos_w[:, :2],
        dim=-1,
      ),
      "cmd_vs_sim_shoulder_heading_error": quat_error_magnitude(
        _yaw_quat_tensor(self.cmd_shoulder_heading_w),
        self.sim_shoulder_yaw_quat_w,
      ),
      "cmd_vs_sim_left_shoulder_height_error": torch.abs(
        self.cmd_left_shoulder_height
        - self.sim_left_shoulder_height
      ),
      "cmd_vs_sim_right_shoulder_height_error": torch.abs(
        self.cmd_right_shoulder_height
        - self.sim_right_shoulder_height
      ),
    }

    for name, value in metric_values.items():
      if torch.any(pending):
        value = value.clone()
        value[pending] = 0.0
      self.metrics[name] = value

  # -----------------------------------------------------------------------
  # Viewer debugging.
  # -----------------------------------------------------------------------

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    self.ensure_episode_alignment()

    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    viz = self.cfg.viz

    cmd_left_rot = matrix_from_quat(self.cmd_left_wrist_quat_w)
    cmd_right_rot = matrix_from_quat(self.cmd_right_wrist_quat_w)
    cmd_shoulder_rot = matrix_from_quat(
      _yaw_quat_tensor(self.cmd_shoulder_heading_w)
    )

    sim_left_rot = matrix_from_quat(self.sim_left_wrist_quat_w)
    sim_right_rot = matrix_from_quat(self.sim_right_wrist_quat_w)
    sim_shoulder_rot = matrix_from_quat(self.sim_shoulder_yaw_quat_w)
    sim_stabilization_rot = matrix_from_quat(
      self.sim_stabilization_body_quat_w
    )

    identity = np.eye(3, dtype=np.float32)

    for batch in env_indices:
      env_origin = self._env.scene.env_origins[batch]

      cmd_left_pos = self.cmd_left_wrist_pos_w[batch]
      cmd_right_pos = self.cmd_right_wrist_pos_w[batch]
      sim_left_pos = self.sim_left_wrist_pos_w[batch]
      sim_right_pos = self.sim_right_wrist_pos_w[batch]
      sim_shoulder_mid = self.sim_shoulder_mid_pos_w[batch]
      sim_stabilization_pos = self.sim_stabilization_body_pos_w[batch]

      if viz.show_command_world:
        visualizer.add_frame(
          position=env_origin.detach().cpu().numpy(),
          rotation_matrix=identity,
          scale=viz.command_world_scale,
          axis_radius=viz.axis_radius,
          label=f"episode_world_{batch}",
        )

      if viz.show_command_frames:
        visualizer.add_frame(
          position=cmd_left_pos.detach().cpu().numpy(),
          rotation_matrix=cmd_left_rot[batch].detach().cpu().numpy(),
          scale=viz.command_wrist_scale,
          axis_radius=viz.axis_radius,
          label=f"cmd_left_wrist_{batch}",
          axis_colors=_CMD_FRAME_COLORS,
        )
        visualizer.add_frame(
          position=cmd_right_pos.detach().cpu().numpy(),
          rotation_matrix=cmd_right_rot[batch].detach().cpu().numpy(),
          scale=viz.command_wrist_scale,
          axis_radius=viz.axis_radius,
          label=f"cmd_right_wrist_{batch}",
          axis_colors=_CMD_FRAME_COLORS,
        )

      if viz.show_sim_frames:
        visualizer.add_frame(
          position=sim_left_pos.detach().cpu().numpy(),
          rotation_matrix=sim_left_rot[batch].detach().cpu().numpy(),
          scale=viz.sim_wrist_scale,
          axis_radius=viz.axis_radius,
          label=f"sim_left_wrist_{batch}",
        )
        visualizer.add_frame(
          position=sim_right_pos.detach().cpu().numpy(),
          rotation_matrix=sim_right_rot[batch].detach().cpu().numpy(),
          scale=viz.sim_wrist_scale,
          axis_radius=viz.axis_radius,
          label=f"sim_right_wrist_{batch}",
        )

      if viz.show_error_arrows:
        visualizer.add_arrow(
          start=sim_left_pos.detach().cpu().numpy(),
          end=cmd_left_pos.detach().cpu().numpy(),
          color=_LEFT_ERROR_COLOR,
          width=viz.error_arrow_width,
          label=f"cmd_vs_sim_left_wrist_pos_{batch}",
        )
        visualizer.add_arrow(
          start=sim_right_pos.detach().cpu().numpy(),
          end=cmd_right_pos.detach().cpu().numpy(),
          color=_RIGHT_ERROR_COLOR,
          width=viz.error_arrow_width,
          label=f"cmd_vs_sim_right_wrist_pos_{batch}",
        )

      # NPZ body-placement target is shoulder-mid XY + shoulder-derived yaw.
      # Z below is display-only: mean desired shoulder height.
      cmd_shoulder_mid_proxy = torch.stack(
        (
          self.cmd_shoulder_mid_xy_w[batch, 0],
          self.cmd_shoulder_mid_xy_w[batch, 1],
          env_origin[2]
          + 0.5
          * (
            self.cmd_left_shoulder_height[batch]
            + self.cmd_right_shoulder_height[batch]
          ),
        )
      )

      if viz.show_command_frames:
        visualizer.add_frame(
          position=cmd_shoulder_mid_proxy.detach().cpu().numpy(),
          rotation_matrix=cmd_shoulder_rot[batch].detach().cpu().numpy(),
          scale=viz.command_shoulder_frame_scale,
          axis_radius=viz.axis_radius,
          label=f"cmd_shoulder_mid_frame_{batch}",
          axis_colors=_CMD_FRAME_COLORS,
        )

      if viz.show_sim_frames:
        visualizer.add_frame(
          position=sim_shoulder_mid.detach().cpu().numpy(),
          rotation_matrix=sim_shoulder_rot[batch].detach().cpu().numpy(),
          scale=viz.sim_shoulder_frame_scale,
          axis_radius=viz.axis_radius,
          label=f"sim_shoulder_mid_frame_{batch}",
        )

      if viz.show_stabilization_body_frame:
        visualizer.add_frame(
          position=sim_stabilization_pos.detach().cpu().numpy(),
          rotation_matrix=sim_stabilization_rot[batch].detach().cpu().numpy(),
          scale=viz.stabilization_body_scale,
          axis_radius=viz.axis_radius,
          alpha=0.65,
          label=f"sim_{self.cfg.stabilization_body_name}_{batch}",
        )

      if viz.show_error_arrows:
        cmd_xy_at_sim_z = sim_shoulder_mid.clone()
        cmd_xy_at_sim_z[:2] = self.cmd_shoulder_mid_xy_w[batch]
        visualizer.add_arrow(
          start=sim_shoulder_mid.detach().cpu().numpy(),
          end=cmd_xy_at_sim_z.detach().cpu().numpy(),
          color=_SHOULDER_MID_ERROR_COLOR,
          width=viz.error_arrow_width,
          label=f"cmd_vs_sim_shoulder_mid_xy_{batch}",
        )

      if viz.show_shoulder_heights:
        for side, sim_shoulder_pos, cmd_height in (
          (
            "left",
            self.sim_left_shoulder_pos_w[batch],
            self.cmd_left_shoulder_height[batch],
          ),
          (
            "right",
            self.sim_right_shoulder_pos_w[batch],
            self.cmd_right_shoulder_height[batch],
          ),
        ):
          target = sim_shoulder_pos.clone()
          target[2] = env_origin[2] + cmd_height

          visualizer.add_sphere(
            center=target.detach().cpu().numpy(),
            radius=viz.shoulder_target_radius,
            color=_SHOULDER_HEIGHT_TARGET_COLOR,
            label=f"cmd_{side}_shoulder_height_{batch}",
          )

          if viz.show_error_arrows:
            visualizer.add_arrow(
              start=sim_shoulder_pos.detach().cpu().numpy(),
              end=target.detach().cpu().numpy(),
              color=_SHOULDER_HEIGHT_ERROR_COLOR,
              width=viz.shoulder_error_arrow_width,
              label=f"cmd_vs_sim_{side}_shoulder_height_{batch}",
            )


@dataclass(kw_only=True)
class SparseWholeBodyCommandCfg(CommandTermCfg):
  # Preferred training mode: recursively scan a directory tree.
  command_dir: str = ""
  command_dir_env_var: str = "UNITREE_TELEOP_COMMAND_DIR"

  # Backward-compatible single-file mode for debugging/play.
  command_file: str = ""
  command_file_env_var: str = "UNITREE_TELEOP_COMMAND_FILE"

  recursive_scan: bool = True
  skip_invalid_files: bool = True
  motion_sampling_weight_mode: Literal["uniform", "duration"] = "uniform"
  fixed_motion_id: int | None = None

  entity_name: str = "robot"

  stabilization_body_name: str = "torso_link"
  left_wrist_body_name: str = "left_wrist_yaw_link"
  right_wrist_body_name: str = "right_wrist_yaw_link"
  left_shoulder_body_name: str = "left_shoulder_roll_link"
  right_shoulder_body_name: str = "right_shoulder_roll_link"

  sampling_mode: Literal["start", "uniform"] = "start"
  loop: bool = False
  canonicalize_heading: bool = False

  # Pre-roll from the post-reset simulated task pose to the aligned NPZ target.
  # Set to 0.0 to recover the previous hard-start behavior.
  warmup_duration_s: float = 0.8
  warmup_profile: Literal["linear", "smoothstep"] = "smoothstep"

  @dataclass
  class VizCfg:
    show_command_world: bool = True
    show_command_frames: bool = True
    show_sim_frames: bool = True
    show_stabilization_body_frame: bool = True
    show_error_arrows: bool = True
    show_shoulder_heights: bool = True

    command_world_scale: float = 0.25
    command_wrist_scale: float = 0.12
    sim_wrist_scale: float = 0.09
    command_shoulder_frame_scale: float = 0.16
    sim_shoulder_frame_scale: float = 0.13
    stabilization_body_scale: float = 0.10

    axis_radius: float = 0.007
    error_arrow_width: float = 0.012
    shoulder_error_arrow_width: float = 0.008
    shoulder_target_radius: float = 0.025

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> SparseWholeBodyCommand:
    return SparseWholeBodyCommand(self, env)
