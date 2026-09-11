from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _curriculum_scale(env: ManagerBasedRlEnv, warmup: int, ramp: int) -> float:
  scale = (env.common_step_counter - warmup) / ramp
  return float(max(0.0, min(1.0, scale)))


def _sample_vector(
  ranges: dict[str, tuple[float, float]],
  shape: tuple[int, ...],
  device: str,
) -> torch.Tensor:
  result = torch.empty(*shape, 3, device=device)
  for axis, key in enumerate(("x", "y", "z")):
    result[..., axis].uniform_(*ranges.get(key, (0.0, 0.0)))
  return result


def randomize_hand_wrench(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  force_ranges: dict[str, tuple[float, float]],
  torque_ranges: dict[str, tuple[float, float]],
  curriculum_warmup_steps: int,
  curriculum_ramp_steps: int,
) -> None:
  asset: Entity = env.scene[asset_cfg.name]
  scale = _curriculum_scale(
    env, curriculum_warmup_steps, curriculum_ramp_steps
  )
  num_bodies = len(asset_cfg.body_ids)
  forces = scale * _sample_vector(
    force_ranges, (len(env_ids), num_bodies), env.device
  )
  torques = scale * _sample_vector(
    torque_ranges, (len(env_ids), num_bodies), env.device
  )
  asset.write_external_wrench_to_sim(
    forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
  )
  if not hasattr(env, "_teacher_hand_wrench_w"):
    env._teacher_hand_wrench_w = torch.zeros(
      env.num_envs, num_bodies, 6, device=env.device
    )
  env._teacher_hand_wrench_w[env_ids] = torch.cat((forces, torques), dim=-1)


def push_with_recorded_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg,
  curriculum_warmup_steps: int,
  curriculum_ramp_steps: int,
) -> None:
  asset: Entity = env.scene[asset_cfg.name]
  ranges = [
    velocity_range.get(key, (0.0, 0.0))
    for key in ("x", "y", "z", "roll", "pitch", "yaw")
  ]
  delta = torch.empty(len(env_ids), 6, device=env.device)
  for axis, bounds in enumerate(ranges):
    delta[:, axis].uniform_(*bounds)
  delta *= _curriculum_scale(
    env, curriculum_warmup_steps, curriculum_ramp_steps
  )
  velocity = asset.data.root_link_vel_w[env_ids].clone() + delta
  asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
  if not hasattr(env, "_teacher_push_delta_w"):
    env._teacher_push_delta_w = torch.zeros(env.num_envs, 6, device=env.device)
  env._teacher_push_delta_w[env_ids] = delta
