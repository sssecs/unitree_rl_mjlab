from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import SparseWholeBodyCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def command_finished(
  env: ManagerBasedRlEnv,
  command_name: str,
  margin_s: float = 0.0,
) -> torch.Tensor:
  """Normal end-of-clip truncation. Configure with time_out=True."""
  command = cast(
    SparseWholeBodyCommand,
    env.command_manager.get_term(command_name),
  )
  duration = command.current_motion_duration
  threshold = torch.clamp(duration - margin_s, min=0.0)
  finished = command.command_time >= threshold
  return finished & (~command.warmup_active)
