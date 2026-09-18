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
  """Finish after the configured post-motion behavior.

  ``terminate``:
      End immediately after the selected NPZ reaches its final frame.
  ``recover``:
      End only after the return-to-neutral blend + hold completes.
  ``recover_then_chain``:
      Never end because of the command; the command term samples another clip
      after recovery and the environment continues until another termination.
  """
  command = cast(
    SparseWholeBodyCommand,
    env.command_manager.get_term(command_name),
  )

  if command.cfg.post_motion_behavior == "recover_then_chain":
    return torch.zeros(
      command.num_envs, dtype=torch.bool, device=command.device
    )

  if command.cfg.post_motion_behavior == "recover":
    return command.recovery_done

  duration = command.current_motion_duration
  threshold = torch.clamp(duration - margin_s, min=0.0)
  finished = command.command_time >= threshold
  return finished & (~command.warmup_active)
