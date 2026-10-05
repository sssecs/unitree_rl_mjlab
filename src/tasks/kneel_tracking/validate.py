"""Bounded full-reference lifecycle check; this is not a policy evaluation."""

import argparse
import json

import torch
import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def validate(device="cuda:0", num_envs=4, task="Unitree-G1-Kneel-Tracking"):
  cfg = load_env_cfg(task)
  command_name = next(iter(cfg.commands))
  cfg.scene.num_envs = num_envs
  # Isolate command lifecycle from policy quality so the whole clip is covered.
  cfg.terminations = {"time_out": cfg.terminations["time_out"]}
  with np.load(cfg.commands[command_name].motion_file, allow_pickle=False) as data:
    cfg.episode_length_s = float(data["times"][-1]) + 2.0
  cfg.sim.nan_guard.enabled = True
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    obs, _ = env.reset()
    command = env.command_manager.get_term(command_name)
    assert torch.all(command.time_steps == 0)
    actions = torch.zeros((num_envs, env.action_manager.total_action_dim), device=device)
    frames = command.motion.time_step_total
    resets = []
    original_resample = command._resample_command

    def record_reset(env_ids):
      resets.append(env_ids.clone())
      return original_resample(env_ids)

    command._resample_command = record_reset
    with torch.no_grad():
      for step in range(frames + 10):
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(reward).all(), f"Nonfinite reward at step {step}"
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert not terminated.any() and not truncated.any()
        assert torch.all(command.time_steps == min(step, frames - 1))
    assert not resets, "Clip endpoint unexpectedly reset/teleported the robot"
    assert torch.all(command.time_steps == frames - 1)
    assert all(torch.isfinite(value).all() for value in command.metrics.values())
    obs, _ = env.reset()
    assert torch.all(command.time_steps == 0)
    assert len(resets) == 1
    assert all(torch.isfinite(value).all() for value in obs.values())
    result = {
      "frames": frames, "steps": frames + 10, "num_envs": num_envs,
      "actor_dim": obs["actor"].shape[-1], "critic_dim": obs["critic"].shape[-1],
      "finite": True, "endpoint_resets": 0, "explicit_reset_frame": 0,
    }
    print(json.dumps(result, indent=2))
    return result
  finally:
    env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--task", default="Unitree-G1-Kneel-Tracking",
                      choices=("Unitree-G1-Kneel-Tracking", "Unitree-G1-GMR-Tracking"))
  args = parser.parse_args()
  validate(args.device, args.num_envs, args.task)
