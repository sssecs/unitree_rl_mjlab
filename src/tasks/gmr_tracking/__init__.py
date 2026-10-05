"""Train a teacher on one selected GMR G1 reference clip."""

import os
from pathlib import Path

import numpy as np
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.kneel_tracking.env_cfg import kneel_env_cfg, kneel_ppo_cfg


def gmr_env_cfg(play=False):
  path = Path(os.environ.get(
    "UNITREE_GMR_MOTION_FILE", str(Path(__file__).parent / "data" / "pico_50hz.npz")
  )).expanduser().resolve()
  with np.load(path, allow_pickle=False) as data:
    if "times" not in data or "body_pos_w" not in data:
      raise ValueError("UNITREE_GMR_MOTION_FILE must point to a converted FK archive")
    duration = float(data["times"][-1])
  return kneel_env_cfg(
    play=play, motion_file=path, command_name="reference",
    episode_length_s=duration + 0.2, include_contact_targets=False,
  )


def gmr_ppo_cfg():
  cfg = kneel_ppo_cfg()
  cfg.experiment_name = "g1_gmr_tracking"
  return cfg


register_mjlab_task(
  task_id="Unitree-G1-GMR-Tracking",
  env_cfg=gmr_env_cfg(), play_env_cfg=gmr_env_cfg(play=True), rl_cfg=gmr_ppo_cfg(),
)
