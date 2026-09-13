"""Explicit replay profile for the recommended Stage 5B clutch checkpoint.

Uses play-mode physics/events/terminations; commands match the trained task
with curricula fully enabled. This is visualization, not held-out evaluation.
"""


def apply_stage5b_clutch_profile(env_cfg, mode="mixed"):
  if "wrists" not in env_cfg.commands or "twist" not in env_cfg.commands:
    raise ValueError("Stage 5B clutch profile requires a wrist-recovery task")
  mixtures = {"mixed": (.25, .25), "transport": (1., 0.),
              "adjust": (0., 1.), "balance": (0., 0.)}
  if mode not in mixtures:
    raise ValueError(f"Unknown wrist mode: {mode}")
  twist = env_cfg.commands["twist"]
  wrists = env_cfg.commands["wrists"]
  twist.clutch_enabled = wrists.clutch_enabled = True
  twist.resampling_time_range = (1000., 1000.)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.init_velocity_prob = 0.
  twist.transport_probability, twist.adjust_probability = mixtures[mode]
  twist.ranges.lin_vel_x = (-.10, .10)
  twist.ranges.lin_vel_y = (-.05, .05)
  twist.ranges.ang_vel_z = (-.10, .10)
  twist.adjust_linear_speed = .04
  twist.adjust_angular_speed = .06
  twist.adjust_distance_limit = .08
  twist.adjust_yaw_limit = .12
  twist.adjust_delay_s = 3.
  twist.adjust_duration_s = 2.
  for command in (twist, wrists):
    command.curriculum_warmup_steps = -1
    command.curriculum_ramp_steps = 1
  wrists.reach_probability = .5
  wrists.asymmetric_probability = .5
  wrists.height_probability = .5
  wrists.shoulder_height_range = (.78, .98)
  wrists.wrist_height_offset_range = (-.02, .02)
  wrists.low_reach_extension_range = (.06, .14)
  print(f"[INFO] Stage 5B clutch replay: mode={mode}, curricula fully enabled")


def apply_ground_workspace_profile(env_cfg, mode="mixed"):
  """Replay ground25 training mixture or its conditional near-ground subset."""
  if mode not in ("mixed", "ground"):
    raise ValueError("Ground workspace supports --wrist-mode mixed or ground")
  apply_stage5b_clutch_profile(env_cfg, "balance")
  wrists = env_cfg.commands["wrists"]
  wrists.height_spatial_sampling = True
  wrists.low_reach_extension_range = (.02, .18)
  wrists.ground_probability = .25 if mode == "mixed" else 1.
  wrists.height_probability = .5 if mode == "mixed" else 1.
  wrists.ground_wrist_height_range = (.16, .28)
  wrists.ground_other_wrist_raise_range = (.10, .20)
  print(f"[INFO] Ground workspace replay: {mode}; zero base command; wrist z=0.16..0.28m")
