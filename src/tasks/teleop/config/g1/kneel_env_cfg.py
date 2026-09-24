from __future__ import annotations

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

import src.tasks.teleop.mdp as mdp
from src.tasks.teleop.mdp import SparseWholeBodyCommandCfg

from .env_cfgs import unitree_g1_teleop_env_cfg


LEFT_KNEE_SUPPORT_GEOMS = (
  "left_shin_collision",
  "left_linkage_brace_collision",
)
RIGHT_KNEE_SUPPORT_GEOMS = (
  "right_shin_collision",
  "right_linkage_brace_collision",
)


def _knee_contact_sensor(name: str, geoms: tuple[str, ...]) -> ContactSensorCfg:
  pattern = "^(" + "|".join(geoms) + ")$"
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=pattern,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


def unitree_g1_teleop_kneel_env_cfg(
  play: bool = False,
  command_dir: str = "",
  command_file: str = "",
  history_length: int = 25,
  descriptor_style: bool = True,
):
  """Static single-knee proof-of-concept environment.

  This keeps the existing absolute-world privileged teacher and PPO interface,
  but adapts rewards/sensors to the new EgoDex+PICO synthesized clips.
  """
  cfg = unitree_g1_teleop_env_cfg(
    play=play,
    command_dir=command_dir,
    command_file=command_file,
    history_length=history_length,
  )

  teleop_cmd = cfg.commands["teleop"]
  assert isinstance(teleop_cmd, SparseWholeBodyCommandCfg)
  teleop_cmd.style_mode = "descriptor" if descriptor_style else "baseline"

  # Static-kneel POC deliberately trains without the COM balance objective.
  # The runtime extension already supports knee-aware support polygons for the
  # later balance ablation, so enabling reward/reward_gate later is local.
  teleop_cmd.balance_mode = "off"

  # The synthesized clip already contains PICO resume/stand-up at the end.
  # Do not add a second learned recovery trajectory after the recorded clip.
  if not play:
    teleop_cmd.post_motion_behavior = "terminate"

  # Phase-aware task weighting comes from ``wrist_tracking_weight_scale`` in
  # the synthesized NPZ.  Old datasets transparently default to 1.0.
  cfg.rewards["wrist_pos_coarse"].func = mdp.kneel_wrist_position_coarse_exp
  cfg.rewards["wrist_pos_fine"].func = mdp.kneel_wrist_position_fine_tanh
  cfg.rewards["inter_wrist"].func = mdp.kneel_inter_wrist_position_tracking_exp
  cfg.rewards["wrist_linear_velocity"].func = (
    mdp.kneel_wrist_linear_velocity_tracking_l2
  )
  cfg.rewards["wrist_orientation"].func = (
    mdp.kneel_wrist_orientation_tracking_exp
  )

  # Position is the primary task.  Orientation during the PICO transition is
  # not independently calibrated to a G1 wrist frame, so use a weaker initial
  # orientation objective for this mechanism experiment.
  cfg.rewards["wrist_orientation"].weight = 0.75

  cfg.rewards["human_style"].func = mdp.kneel_human_style_reward
  cfg.rewards["human_style"].weight = 2.0 if descriptor_style else 0.0
  if descriptor_style:
    cfg.rewards["descriptor_knee_contact_match"] = RewardTermCfg(
      func=mdp.descriptor_knee_contact_match_reward,
      weight=1.0,
      # Begin guiding the knee descent during the descriptor's transition,
      # while remaining negligible for the standing portion of a motion.
      params={
        "command_name": "teleop",
        "activation_height": 0.35,
        "activation_std": 0.04,
        "approach_std": 0.20,
      },
    )

  cfg.rewards["quiet_feet"].func = mdp.kneel_quiet_feet_when_task_stable

  # Balance stays diagnostic/off for the first static-kneel run.
  cfg.rewards["com_balance"].weight = 0.0

  # Explicit left/right knee/shin ground sensors serve two purposes:
  # 1) episode kneeling-contact metrics;
  # 2) optional knee-aware COM support hull in later balance runs.
  left_knee = _knee_contact_sensor(
    "left_knee_ground_contact", LEFT_KNEE_SUPPORT_GEOMS
  )
  right_knee = _knee_contact_sensor(
    "right_knee_ground_contact", RIGHT_KNEE_SUPPORT_GEOMS
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (left_knee, right_knee)
  cfg.sim.contact_sensor_maxmatch = max(cfg.sim.contact_sensor_maxmatch, 160)

  return cfg


def unitree_g1_teleop_kneel_target_env_cfg(
  play: bool = False,
  command_dir: str = "",
  command_file: str = "",
  history_length: int = 25,
):
  """Kneeling teacher with the current target descriptor in actor and critic."""
  cfg = unitree_g1_teleop_kneel_env_cfg(
    play=play,
    command_dir=command_dir,
    command_file=command_file,
    history_length=history_length,
  )
  for group in ("actor", "critic"):
    cfg.observations[group].terms["target_style_descriptor"] = ObservationTermCfg(
      func=mdp.target_style_descriptor,
      params={"command_name": "teleop"},
      history_length=0,
    )
  return cfg


def unitree_g1_teleop_kneel_baseline_env_cfg(
  play: bool = False,
  command_dir: str = "",
  command_file: str = "",
  history_length: int = 25,
):
  """Command-only baseline on exactly the same synthesized trajectories."""
  return unitree_g1_teleop_kneel_env_cfg(
    play=play,
    command_dir=command_dir,
    command_file=command_file,
    history_length=history_length,
    descriptor_style=False,
  )
