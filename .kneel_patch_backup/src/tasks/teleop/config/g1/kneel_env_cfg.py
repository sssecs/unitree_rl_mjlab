from __future__ import annotations

import copy

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

import src.tasks.teleop.mdp as mdp
from src.assets.robots.unitree_g1.g1_constants import (
  HOME_KEYFRAME,
  KNEES_BENT_KEYFRAME,
)
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
  """Static single-knee environment on synthesized EgoDex+PICO clips."""
  cfg = unitree_g1_teleop_env_cfg(
    play=play,
    command_dir=command_dir,
    command_file=command_file,
    history_length=history_length,
  )

  teleop_cmd = cfg.commands["teleop"]
  assert isinstance(teleop_cmd, SparseWholeBodyCommandCfg)
  teleop_cmd.style_mode = "descriptor" if descriptor_style else "baseline"
  teleop_cmd.balance_mode = "off"

  # The synthesized clip already contains PICO resume/stand-up at the end.
  if not play:
    teleop_cmd.post_motion_behavior = "terminate"

  cfg.rewards["wrist_pos_coarse"].func = mdp.kneel_wrist_position_coarse_exp
  cfg.rewards["wrist_pos_fine"].func = mdp.kneel_wrist_position_fine_tanh
  cfg.rewards["inter_wrist"].func = mdp.kneel_inter_wrist_position_tracking_exp
  cfg.rewards["wrist_linear_velocity"].func = (
    mdp.kneel_wrist_linear_velocity_tracking_l2
  )
  cfg.rewards["wrist_orientation"].func = (
    mdp.kneel_wrist_orientation_tracking_exp
  )
  cfg.rewards["wrist_orientation"].weight = 0.75

  cfg.rewards["human_style"].func = mdp.kneel_human_style_reward
  cfg.rewards["human_style"].weight = 2.0 if descriptor_style else 0.0
  if descriptor_style:
    cfg.rewards["descriptor_knee_contact_match"] = RewardTermCfg(
      func=mdp.descriptor_knee_contact_match_reward,
      weight=1.0,
      params={
        "command_name": "teleop",
        "activation_height": 0.45,
        "activation_std": 0.08,
        "approach_std": 0.35,
        "approach_weight": 1.25,
        "contact_weight": 3.0,
        "asymmetry_weight": 0.75,
        "wrong_contact_weight": 1.0,
      },
    )

  cfg.rewards["quiet_feet"].func = mdp.kneel_quiet_feet_when_task_stable
  cfg.rewards["com_balance"].weight = 0.0

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


def _configure_kneel_acquisition_rewards(cfg) -> None:
  """K0 curriculum: first make one-knee support discoverable.

  Wrist/shoulder precision is intentionally weak here.  Once knee-contact rate
  is high, continue training with Unitree-G1-Teleop-Kneel-TargetDescriptor.
  """
  cfg.rewards["wrist_pos_coarse"].weight = 0.25
  cfg.rewards["wrist_pos_fine"].weight = 0.25
  cfg.rewards["inter_wrist"].weight = 0.0
  cfg.rewards["wrist_linear_velocity"].weight = -0.005
  cfg.rewards["wrist_orientation"].weight = 0.0

  cfg.rewards["shoulder_mid_xy"].weight = 0.0
  cfg.rewards["shoulder_height"].weight = 0.0
  cfg.rewards["shoulder_heading"].weight = 0.0

  cfg.rewards["human_style"].func = mdp.kneel_acquisition_style_reward
  cfg.rewards["human_style"].weight = 1.5

  support = cfg.rewards.get("descriptor_knee_contact_match")
  if support is None:
    cfg.rewards["descriptor_knee_contact_match"] = RewardTermCfg(
      func=mdp.descriptor_knee_contact_match_reward,
      weight=1.0,
      params={"command_name": "teleop"},
    )
    support = cfg.rewards["descriptor_knee_contact_match"]
  support.weight = 1.0
  support.params = {
    "command_name": "teleop",
    "activation_height": 0.50,
    "activation_std": 0.10,
    "approach_std": 0.38,
    "approach_weight": 1.50,
    "contact_weight": 4.00,
    "asymmetry_weight": 1.00,
    "wrong_contact_weight": 1.25,
    "asymmetry_margin": 0.03,
    "asymmetry_std": 0.07,
  }

  # Remove regularizers that directly discourage discovering a new support mode.
  cfg.rewards["quiet_feet"].weight = 0.0
  cfg.rewards["feet_slide"].weight = -0.03
  cfg.rewards["foot_orientation"].weight = -0.02
  cfg.rewards["stabilization_roll"].weight = -0.03
  cfg.rewards["recovery_joint_posture"].weight = 0.0
  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["joint_acc_l2"].weight = -1.0e-7

  # A single-knee pose can live near the normal walking soft-limit envelope.
  # Keep limits active, but do not let them dominate support acquisition.
  cfg.rewards["joint_limit"].weight = -2.0
  cfg.rewards["undesired_ground_contacts"].weight = -0.5
  cfg.rewards["self_collisions"].weight = -0.5
  cfg.rewards["termination"].weight = -30.0
  cfg.rewards["com_balance"].weight = 0.0


def unitree_g1_teleop_kneel_acquire_env_cfg(
  play: bool = False,
  command_dir: str = "",
  command_file: str = "",
  history_length: int = 25,
  bent_init: bool = True,
):
  """K0 single-knee acquisition curriculum.

  Recommended usage: train LEFT and RIGHT separately with
  UNITREE_TELEOP_KNEEL_SIDE, starting from the bent-knee reset.  The actor gets
  the target descriptor so support side and transition phase are observable.
  """
  cfg = unitree_g1_teleop_kneel_target_env_cfg(
    play=play,
    command_dir=command_dir,
    command_file=command_file,
    history_length=history_length,
  )
  _configure_kneel_acquisition_rewards(cfg)

  if bent_init:
    cfg.scene.entities["robot"].init_state = copy.deepcopy(KNEES_BENT_KEYFRAME)
  else:
    cfg.scene.entities["robot"].init_state = copy.deepcopy(HOME_KEYFRAME)
  return cfg


def unitree_g1_teleop_kneel_acquire_home_env_cfg(
  play: bool = False,
  command_dir: str = "",
  command_file: str = "",
  history_length: int = 25,
):
  """Same K0 reward, but reset from the normal HOME pose for stage-2 fine-tuning."""
  return unitree_g1_teleop_kneel_acquire_env_cfg(
    play=play,
    command_dir=command_dir,
    command_file=command_file,
    history_length=history_length,
    bent_init=False,
  )


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
