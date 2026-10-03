"""Single-clip kneeling teacher, using the existing G1 collision model."""

from pathlib import Path

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from src.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from . import mdp


def kneel_env_cfg(play=False):
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  original = cfg.commands.pop("motion")
  cfg.commands["kneel"] = mdp.KneelCommandCfg(
    entity_name="robot",
    motion_file=str(Path(__file__).parent / "data" / "kneel_50hz.npz"),
    anchor_body_name=original.anchor_body_name,
    body_names=original.body_names,
    resampling_time_range=(1e9, 1e9),
    sampling_mode="start",
    pose_range={}, velocity_range={}, joint_position_range=(0.0, 0.0),
  )
  for group in cfg.observations.values():
    for term in group.terms.values():
      if term.params.get("command_name") == "motion":
        term.params["command_name"] = "kneel"
    group.terms.pop("motion_anchor_pos_b", None)
    group.terms["command"] = ObservationTermCfg(func=mdp.future_reference)
    group.terms["contact_phase"] = ObservationTermCfg(func=mdp.contact_phase)
    group.enable_corruption = False  # First validate the nominal teacher.
  for term in cfg.rewards.values():
    if term.params.get("command_name") == "motion":
      term.params["command_name"] = "kneel"
  # Let physical contacts produce a feasible approximation. Keep root height
  # broad and weak, with no exact global XY target.
  cfg.rewards.pop("motion_global_root_pos")
  cfg.rewards.pop("motion_body_lin_vel")
  cfg.rewards.pop("motion_body_ang_vel")
  cfg.rewards["joint_position"] = RewardTermCfg(func=mdp.joint_tracking, weight=0.6)
  cfg.rewards["joint_velocity"] = RewardTermCfg(
    func=mdp.joint_tracking, weight=0.2, params={"velocity": True, "std": 2.0}
  )
  cfg.rewards["root_linear_velocity"] = RewardTermCfg(func=mdp.root_velocity_tracking, weight=1.0)
  cfg.rewards["root_angular_velocity"] = RewardTermCfg(
    func=mdp.root_velocity_tracking, weight=0.2, params={"angular": True, "std": 2.0}
  )
  cfg.rewards["action_rate_l2"].weight = -0.01
  # The reference approaches hard joint limits. Preserve those limits in the
  # model, but do not penalize legal kneel poses for entering the 90% soft range.
  cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor = 1.0
  cfg.events = {}
  for term in cfg.terminations.values():
    if term.params.get("command_name") == "motion":
      term.params["command_name"] = "kneel"
  cfg.terminations["anchor_pos"].params["threshold"] = 0.5
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.5
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link",
  )
  # Knee/shin ground contacts are permitted and logged, never treated as falls.
  sensors = list(cfg.scene.sensors)
  for side in ("left", "right"):
    sensors.append(ContactSensorCfg(
      name=f"{side}_knee_ground",
      primary=ContactMatch(mode="geom", pattern=f"{side}_shin_collision", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"), reduce="netforce", num_slots=1,
    ))
  cfg.scene.sensors = tuple(sensors)
  cfg.sim.nconmax = 64
  cfg.sim.njmax = 400
  cfg.episode_length_s = 8.0
  return cfg


def kneel_ppo_cfg():
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_kneel_tracking"
  cfg.max_iterations = 20
  cfg.save_interval = 10
  return cfg
