"""G1 29-DoF pure-RL wrist tracking and recovery environment."""

import math
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from src.tasks.wrist_recovery import mdp


WRISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
SHOULDERS = ("left_shoulder_pitch_link", "right_shoulder_pitch_link")
TORSO = "torso_link"
FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


def _teacher_observations() -> ObservationGroupCfg:
  terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "base_velocity_command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "twist"}
    ),
    "wrist_command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "wrists"}
    ),
    "wrist_pose_error": ObservationTermCfg(
      func=mdp.wrist_pose_error, params={"command_name": "wrists"}
    ),
    "wrist_velocity_error": ObservationTermCfg(
      func=mdp.wrist_velocity_error, params={"command_name": "wrists"}
    ),
    "feet_state": ObservationTermCfg(
      func=mdp.feet_state,
      params={"command_name": "wrists", "sensor_name": "feet_ground_contact"},
    ),
    "disturbance": ObservationTermCfg(func=mdp.teacher_disturbance),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, scale=0.05),
    "last_action": ObservationTermCfg(func=mdp.last_action),
  }
  return ObservationGroupCfg(
    terms=terms,
    concatenate_terms=True,
    enable_corruption=False,
    history_length=3,
  )


def unitree_g1_wrist_recovery_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=False)
  # This value is per process. Four-GPU DDP therefore uses 4,096 envs globally.
  cfg.scene.num_envs = 64 if play else 1024
  cfg.episode_length_s = 12.0

  teacher = _teacher_observations()
  cfg.observations = {"actor": teacher, "critic": _teacher_observations()}

  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.rel_standing_envs = 1.0
  twist.heading_command = False
  twist.ranges.heading = None
  twist.ranges.lin_vel_x = (0.0, 0.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  cfg.commands["twist"] = mdp.ClutchedVelocityCommandCfg(
    **{f.name: getattr(twist, f.name) for f in fields(twist)}
  )
  cfg.commands["wrists"] = mdp.BimanualWristCommandCfg(
    entity_name="robot",
    wrist_body_names=WRISTS,
    foot_body_names=FEET,
    shoulder_body_names=SHOULDERS,
    torso_body_name=TORSO,
    resampling_time_range=(1.0e9, 1.0e9),
    reach_probability=0.5,
    asymmetric_probability=0.0,
    extension_range=(0.12, 0.28),
    lateral_offset_range=(-0.10, 0.10),
    vertical_offset_range=(-0.06, 0.06),
    orientation_angle_range=(-0.25, 0.25),
    reach_delay_s=1.0,
    reach_duration_s=2.0,
    curriculum_warmup_steps=30_000,
    curriculum_ramp_steps=60_000,
  )

  nonfoot_ground = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=FOOT_GEOMS,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (nonfoot_ground,)

  cfg.events.pop("push_robot", None)
  cfg.events["hand_payload"] = EventTermCfg(
    func=mdp.randomize_hand_wrench,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=WRISTS, preserve_order=True
      ),
      "force_ranges": {"x": (-2.0, 2.0), "y": (-1.0, 1.0), "z": (-12.0, -4.0)},
      "torque_ranges": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.2, 0.2)},
      "curriculum_warmup_steps": 30_000,
      "curriculum_ramp_steps": 60_000,
    },
  )
  cfg.events["push_robot"] = EventTermCfg(
    func=mdp.push_with_recorded_velocity,
    mode="interval",
    interval_range_s=(2.0, 5.0),
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "velocity_range": {
        "x": (-0.6, 0.6),
        "y": (-0.6, 0.6),
        "z": (-0.1, 0.1),
        "roll": (-0.2, 0.2),
        "pitch": (-0.2, 0.2),
        "yaw": (-0.3, 0.3),
      },
      "curriculum_warmup_steps": 30_000,
      "curriculum_ramp_steps": 60_000,
    },
  )

  robot = SceneEntityCfg("robot")
  feet_slide_cfg = SceneEntityCfg(
    "robot", body_names=FEET, preserve_order=True
  )
  quiet_feet_cfg = SceneEntityCfg(
    "robot", body_names=FEET, preserve_order=True
  )
  cfg.rewards = {
    "wrist_pos_coarse": RewardTermCfg(
      func=mdp.wrist_position_error_exp,
      weight=3.0,
      params={"command_name": "wrists", "std": 0.20},
    ),
    "wrist_pos_fine": RewardTermCfg(
      func=mdp.wrist_position_error_tanh,
      weight=4.0,
      params={"command_name": "wrists", "std": 0.05},
    ),
    "height_wrist_pos": RewardTermCfg(
      func=mdp.height_wrist_position_error_huber,
      weight=-8.0,
      params={"command_name": "wrists", "delta": 0.05},
    ),
    "wrist_ori": RewardTermCfg(
      func=mdp.wrist_orientation_error_exp,
      weight=1.0,
      params={"command_name": "wrists", "std": 0.5},
    ),
    "inter_wrist": RewardTermCfg(
      func=mdp.inter_wrist_position_error_exp,
      weight=1.0,
      params={"command_name": "wrists", "std": 0.10},
    ),
    "wrist_velocity": RewardTermCfg(
      func=mdp.wrist_velocity_error_l2,
      weight=-0.03,
      params={"command_name": "wrists"},
    ),
    "track_lin_vel": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_ang_vel": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "alive": RewardTermCfg(func=mdp.is_alive, weight=2.0),
    "termination": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "flat_orientation": RewardTermCfg(
      func=mdp.flat_orientation_l2, weight=-3.0, params={"asset_cfg": robot}
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height_when_shoulder_inactive_l2,
      weight=-5.0,
      params={
        "command_name": "wrists",
        "target_height": 0.78,
        "asset_cfg": robot,
      },
    ),
    "shoulder_height": RewardTermCfg(
      func=mdp.shoulder_height_tracking_exp,
      weight=3.0,
      params={"command_name": "wrists", "std": 0.08},
    ),
    "height_shoulder_dense": RewardTermCfg(
      func=mdp.height_shoulder_error_huber,
      weight=-6.0,
      params={"command_name": "wrists", "delta": 0.05},
    ),
    "backward_lean": RewardTermCfg(
      func=mdp.torso_backward_lean_l2,
      weight=-20.0,
      params={
        "command_name": "wrists",
        "deadzone": math.sin(math.radians(5.0)),
      },
    ),
    "shoulder_level": RewardTermCfg(
      func=mdp.shoulder_height_difference_l2,
      weight=-10.0,
      params={"command_name": "wrists", "deadzone": 0.03},
    ),
    "base_horizontal_velocity": RewardTermCfg(
      func=mdp.base_horizontal_velocity_l2,
      weight=-0.5,
      params={"asset_cfg": robot},
    ),
    "base_yaw_rate": RewardTermCfg(
      func=mdp.base_yaw_rate_l2, weight=-0.5, params={"asset_cfg": robot}
    ),
    "vertical_velocity": RewardTermCfg(
      func=mdp.vertical_velocity_l2, weight=-1.0, params={"asset_cfg": robot}
    ),
    "roll_pitch_velocity": RewardTermCfg(
      func=mdp.roll_pitch_velocity_l2,
      weight=-0.05,
      params={"asset_cfg": robot},
    ),
    "joint_deviation_arms": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.1,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(r".*_shoulder_.*_joint", r".*_elbow_joint", r".*_wrist_.*"),
        )
      },
    ),
    "joint_deviation_waist": RewardTermCfg(
      func=mdp.joint_deviation_with_height_scale_l1,
      weight=-1.0,
      params={
        "command_name": "wrists",
        "active_scale": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=r"waist.*"),
      },
    ),
    "joint_deviation_hip": RewardTermCfg(
      func=mdp.joint_deviation_with_height_scale_l1,
      weight=-1.0,
      params={
        "command_name": "wrists",
        "active_scale": 0.1,
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=(r".*_hip_roll_joint", r".*_hip_yaw_joint")
        )
      },
    ),
    "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_torque": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5),
    "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.04),
    "energy": RewardTermCfg(
      func=mdp.electrical_power_cost,
      weight=-1.0e-5,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",))
      },
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide,
      weight=-0.15,
      params={"sensor_name": "feet_ground_contact", "asset_cfg": feet_slide_cfg},
    ),
    "quiet_feet": RewardTermCfg(
      func=mdp.quiet_feet_when_task_stable,
      weight=-0.5,
      params={
        "command_name": "wrists",
        "velocity_command_name": "twist",
        "error_threshold": 0.04,
        "base_speed_threshold": 0.12,
        "base_ang_speed_threshold": 0.35,
        "asset_cfg": quiet_feet_cfg,
      },
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts,
      weight=-1.0,
      params={"sensor_name": "nonfoot_ground_touch", "force_threshold": 1.0},
    ),
    "self_collisions": cfg.rewards["self_collisions"],
  }

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.35},
    ),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": 1.0}
    ),
    "backward_lean": TerminationTermCfg(
      func=mdp.excessive_backward_lean,
      params={
        "command_name": "wrists",
        "maximum_forward_axis_z": math.sin(math.radians(25.0)),
        "minimum_phase": 0.5,
      },
    ),
  }
  cfg.curriculum = {}

  if play:
    # Viser requires a positive slider range even though this teacher was trained
    # only with zero base commands. Standing probability remains 1.0, so the
    # default sampled command is still exactly zero.
    twist.ranges.lin_vel_x = (-0.1, 0.1)
    twist.ranges.lin_vel_y = (-0.1, 0.1)
    twist.ranges.ang_vel_z = (-0.1, 0.1)
    wrist_cfg = cfg.commands["wrists"]
    assert isinstance(wrist_cfg, mdp.BimanualWristCommandCfg)
    wrist_cfg.curriculum_warmup_steps = -1
    wrist_cfg.curriculum_ramp_steps = 1
    cfg.events["hand_payload"].params["curriculum_warmup_steps"] = -1
    cfg.events["hand_payload"].params["curriculum_ramp_steps"] = 1
    cfg.events["push_robot"].params["curriculum_warmup_steps"] = -1
    cfg.events["push_robot"].params["curriculum_ramp_steps"] = 1

  return cfg
