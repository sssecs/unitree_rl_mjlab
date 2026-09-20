"""Absolute-state teacher task for sparse whole-body teleoperation."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.teleop.mdp as mdp
from src.tasks.teleop.mdp import SparseWholeBodyCommandCfg


def make_teleop_env_cfg(
  history_length: int = 25,
) -> ManagerBasedRlEnvCfg:
  """Create the flat-ground privileged teacher environment."""
  if history_length < 1:
    raise ValueError("history_length must be >= 1")

  # -----------------------------------------------------------------------
  # Observations
  #
  # Teacher actor sees:
  #   1) absolute command target in a fixed episode/world frame
  #   2) absolute simulated task state in the same fixed frame
  #   3) explicit CURRENT target-minus-sim vector errors
  #
  # The explicit error is causal: it never indexes a future NPZ frame.
  # Observation history contains only current/past samples.
  # -----------------------------------------------------------------------

  actor_terms = {
    "command_absolute": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "teleop"},
    ),
    "sim_task_state_absolute": ObservationTermCfg(
      func=mdp.sim_task_state_absolute,
      params={"command_name": "teleop"},
    ),
    "command_wrist_lin_vel_absolute": ObservationTermCfg(
      func=mdp.command_wrist_linear_velocity_absolute,
      params={"command_name": "teleop"},
    ),
    "sim_wrist_lin_vel_absolute": ObservationTermCfg(
      func=mdp.sim_wrist_linear_velocity_absolute,
      params={"command_name": "teleop"},
    ),
    "task_error_vector": ObservationTermCfg(
      func=mdp.teleop_explicit_vector_errors,
      params={"command_name": "teleop"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "task_errors": ObservationTermCfg(
      func=mdp.teleop_tracking_errors,
      params={"command_name": "teleop"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=history_length,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=history_length,
    ),
  }

  # -----------------------------------------------------------------------
  # Actions
  # -----------------------------------------------------------------------

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override in robot-specific config.
      use_default_offset=True,
    )
  }

  # -----------------------------------------------------------------------
  # Command
  # -----------------------------------------------------------------------

  commands: dict[str, CommandTermCfg] = {
    "teleop": SparseWholeBodyCommandCfg(
      entity_name="robot",
      command_dir="",
      command_dir_env_var="UNITREE_TELEOP_COMMAND_DIR",
      command_file="",
      command_file_env_var="UNITREE_TELEOP_COMMAND_FILE",
      recursive_scan=True,
      skip_invalid_files=True,
      motion_sampling_weight_mode="uniform",
      fixed_motion_id=None,
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      canonicalize_heading=False,
      sampling_mode="start",
      warmup_duration_s=0.8,
      warmup_profile="smoothstep",
      post_motion_behavior="recover",
      recovery_duration_s=1.0,
      recovery_hold_s=0.4,
      recovery_profile="smoothstep",
      loop=False,
    )
  }

  # -----------------------------------------------------------------------
  # Reset / domain randomization.
  #
  # Random SE(2) reset is deliberate. Each episode aligns the NPZ ONCE to
  # the reset shoulder-mid XY/heading, preventing a fixed world trajectory
  # from becoming an open-loop memorization shortcut.
  # -----------------------------------------------------------------------

  events: dict[str, EventTermCfg] = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-1.0, 1.0),
          "y": (-1.0, 1.0),
          "z": (0.0, 0.0),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(4.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.35, 0.35),
          "y": (-0.35, 0.35),
          "z": (-0.2, 0.2),
          "roll": (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
          "yaw": (-0.5, 0.5),
        }
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.4, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.03, 0.03),
          1: (-0.03, 0.03),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  # -----------------------------------------------------------------------
  # Rewards: always absolute cmd-vs-sim task-space tracking.
  # -----------------------------------------------------------------------

  rewards: dict[str, RewardTermCfg] = {
    "wrist_pos_coarse": RewardTermCfg(
      func=mdp.wrist_position_coarse_exp,
      weight=3.0,
      params={"command_name": "teleop", "std": 0.20},
    ),
    "wrist_pos_fine": RewardTermCfg(
      func=mdp.wrist_position_fine_tanh,
      weight=4.0,
      params={"command_name": "teleop", "std": 0.05},
    ),
    "inter_wrist": RewardTermCfg(
      func=mdp.inter_wrist_position_tracking_exp,
      weight=0.50,
      params={"command_name": "teleop", "std": 0.10},
    ),
    "wrist_linear_velocity": RewardTermCfg(
      func=mdp.wrist_linear_velocity_tracking_l2,
      weight=-0.02,
      params={"command_name": "teleop"},
    ),
    "wrist_orientation": RewardTermCfg(
      func=mdp.wrist_orientation_tracking_exp,
      weight=1.5,
      params={"command_name": "teleop", "std": 0.50},
    ),
    "shoulder_mid_xy": RewardTermCfg(
      func=mdp.shoulder_mid_xy_tracking_exp,
      weight=0.60,
      params={
        "command_name": "teleop",
        "std": 0.20,
        "tolerance": 0.10,
      },
    ),
    "shoulder_height": RewardTermCfg(
      func=mdp.shoulder_height_tracking_exp,
      weight=0.50,
      params={
        "command_name": "teleop",
        "std": 0.12,
        "tolerance": 0.05,
      },
    ),
    "shoulder_heading": RewardTermCfg(
      func=mdp.shoulder_heading_tracking_exp,
      weight=0.30,
      params={
        "command_name": "teleop",
        "std": 0.35,
        "tolerance": 0.15,
      },
    ),

    # Only active after the recorded trajectory ends.  This makes recovery
    # converge toward the G1 nominal standing posture without constraining the
    # redundancy used during the actual human-motion tracking phase.
    "recovery_joint_posture": RewardTermCfg(
      func=mdp.recovery_joint_posture_l2,
      weight=-0.10,
      params={
        "command_name": "teleop",
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.05,
    ),
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-7,
    ),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide,
      weight=-0.15,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "quiet_feet": RewardTermCfg(
      func=mdp.quiet_feet_when_task_stable,
      weight=-0.50,
      params={
        "command_name": "teleop",
        "wrist_error_threshold": 0.04,
        "command_wrist_speed_threshold": 0.05,
        "command_shoulder_speed_threshold": 0.03,
        "command_heading_rate_threshold": 0.15,
        "base_speed_threshold": 0.12,
        "base_ang_speed_threshold": 0.35,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "undesired_ground_contacts": RewardTermCfg(
      func=mdp.undesired_ground_contacts,
      weight=-1.0,
      params={
        "sensor_name": "nonfoot_nonknee_ground_touch",
        "force_threshold": 1.0,
      },
    ),
    "termination": RewardTermCfg(
      func=mdp.is_terminated,
      weight=-200.0,
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={
        "sensor_name": "self_collision",
        "force_threshold": 10.0,
      },
    ),
  }

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(
      func=mdp.time_out,
      time_out=True,
    ),
    "command_finished": TerminationTermCfg(
      func=mdp.command_finished,
      params={
        "command_name": "teleop",
        "margin_s": 0.0,
      },
      time_out=True,
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=3.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=3.0,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=48,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=120.0,
  )
