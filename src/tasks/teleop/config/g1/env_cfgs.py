"""Unitree G1 privileged absolute-state teleoperation teacher."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from src.tasks.teleop.mdp import SparseWholeBodyCommandCfg
from src.tasks.teleop.teleop_env_cfg import make_teleop_env_cfg


def unitree_g1_teleop_env_cfg(
  play: bool = False,
  command_dir: str = "",
  command_file: str = "",
) -> ManagerBasedRlEnvCfg:
  cfg = make_teleop_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern="pelvis",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="subtree",
      pattern="pelvis",
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  teleop_cmd = cfg.commands["teleop"]
  assert isinstance(teleop_cmd, SparseWholeBodyCommandCfg)
  teleop_cmd.command_dir = command_dir
  teleop_cmd.command_file = command_file
  teleop_cmd.stabilization_body_name = "torso_link"
  teleop_cmd.left_wrist_body_name = "left_wrist_yaw_link"
  teleop_cmd.right_wrist_body_name = "right_wrist_yaw_link"
  teleop_cmd.left_shoulder_body_name = "left_shoulder_roll_link"
  teleop_cmd.right_shoulder_body_name = "right_shoulder_roll_link"

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params[
    "asset_cfg"
  ].body_names = ("torso_link",)

  cfg.rewards["torso_upright"].params[
    "asset_cfg"
  ].body_names = ("torso_link",)

  cfg.viewer.body_name = "torso_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Deterministic initial placement is easier to inspect in the viewer.
    reset_pose = cfg.events["reset_base"].params["pose_range"]
    reset_pose["x"] = (0.0, 0.0)
    reset_pose["y"] = (0.0, 0.0)
    reset_pose["yaw"] = (0.0, 0.0)

    teleop_cmd.sampling_mode = "start"
    # Sorted recursive file order -> motion 0 is deterministic for viewer debug.
    teleop_cmd.fixed_motion_id = 0
    teleop_cmd.fixed_motion_ids = None
    # Viewer repeatedly runs: warm-up -> motion -> recover -> same motion.
    teleop_cmd.loop = False
    teleop_cmd.post_motion_behavior = "recover_then_chain"
    cfg.terminations.pop("command_finished", None)

  return cfg
