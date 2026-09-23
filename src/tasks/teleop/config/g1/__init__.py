from mjlab.tasks.registry import register_mjlab_task

from src.tasks.teleop.rl import TeleopOnPolicyRunner

from .env_cfgs import unitree_g1_teleop_env_cfg
from .kneel_env_cfg import (
  unitree_g1_teleop_kneel_baseline_env_cfg,
  unitree_g1_teleop_kneel_env_cfg,
  unitree_g1_teleop_kneel_target_env_cfg,
)
from .rl_cfg import unitree_g1_teleop_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-G1-Teleop",
  env_cfg=unitree_g1_teleop_env_cfg(),
  play_env_cfg=unitree_g1_teleop_env_cfg(play=True),
  rl_cfg=unitree_g1_teleop_ppo_runner_cfg(),
  runner_cls=TeleopOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Teleop-Kneel",
  env_cfg=unitree_g1_teleop_kneel_env_cfg(),
  play_env_cfg=unitree_g1_teleop_kneel_env_cfg(play=True),
  rl_cfg=unitree_g1_teleop_ppo_runner_cfg(),
  runner_cls=TeleopOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Teleop-Kneel-Baseline",
  env_cfg=unitree_g1_teleop_kneel_baseline_env_cfg(),
  play_env_cfg=unitree_g1_teleop_kneel_baseline_env_cfg(play=True),
  rl_cfg=unitree_g1_teleop_ppo_runner_cfg(),
  runner_cls=TeleopOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Teleop-Kneel-TargetDescriptor",
  env_cfg=unitree_g1_teleop_kneel_target_env_cfg(),
  play_env_cfg=unitree_g1_teleop_kneel_target_env_cfg(play=True),
  rl_cfg=unitree_g1_teleop_ppo_runner_cfg(),
  runner_cls=TeleopOnPolicyRunner,
)
