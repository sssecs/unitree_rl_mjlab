from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_g1_wrist_recovery_env_cfg
from .rl_cfg import unitree_g1_wrist_recovery_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-G1-Wrist-Recovery-Teacher",
  env_cfg=unitree_g1_wrist_recovery_env_cfg(),
  play_env_cfg=unitree_g1_wrist_recovery_env_cfg(play=True),
  rl_cfg=unitree_g1_wrist_recovery_ppo_runner_cfg(),
)
