"""G1 kneeling reference teacher task."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import kneel_env_cfg, kneel_ppo_cfg

register_mjlab_task(
  task_id="Unitree-G1-Kneel-Tracking",
  env_cfg=kneel_env_cfg(),
  play_env_cfg=kneel_env_cfg(play=True),
  rl_cfg=kneel_ppo_cfg(),
)
