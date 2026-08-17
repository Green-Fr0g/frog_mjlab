from mjlab.tasks.registry import register_mjlab_task
from frog_mjlab.tasks.locomotion.rl import LocomotionOnPolicyRunner

from .env_cfgs import (
  unitree_h2_flat_env_cfg,
  unitree_h2_rough_env_cfg,
)
from .rl_cfg import unitree_h2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-H2-Rough",
  env_cfg=unitree_h2_rough_env_cfg(),
  play_env_cfg=unitree_h2_rough_env_cfg(play=True),
  rl_cfg=unitree_h2_ppo_runner_cfg(),
  runner_cls=LocomotionOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-H2-Flat",
  env_cfg=unitree_h2_flat_env_cfg(),
  play_env_cfg=unitree_h2_flat_env_cfg(play=True),
  rl_cfg=unitree_h2_ppo_runner_cfg(),
  runner_cls=LocomotionOnPolicyRunner,
)
