from mjlab.tasks.registry import register_mjlab_task
from frog_mjlab.tasks.amp.rl import AMPOnPolicyRunner

from .amp_env_cfgs import g1_amp_env_cfg
from .amp_rl_cfg import g1_amp_ppo_runner_cfg
from .wasabi_env_cfg import g1_wasabi_flat_env_cfg
from .wasabi_rl_cfg import g1_wasabi_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-AMP",
  env_cfg=g1_amp_env_cfg(),
  play_env_cfg=g1_amp_env_cfg(play=True),
  rl_cfg=g1_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-WASABI-Flat",
  env_cfg=g1_wasabi_flat_env_cfg(),
  play_env_cfg=g1_wasabi_flat_env_cfg(play=True),
  rl_cfg=g1_wasabi_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
