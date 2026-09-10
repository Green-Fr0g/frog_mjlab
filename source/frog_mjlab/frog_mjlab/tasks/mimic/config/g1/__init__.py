from mjlab.tasks.registry import register_mjlab_task
from frog_mjlab.tasks.mimic.rl import MotionMimicOnPolicyRunner

from .env_cfgs import unitree_g1_flat_mimic_env_cfg
from .rl_cfg import unitree_g1_mimic_ppo_runner_cfg

register_mjlab_task(
  task_id="FrogMjlab-G1-Mimic",
  env_cfg=unitree_g1_flat_mimic_env_cfg(),
  play_env_cfg=unitree_g1_flat_mimic_env_cfg(play=True),
  rl_cfg=unitree_g1_mimic_ppo_runner_cfg(),
  runner_cls=MotionMimicOnPolicyRunner,
)

register_mjlab_task(
  task_id="FrogMjlab-G1-Mimic-No-State-Estimation",
  env_cfg=unitree_g1_flat_mimic_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_mimic_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_mimic_ppo_runner_cfg(),
  runner_cls=MotionMimicOnPolicyRunner,
)
