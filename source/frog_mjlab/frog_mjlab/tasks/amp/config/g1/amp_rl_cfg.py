"""RL configuration for Unitree G1 AMP locomotion task."""

import os
from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# AMP motion data directory (npz files)
_MOTION_DATA_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  "motions", "WalkandRun",
)

_AMP_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

_AMP_ALL_BODY_NAMES = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)


@dataclass
class RslRlAmpAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO algorithm config with nested AMP settings for frog_rl."""

  amp_cfg: dict = field(default_factory=dict)
  rnd_cfg: dict[str, Any] | None = None


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config for the G1 AMP task."""


def g1_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create RL runner configuration for Unitree G1 AMP locomotion task."""
  return RslRlAmpRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlAmpAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="AMPPPO",
      amp_cfg={
        "amp_reward_coef": 0.1,
        "amp_replay_buffer_size": 2000000,
        "amp_discr_hidden_dims": [1024, 512],
        "amp_discr_activation": "relu",
        "discriminator_lr": 1.0e-3,
        "grad_pen_coef": 10.0,
        "amp_trunk_weight_decay": 1.0e-3,
        "amp_head_weight_decay": 1.0e-2,
        "amp_task_reward_lerp": 0.75,
        "expert_state_key": "amp",
        "motion_loader_class_name": (
          "frog_mjlab.tasks.amp.utils.motion_loader:AMPBodyStateMotionLoader"
        ),
        "motion_loader_kwargs": {
          "motion_files": os.path.normpath(_MOTION_DATA_DIR),
          "body_names": _AMP_BODY_NAMES,
          "anchor_name": "torso_link",
          "all_body_names": _AMP_ALL_BODY_NAMES,
        },
      },
    ),
    experiment_name="g1_amp_locomotion",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=100001,
  )
