"""RL configuration for Unitree G1 WASABI locomotion task."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class RslRlWasabiAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO algorithm config with nested WASABI settings for frog_rl."""

  wasabi_cfg: dict = field(default_factory=dict)
  rnd_cfg: dict[str, Any] | None = None


@dataclass
class RslRlWasabiRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config for WASABI training."""


def g1_wasabi_ppo_runner_cfg() -> RslRlWasabiRunnerCfg:
  """Create RL runner configuration for Unitree G1 WASABI locomotion task."""
  return RslRlWasabiRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlWasabiAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.008,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="WasabiPPO",
      wasabi_cfg={
        "wasabi_policy_state_key": "wasabi_policy",
        "wasabi_reference_state_key": "wasabi_reference",
        "wasabi_discr_hidden_dims": [512, 256],
        "wasabi_discr_activation": "elu",
        "wasabi_normalize_input": True,
        "wasabi_normalization_until": int(1e8),
        "wasabi_reward_type": "log",
        "wasabi_reward_coef": 1.0,
        "wasabi_task_reward_weight": 1.0,
        "wasabi_loss_type": "BCEWithLogitsLoss",
        "wasabi_loss_coef": 1.0,
        "wasabi_grad_pen_coef": 10.0,
        "wasabi_grad_tolerance": 0.0,
        "wasabi_trunk_weight_decay": 0.0,
        "wasabi_head_weight_decay": 0.0,
        "wasabi_discriminator_optimizer": "adamw",
        "wasabi_discriminator_lr": 1.0e-3,
      },
    ),
    experiment_name="g1_wasabi_flat",
    logger="tensorboard",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=5000,
  )
