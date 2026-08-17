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
        "policy_state_key": "wasabi_policy",
        "reference_state_key": "wasabi_reference",
        "hidden_dims": [512, 256],
        "activation": "elu",
        "normalize_input": True,
        "normalization_until": int(1e8),
        "reward_type": "log",
        "reward_coef": 1.0,
        "task_reward_weight": 1.0,
        "loss_type": "BCEWithLogitsLoss",
        "loss_coef": 1.0,
        "gradient_penalty_coef": 10.0,
        "gradient_tolerance": 0.0,
        "weight_decay_coef": 0.0,
        "logit_weight_decay_coef": 0.0,
        "discriminator_backbone_gradient_only": False,
        "discriminator_optimizer": "adamw",
        "learning_rate": 1.0e-3,
      },
    ),
    experiment_name="g1_wasabi_flat",
    logger="tensorboard",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=5000,
  )
