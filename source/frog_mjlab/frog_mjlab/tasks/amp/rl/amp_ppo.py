"""AMP-PPO adapter for mjlab environments."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from frog_rl.algorithms.amp_ppo import AMPPPO
from frog_rl.algorithms.ppo import PPO


class MjlabAMPPPO(AMPPPO):
  """AMP-PPO with mjlab reward shape compatibility.

  The frog_lab AMPPPO implementation forwards AMP rewards as ``[num_envs, 1]``.
  The current frog_rl PPO timeout bootstrap path expects reward vectors shaped
  as ``[num_envs]``. Keep frog_rl unchanged and adapt only the mjlab AMP task.
  """

  def process_env_step(
    self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
  ) -> None:
    """Store AMP transition and forward vector AMP rewards to PPO."""
    if self._current_amp_state is None:
      raise RuntimeError("AMPPPO.process_env_step() must be called after act().")

    next_amp_state = obs[self.expert_state_key]
    terminal_key = f"terminal_{self.expert_state_key}s"
    if terminal_key in extras:
      reset_env_ids = (dones > 0).flatten().nonzero(as_tuple=False).flatten()
      next_amp_state = next_amp_state.clone()
      next_amp_state[reset_env_ids] = extras[terminal_key][reset_env_ids]

    self.amp_storage.insert(self._current_amp_state, next_amp_state)
    amp_reward, _ = self.discriminator.reward(
      self._current_amp_state,
      next_amp_state,
      rewards,
      self.amp_normalizer,
    )
    PPO.process_env_step(self, obs, amp_reward.reshape(-1), dones, extras)
    self._current_amp_state = next_amp_state
