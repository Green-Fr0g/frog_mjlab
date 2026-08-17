"""WASABI observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from frog_mjlab.tasks.amp.utils.wasabi_motion_reference import WasabiMotionReference

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _reference(env: ManagerBasedRlEnv) -> WasabiMotionReference:
  return WasabiMotionReference.for_env(env)


def _maybe_reference(env: ManagerBasedRlEnv) -> WasabiMotionReference | None:
  try:
    return _reference(env)
  except RuntimeError:
    return None


def _joint_ids(asset_cfg: SceneEntityCfg, count: int, device: torch.device) -> torch.Tensor:
  joint_ids = getattr(asset_cfg, "joint_ids", None)
  if joint_ids is None:
    return torch.arange(count, device=device)
  if isinstance(joint_ids, slice):
    return torch.arange(count, device=device)[joint_ids]
  if len(joint_ids) == 0:
    return torch.arange(count, device=device)
  return torch.as_tensor(joint_ids, device=device, dtype=torch.long)


def projected_gravity_reference_as_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  del asset_cfg
  reference = _maybe_reference(env)
  if reference is None:
    return torch.zeros(env.num_envs, 3, device=env.device)
  return reference.projected_gravity_b()


def joint_pos_rel_reference_as_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot = env.scene[robot_cfg.name]
  reference = _maybe_reference(env)
  if reference is None:
    ids = _joint_ids(asset_cfg, robot.data.default_joint_pos.shape[-1], robot.data.default_joint_pos.device)
    return torch.zeros(env.num_envs, len(ids), device=robot.data.default_joint_pos.device)
  ids = _joint_ids(asset_cfg, reference.joint_pos.shape[-1], reference.device)
  return reference.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]


def joint_vel_rel_reference_as_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot = env.scene[robot_cfg.name]
  reference = _maybe_reference(env)
  if reference is None:
    ids = _joint_ids(asset_cfg, robot.data.default_joint_vel.shape[-1], robot.data.default_joint_vel.device)
    return torch.zeros(env.num_envs, len(ids), device=robot.data.default_joint_vel.device)
  ids = _joint_ids(asset_cfg, reference.joint_vel.shape[-1], reference.device)
  return reference.joint_vel[:, ids] - robot.data.default_joint_vel[:, ids]


def base_lin_vel_reference_as_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  del asset_cfg
  reference = _maybe_reference(env)
  if reference is None:
    return torch.zeros(env.num_envs, 3, device=env.device)
  return reference.base_lin_vel_b()


def base_ang_vel_reference_as_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  del asset_cfg
  reference = _maybe_reference(env)
  if reference is None:
    return torch.zeros(env.num_envs, 3, device=env.device)
  return reference.base_ang_vel_b()


def projected_gravity_wasabi_policy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.projected_gravity_b


def base_lin_vel_wasabi_policy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_lin_vel_b


def base_ang_vel_wasabi_policy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_ang_vel_b
