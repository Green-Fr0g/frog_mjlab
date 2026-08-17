"""WASABI motion-reference event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from frog_mjlab.tasks.amp.utils.wasabi_motion_reference import WasabiMotionReference

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def init_wasabi_motion_reference(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  motion_files: str,
  body_names: tuple[str, ...],
  anchor_name: str,
  root_name: str,
  all_body_names: tuple[str, ...],
  joint_names: tuple[str, ...],
  time_between_frames: float = 0.02,
) -> None:
  del env_ids
  asset = env.scene["robot"]
  actual_body_names = tuple(getattr(asset, "body_names", ()))
  if actual_body_names and actual_body_names != tuple(all_body_names):
    raise ValueError(
      "WASABI robot/motion body order mismatch. "
      f"Robot={actual_body_names}, motion={tuple(all_body_names)}"
    )
  actual_joint_names = tuple(getattr(asset, "joint_names", ()))
  if actual_joint_names and actual_joint_names != tuple(joint_names):
    raise ValueError(
      "WASABI robot/motion joint order mismatch. "
      f"Robot={actual_joint_names}, motion={tuple(joint_names)}"
    )
  WasabiMotionReference.initialize_for_env(
    env,
    motion_files=motion_files,
    body_names=body_names,
    anchor_name=anchor_name,
    root_name=root_name,
    all_body_names=all_body_names,
    joint_names=joint_names,
    time_between_frames=time_between_frames,
  )


def reset_wasabi_motion_reference(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
) -> None:
  reference = WasabiMotionReference.for_env(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  reference.reset(env_ids)
  reference.write_robot_state(env, env_ids, asset_cfg)


def advance_wasabi_motion_reference(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
) -> None:
  del env_ids
  WasabiMotionReference.for_env(env).advance()
