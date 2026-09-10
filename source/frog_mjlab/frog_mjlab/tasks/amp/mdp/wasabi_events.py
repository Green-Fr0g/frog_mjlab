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
  body_names: tuple[str, ...] = (),
  anchor_name: str = "",
  root_name: str = "",
  all_body_names: tuple[str, ...] = (),
  joint_names: tuple[str, ...] = (),
  time_between_frames: float = 0.02,
) -> None:
  """Load the WASABI reference motions, adopting the live robot layout.

  The robot entity is the source of truth for body/joint ordering, so motion
  data is reordered onto it by name (no hard-coded layout required).  Names
  supplied by the cfg are validated against the entity when present.
  """
  del env_ids
  asset = env.scene["robot"]
  actual_body_names = tuple(getattr(asset, "body_names", ()) or ())
  actual_joint_names = tuple(getattr(asset, "joint_names", ()) or ())
  if actual_body_names and tuple(all_body_names) and set(actual_body_names) != set(all_body_names):
    raise ValueError(
      "WASABI robot/cfg body name mismatch. "
      f"Robot={actual_body_names}, cfg={tuple(all_body_names)}"
    )
  if actual_joint_names and tuple(joint_names) and set(actual_joint_names) != set(joint_names):
    raise ValueError(
      "WASABI robot/cfg joint name mismatch. "
      f"Robot={actual_joint_names}, cfg={tuple(joint_names)}"
    )
  WasabiMotionReference.initialize_for_env(
    env,
    motion_files=motion_files,
    body_names=body_names,
    anchor_name=anchor_name,
    root_name=root_name,
    all_body_names=actual_body_names or tuple(all_body_names),
    joint_names=actual_joint_names or tuple(joint_names),
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
