"""Common WASABI environment configuration helpers."""

from __future__ import annotations

from collections.abc import Sequence

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import frog_mjlab.tasks.amp.mdp as amp_mdp


def make_wasabi_env_cfg(
  cfg: ManagerBasedRlEnvCfg,
  *,
  motion_files: str | Sequence[str],
  body_names: Sequence[str],
  anchor_name: str,
  root_name: str,
  all_body_names: Sequence[str],
  joint_names: Sequence[str],
  play: bool = False,
  time_between_frames: float = 0.02,
) -> ManagerBasedRlEnvCfg:
  """Add WASABI reference events and observations to an AMP-style env cfg."""
  joint_names = tuple(joint_names)

  cfg.events.pop("init_motion_loader", None)
  cfg.events.pop("reset_from_motion", None)
  cfg.events["init_wasabi_motion_reference"] = EventTermCfg(
    func=amp_mdp.init_wasabi_motion_reference,
    mode="startup",
    params={
      "motion_files": motion_files,
      "body_names": tuple(body_names),
      "anchor_name": anchor_name,
      "root_name": root_name,
      "all_body_names": tuple(all_body_names),
      "joint_names": joint_names,
      "time_between_frames": time_between_frames,
    },
  )
  cfg.events["reset_wasabi_motion_reference"] = EventTermCfg(
    func=amp_mdp.reset_wasabi_motion_reference,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
  )
  cfg.events["advance_wasabi_motion_reference"] = EventTermCfg(
    func=amp_mdp.advance_wasabi_motion_reference,
    mode="interval",
    interval_range_s=(time_between_frames, time_between_frames),
    params={},
  )

  cfg.observations["wasabi_policy"] = ObservationGroupCfg(
    terms={
      "projected_gravity": ObservationTermCfg(
        func=amp_mdp.projected_gravity_wasabi_policy,
        params={"asset_cfg": SceneEntityCfg("robot")},
        history_length=10,
      ),
      "joint_pos_rel": ObservationTermCfg(
        func=amp_mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
        history_length=10,
        flatten_history_dim=True,
      ),
      "joint_vel": ObservationTermCfg(
        func=amp_mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
        scale=0.05,
        history_length=10,
        flatten_history_dim=True,
      ),
      "base_lin_vel": ObservationTermCfg(
        func=amp_mdp.base_lin_vel_wasabi_policy,
        params={"asset_cfg": SceneEntityCfg("robot")},
        history_length=10,
        flatten_history_dim=True,
      ),
      "base_ang_vel": ObservationTermCfg(
        func=amp_mdp.base_ang_vel_wasabi_policy,
        params={"asset_cfg": SceneEntityCfg("robot")},
        history_length=10,
        flatten_history_dim=True,
      ),
    },
    concatenate_terms=False,
    enable_corruption=False,
  )
  cfg.observations["wasabi_reference"] = ObservationGroupCfg(
    terms={
      "projected_gravity": ObservationTermCfg(
        func=amp_mdp.projected_gravity_reference_as_state,
        params={"asset_cfg": SceneEntityCfg("robot")},
        history_length=10,
      ),
      "joint_pos_rel": ObservationTermCfg(
        func=amp_mdp.joint_pos_rel_reference_as_state,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names),
          "robot_cfg": SceneEntityCfg("robot", joint_names=joint_names),
        },
        history_length=10,
        flatten_history_dim=True,
      ),
      "joint_vel": ObservationTermCfg(
        func=amp_mdp.joint_vel_rel_reference_as_state,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names),
          "robot_cfg": SceneEntityCfg("robot", joint_names=joint_names),
        },
        scale=0.05,
        history_length=10,
        flatten_history_dim=True,
      ),
      "base_lin_vel": ObservationTermCfg(
        func=amp_mdp.base_lin_vel_reference_as_state,
        params={"asset_cfg": SceneEntityCfg("robot")},
        history_length=10,
        flatten_history_dim=True,
      ),
      "base_ang_vel": ObservationTermCfg(
        func=amp_mdp.base_ang_vel_reference_as_state,
        params={"asset_cfg": SceneEntityCfg("robot")},
        history_length=10,
        flatten_history_dim=True,
      ),
    },
    concatenate_terms=False,
    enable_corruption=False,
  )

  if play:
    cfg.events.pop("push_robot", None)

  return cfg
