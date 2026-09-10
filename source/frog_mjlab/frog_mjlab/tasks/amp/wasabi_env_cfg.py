"""WASABI flat-terrain configuration (standalone base).

Mirrors ``frog_lab/tasks/amp/wasabi_env_cfg.py``: this module owns a complete
and self-contained environment definition (``WasabiFlatEnvCfg``) instead of
deriving from the AMP configuration.  Robot-specific packages subclass it and
fill in the per-robot fields (motion files, body/joint names, ...).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import frog_mjlab.tasks.amp.mdp as mdp


def _wasabi_sensors() -> tuple[ContactSensorCfg, ContactSensorCfg]:
  """Create the foot-ground and self-collision contact sensors."""
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  return feet_ground_cfg, self_collision_cfg


def _wasabi_observations(
  joint_names: Sequence[str],
) -> dict[str, ObservationGroupCfg]:
  """Create the actor/critic plus WASABI policy and reference groups."""
  joint_names = tuple(joint_names)

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "body_pos_b": ObservationTermCfg(
      func=mdp.robot_body_pos_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "body_ori_b": ObservationTermCfg(
      func=mdp.robot_body_ori_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
  }

  wasabi_policy_terms = {
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity_wasabi_policy,
      params={"asset_cfg": SceneEntityCfg("robot")},
      history_length=10,
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
      history_length=10,
      flatten_history_dim=True,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
      scale=0.05,
      history_length=10,
      flatten_history_dim=True,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel_wasabi_policy,
      params={"asset_cfg": SceneEntityCfg("robot")},
      history_length=10,
      flatten_history_dim=True,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel_wasabi_policy,
      params={"asset_cfg": SceneEntityCfg("robot")},
      history_length=10,
      flatten_history_dim=True,
    ),
  }

  wasabi_reference_terms = {
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity_reference_as_state,
      params={"asset_cfg": SceneEntityCfg("robot")},
      history_length=10,
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=mdp.joint_pos_rel_reference_as_state,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names),
        "robot_cfg": SceneEntityCfg("robot", joint_names=joint_names),
      },
      history_length=10,
      flatten_history_dim=True,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel_reference_as_state,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=joint_names),
        "robot_cfg": SceneEntityCfg("robot", joint_names=joint_names),
      },
      scale=0.05,
      history_length=10,
      flatten_history_dim=True,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel_reference_as_state,
      params={"asset_cfg": SceneEntityCfg("robot")},
      history_length=10,
      flatten_history_dim=True,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel_reference_as_state,
      params={"asset_cfg": SceneEntityCfg("robot")},
      history_length=10,
      flatten_history_dim=True,
    ),
  }

  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=4,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=4,
    ),
    # WASABI consumes these two groups through the discriminator. Terms are
    # concatenated (term-major, history flattened) so each group yields a single
    # 2-D tensor, matching the frog_lab WASABI state layout.
    "wasabi_policy": ObservationGroupCfg(
      terms=wasabi_policy_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "wasabi_reference": ObservationGroupCfg(
      terms=wasabi_reference_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def _wasabi_events(
  *,
  motion_files: str | Sequence[str],
  body_names: Sequence[str],
  anchor_name: str,
  root_name: str,
  all_body_names: Sequence[str],
  joint_names: Sequence[str],
  time_between_frames: float,
) -> dict[str, EventTermCfg]:
  """Create the WASABI reference events plus the shared randomisation events."""
  joint_names = tuple(joint_names)

  return {
    "init_wasabi_motion_reference": EventTermCfg(
      func=mdp.init_wasabi_motion_reference,
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
    ),
    "reset_wasabi_motion_reference": EventTermCfg(
      func=mdp.reset_wasabi_motion_reference,
      mode="reset",
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names)},
    ),
    "advance_wasabi_motion_reference": EventTermCfg(
      func=mdp.advance_wasabi_motion_reference,
      mode="interval",
      interval_range_s=(time_between_frames, time_between_frames),
      params={},
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-1.0, 1.0),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }


def _wasabi_rewards() -> dict[str, RewardTermCfg]:
  """Create the task rewards (same shape and weights as the AMP task)."""
  return {
    "track_anchor_linear_velocity": RewardTermCfg(
      func=mdp.track_anchor_linear_velocity,
      weight=1.0,
      params={
        "command_name": "twist",
        "std": 1.0,
        "mask_delay": False,
        "delay_env_rew_ratio": 1.0,
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "track_anchor_angular_velocity": RewardTermCfg(
      func=mdp.track_anchor_angular_velocity,
      weight=1.0,
      params={
        "command_name": "twist",
        "std": 3.14,
        "mask_delay": False,
        "delay_env_rew_ratio": 1.0,
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "track_root_height": RewardTermCfg(
      func=mdp.track_root_height,
      weight=1.0,
      params={"std": 0.3, "mask_delay": False, "delay_env_rew_ratio": 1.0},
    ),
    "body_ang_vel_xy_l2": RewardTermCfg(
      func=mdp.body_ang_vel_xy_l2,
      weight=0.5,
      params={
        "std": 3.14,
        "mask_delay": False,
        "delay_env_rew_ratio": 1.0,
        "body_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
      },
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }


def _wasabi_terminations() -> dict[str, TerminationTermCfg]:
  """Create the termination terms."""
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "bad_base_height": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.5},
    ),
  }


def _wasabi_commands() -> dict[str, CommandTermCfg]:
  """Create the velocity command term."""
  return {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.05,
      rel_heading_envs=0.25,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.5, 3.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-3.14 / 2, 3.14 / 2),
        heading=(-math.pi / 2, math.pi / 2),
      ),
    )
  }


def _wasabi_actions() -> dict[str, ActionTermCfg]:
  """Create the joint position action term (scale is set per-robot)."""
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }


def _wasabi_metrics() -> dict[str, MetricsTermCfg]:
  """Create the metric terms."""
  return {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }


@dataclass(kw_only=True)
class WasabiFlatEnvCfg(ManagerBasedRlEnvCfg):
  """Standalone WASABI flat-terrain environment configuration.

  This class fully defines its own MDP (observations, events, rewards,
  terminations, commands, actions) and never derives from the AMP task.  Robot
  packages subclass it to fill in the per-robot fields below.
  """

  # Per-robot fields (overridden by robot-specific subclasses).
  motion_files: str | Sequence[str] = ()
  """Reference motion files (npz) or the directory containing them."""

  body_names: Sequence[str] = ()
  """Body subset used by the WASABI discriminator state."""

  anchor_name: str = ""
  """Anchor body name used by the WASABI state and tracking rewards."""

  root_name: str = ""
  """Root body name of the robot."""

  all_body_names: Sequence[str] = ()
  """All body names matching the reference motion data."""

  joint_names: Sequence[str] = ()
  """Ordered joint names used by actions and observations."""

  time_between_frames: float = 0.02
  """Time between two reference motion frames."""

  play: bool = False
  """Whether to build the play-mode variant of the configuration."""

  # Environment defaults (the parent declares ``decimation``/``scene`` without
  # defaults, so they are supplied here).
  decimation: int = 4
  episode_length_s: float = 20.0

  scene: SceneCfg = field(
    default_factory=lambda: SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      sensors=_wasabi_sensors(),
      num_envs=1,
      extent=2.0,
    )
  )

  sim: SimulationCfg = field(
    default_factory=lambda: SimulationCfg(
      nconmax=None,
      njmax=640,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    )
  )

  viewer: ViewerConfig = field(
    default_factory=lambda: ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    )
  )

  def __post_init__(self) -> None:
    self.observations = _wasabi_observations(self.joint_names)
    self.actions = _wasabi_actions()
    self.commands = _wasabi_commands()
    self.events = _wasabi_events(
      motion_files=self.motion_files,
      body_names=self.body_names,
      anchor_name=self.anchor_name,
      root_name=self.root_name,
      all_body_names=self.all_body_names,
      joint_names=self.joint_names,
      time_between_frames=self.time_between_frames,
    )
    self.rewards = _wasabi_rewards()
    self.terminations = _wasabi_terminations()
    self.metrics = _wasabi_metrics()
