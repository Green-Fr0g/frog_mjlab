"""Unitree G1 WASABI environment configuration."""

import os
from dataclasses import dataclass

from frog_mjlab.assets import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from frog_mjlab.tasks.amp.wasabi_env_cfg import WasabiFlatEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

_ROOT_NAME = "pelvis"
_ANCHOR_NAME = "torso_link"
_SITE_NAMES = ("left_foot", "right_foot")
_FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)

_LINK_NAMES = (
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

_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

_WASABI_BODY_NAMES = (
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

_MOTION_FILES = os.path.abspath(
  os.path.join(os.path.dirname(__file__), "motions", "WalkandRun")
)


@dataclass(kw_only=True)
class G1WasabiFlatEnvCfg(WasabiFlatEnvCfg):
  """Unitree G1 flat-terrain WASABI configuration."""

  motion_files: str = _MOTION_FILES
  body_names: tuple[str, ...] = _WASABI_BODY_NAMES
  anchor_name: str = _ANCHOR_NAME
  root_name: str = _ROOT_NAME
  all_body_names: tuple[str, ...] = (_ROOT_NAME, *_LINK_NAMES)
  joint_names: tuple[str, ...] = _JOINT_NAMES

  def __post_init__(self) -> None:
    super().__post_init__()

    # Robot.
    self.scene.entities = {"robot": get_g1_robot_cfg()}

    # Simulation tuning for the G1 model.
    self.sim.njmax = 640
    self.sim.mujoco.ccd_iterations = 50
    self.sim.contact_sensor_maxmatch = 256
    self.sim.nconmax = None

    # Actions use the per-joint scale of the G1 actuators.
    joint_pos_action = self.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    # Viewer frame and command visualisation height.
    self.viewer.body_name = _ANCHOR_NAME
    twist_cmd = self.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    # Domain randomisation targets.
    self.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOM_NAMES
    self.events["base_com"].params["asset_cfg"].body_names = (_ANCHOR_NAME,)

    # Rewards.
    self.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (
      _ANCHOR_NAME,
    )
    self.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (
      _ANCHOR_NAME,
    )
    self.rewards["foot_slip"].params["asset_cfg"].site_names = _SITE_NAMES
    self.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (_ROOT_NAME,)

    # Critic body state.
    self.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (
      _ANCHOR_NAME,
    )
    self.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = (
      _WASABI_BODY_NAMES
    )
    self.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (
      _ANCHOR_NAME,
    )
    self.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = (
      _WASABI_BODY_NAMES
    )

    # Play-mode overrides.
    if self.play:
      self.episode_length_s = int(1e9)
      self.observations["actor"].enable_corruption = False
      self.events.pop("push_robot", None)
      self.curriculum = {}


def g1_wasabi_flat_env_cfg(play: bool = False) -> G1WasabiFlatEnvCfg:
  """Create Unitree G1 flat terrain WASABI configuration."""
  return G1WasabiFlatEnvCfg(play=play)
