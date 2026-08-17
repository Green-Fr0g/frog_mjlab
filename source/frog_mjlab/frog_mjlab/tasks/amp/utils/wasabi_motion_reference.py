"""Motion reference state used by the WASABI task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_apply_inverse


@dataclass
class _MotionClip:
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  fps: float

  @property
  def num_frames(self) -> int:
    return self.joint_pos.shape[0]


class WasabiMotionReference:
  """Loads motion clips and exposes one current frame per environment."""

  _instances: dict[int, "WasabiMotionReference"] = {}

  def __init__(
    self,
    motion_files: str | Sequence[str],
    body_names: Sequence[str],
    anchor_name: str,
    root_name: str,
    all_body_names: Sequence[str],
    joint_names: Sequence[str],
    device: str | torch.device = "cpu",
    time_between_frames: float = 0.02,
  ) -> None:
    self.device = torch.device(device)
    self.body_names = tuple(body_names)
    self.anchor_name = anchor_name
    self.root_name = root_name
    self.all_body_names = tuple(all_body_names)
    self.joint_names = tuple(joint_names)
    self.time_between_frames = float(time_between_frames)
    self._validate_names()

    self.motion_files = self._collect_motion_files(motion_files)
    self._body_indices = tuple(self.all_body_names.index(name) for name in self.body_names)
    self._anchor_index = self.all_body_names.index(self.anchor_name)
    self._root_index = self.all_body_names.index(self.root_name)
    self._clips = [self._load_clip(path) for path in self.motion_files]
    if not self._clips:
      raise FileNotFoundError(f"No WASABI motion files found in: {motion_files}")

    self.num_envs = 0
    self.motion_indices = torch.empty(0, dtype=torch.long, device=self.device)
    self.frame_indices = torch.empty(0, dtype=torch.long, device=self.device)

  @classmethod
  def for_env(cls, env) -> "WasabiMotionReference":
    try:
      return cls._instances[id(env)]
    except KeyError as exc:
      raise RuntimeError("WASABI motion reference has not been initialized for this environment.") from exc

  @classmethod
  def initialize_for_env(
    cls,
    env,
    motion_files: str | Sequence[str],
    body_names: Sequence[str],
    anchor_name: str,
    root_name: str,
    all_body_names: Sequence[str],
    joint_names: Sequence[str],
    time_between_frames: float = 0.02,
  ) -> "WasabiMotionReference":
    reference = cls(
      motion_files=motion_files,
      body_names=body_names,
      anchor_name=anchor_name,
      root_name=root_name,
      all_body_names=all_body_names,
      joint_names=joint_names,
      device=env.device,
      time_between_frames=time_between_frames,
    )
    reference.num_envs = env.num_envs
    reference.motion_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    reference.frame_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    cls._instances[id(env)] = reference
    return reference

  def _validate_names(self) -> None:
    if len(set(self.all_body_names)) != len(self.all_body_names):
      raise ValueError("WASABI all_body_names contains duplicate names.")
    if len(set(self.body_names)) != len(self.body_names):
      raise ValueError("WASABI body_names contains duplicate names.")
    missing = [name for name in self.body_names if name not in self.all_body_names]
    if missing:
      raise ValueError(f"WASABI body_names are missing from all_body_names: {missing}")
    if self.anchor_name not in self.all_body_names:
      raise ValueError(f"WASABI anchor body '{self.anchor_name}' is not in all_body_names.")
    if self.root_name not in self.all_body_names:
      raise ValueError(f"WASABI root body '{self.root_name}' is not in all_body_names.")
    if len(set(self.joint_names)) != len(self.joint_names):
      raise ValueError("WASABI joint_names contains duplicate names.")

  @staticmethod
  def _collect_motion_files(motion_files: str | Sequence[str]) -> list[Path]:
    paths = [Path(motion_files)] if isinstance(motion_files, (str, Path)) else [Path(p) for p in motion_files]
    result: list[Path] = []
    for path in paths:
      path = path.expanduser().resolve()
      if path.is_file():
        if path.suffix != ".npz":
          raise ValueError(f"WASABI motion file must use .npz: {path}")
        result.append(path)
      elif path.is_dir():
        result.extend(sorted(path.rglob("*.npz")))
      else:
        raise FileNotFoundError(f"WASABI motion path does not exist: {path}")
    return sorted(set(result))

  def _load_clip(self, path: Path) -> _MotionClip:
    data = np.load(path)
    required = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel")
    missing = [key for key in required if key not in data]
    if missing:
      raise KeyError(f"WASABI motion file '{path}' is missing keys: {missing}")

    body_pos_w = torch.as_tensor(data["body_pos_w"], dtype=torch.float32, device=self.device)
    body_quat_w = torch.as_tensor(data["body_quat_w"], dtype=torch.float32, device=self.device)
    body_lin_vel_w = torch.as_tensor(data["body_lin_vel_w"], dtype=torch.float32, device=self.device)
    body_ang_vel_w = torch.as_tensor(data["body_ang_vel_w"], dtype=torch.float32, device=self.device)
    joint_pos = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=self.device)
    joint_vel = torch.as_tensor(data["joint_vel"], dtype=torch.float32, device=self.device)

    num_frames = joint_pos.shape[0]
    if body_pos_w.shape != (num_frames, len(self.all_body_names), 3):
      raise ValueError(
        f"WASABI motion '{path}' has body_pos_w shape {tuple(body_pos_w.shape)}; "
        f"expected ({num_frames}, {len(self.all_body_names)}, 3)."
      )
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(self.joint_names):
      raise ValueError(
        f"WASABI motion '{path}' has joint shape {tuple(joint_pos.shape)}; "
        f"expected (frames, {len(self.joint_names)})."
      )
    for name, tensor, last_dim in (
      ("body_quat_w", body_quat_w, 4),
      ("body_lin_vel_w", body_lin_vel_w, 3),
      ("body_ang_vel_w", body_ang_vel_w, 3),
    ):
      if tensor.shape != (num_frames, len(self.all_body_names), last_dim):
        raise ValueError(f"WASABI motion '{path}' has invalid {name} shape {tuple(tensor.shape)}.")

    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 1.0 / self.time_between_frames
    return _MotionClip(body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, joint_pos, joint_vel, fps)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if env_ids.numel() == 0:
      return

    motion_ids = torch.randint(len(self._clips), (env_ids.numel(),), device=self.device)
    self.motion_indices[env_ids] = motion_ids
    frame_ids = torch.zeros_like(motion_ids)
    for clip_id in motion_ids.unique().tolist():
      selected = motion_ids == clip_id
      frame_ids[selected] = torch.randint(
        self._clips[clip_id].num_frames, (int(selected.sum().item()),), device=self.device
      )
    self.frame_indices[env_ids] = frame_ids

  def advance(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    if env_ids.numel() == 0:
      return

    self.frame_indices[env_ids] += 1
    finished = []
    for clip_id in self.motion_indices[env_ids].unique().tolist():
      local = self.motion_indices[env_ids] == clip_id
      env_subset = env_ids[local]
      finished.append(env_subset[self.frame_indices[env_subset] >= self._clips[clip_id].num_frames])
    finished_ids = torch.cat(finished) if finished else env_ids[:0]
    if finished_ids.numel():
      self.reset(finished_ids)

  def _gather(self, attr: str) -> torch.Tensor:
    tensors = [getattr(clip, attr) for clip in self._clips]
    result = torch.empty((self.num_envs, *tensors[0].shape[1:]), device=self.device, dtype=tensors[0].dtype)
    for clip_id, tensor in enumerate(tensors):
      selected = self.motion_indices == clip_id
      if selected.any():
        result[selected] = tensor[self.frame_indices[selected]]
    return result

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._gather("body_pos_w")

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._gather("body_quat_w")

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._gather("body_lin_vel_w")

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._gather("body_ang_vel_w")

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._gather("joint_pos")

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._gather("joint_vel")

  def base_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      self.body_pos_w[:, self._root_index],
      self.body_quat_w[:, self._root_index],
      self.body_lin_vel_w[:, self._root_index],
      self.body_ang_vel_w[:, self._root_index],
    )

  def base_lin_vel_b(self) -> torch.Tensor:
    _, quat, velocity, _ = self.base_state()
    return quat_apply_inverse(quat, velocity)

  def base_ang_vel_b(self) -> torch.Tensor:
    _, quat, _, velocity = self.base_state()
    return quat_apply_inverse(quat, velocity)

  def projected_gravity_b(self) -> torch.Tensor:
    _, quat, _, _ = self.base_state()
    gravity = torch.zeros((self.num_envs, 3), device=self.device)
    gravity[:, 2] = -1.0
    return quat_apply_inverse(quat, gravity)

  def write_robot_state(self, env, env_ids: torch.Tensor, asset_cfg) -> None:
    asset = env.scene[asset_cfg.name]
    pos, quat, lin_vel, ang_vel = self.base_state()
    root_pos = pos[env_ids].clone() + env.scene.env_origins[env_ids]
    root_pose = torch.cat((root_pos, quat[env_ids]), dim=-1)
    root_velocity = torch.cat((lin_vel[env_ids], ang_vel[env_ids]), dim=-1)
    joint_pos = self.joint_pos[env_ids][:, asset_cfg.joint_ids]
    joint_vel = self.joint_vel[env_ids][:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
    asset.write_joint_state_to_sim(
      joint_pos.clamp(limits[..., 0], limits[..., 1]),
      joint_vel,
      joint_ids=asset_cfg.joint_ids,
      env_ids=env_ids,
    )
