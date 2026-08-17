"""Replay a converted motion npz in mjlab without running a policy."""

from __future__ import annotations

import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer


@dataclass(frozen=True)
class ReplayConfig:
  motion_file: str
  task: str = "Unitree-G1-Tracking"
  device: str | None = None
  num_envs: int = 1
  fps: float | None = None
  speed: float = 1.0
  frame_rate: float = 60.0
  loop: bool = True


class _ZeroPolicy:
  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self.env = env

  def __call__(self, obs) -> torch.Tensor:
    del obs
    return torch.zeros(self.env.action_space.shape, device=self.env.device)


class _Motion:
  def __init__(self, path: Path, device: str) -> None:
    data = np.load(path)
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel"):
      if key not in data:
        raise KeyError(f"Motion file '{path}' is missing key '{key}'.")

    self.fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0
    self.body_pos_w = torch.as_tensor(data["body_pos_w"], dtype=torch.float32, device=device)
    self.body_quat_w = torch.as_tensor(data["body_quat_w"], dtype=torch.float32, device=device)
    self.body_lin_vel_w = torch.as_tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
    self.body_ang_vel_w = torch.as_tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
    self.joint_pos = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.as_tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self.num_frames = int(self.joint_pos.shape[0])


def _write_motion_frame(env: ManagerBasedRlEnv, motion: _Motion, frame_id: int) -> None:
  robot = env.scene["robot"]
  env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  root_pos = motion.body_pos_w[frame_id, 0].repeat(env.num_envs, 1)
  root_pos = root_pos + env.scene.env_origins[env_ids]
  root_quat = motion.body_quat_w[frame_id, 0].repeat(env.num_envs, 1)
  root_pose = torch.cat((root_pos, root_quat), dim=-1)

  root_lin_vel = motion.body_lin_vel_w[frame_id, 0].repeat(env.num_envs, 1)
  root_ang_vel = motion.body_ang_vel_w[frame_id, 0].repeat(env.num_envs, 1)
  root_velocity = torch.cat((root_lin_vel, root_ang_vel), dim=-1)

  joint_pos = motion.joint_pos[frame_id].repeat(env.num_envs, 1)
  joint_vel = motion.joint_vel[frame_id].repeat(env.num_envs, 1)

  robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
  robot.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
  robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()


def run_replay(cfg: ReplayConfig) -> None:
  configure_torch_backends()

  motion_path = Path(cfg.motion_file).expanduser().resolve()
  if not motion_path.exists():
    raise FileNotFoundError(f"Motion file not found: {motion_path}")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = cfg.num_envs

  if "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg):
    env_cfg.commands["motion"].motion_file = str(motion_path)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  motion = _Motion(motion_path, device=device)
  fps = cfg.fps or motion.fps
  frame_dt = 1.0 / (fps * cfg.speed)
  viewer = NativeMujocoViewer(env, _ZeroPolicy(env), frame_rate=cfg.frame_rate)

  interrupted = False

  def _sigint_handler(signum, frame):
    del signum, frame
    nonlocal interrupted
    interrupted = True

  prev_handler = signal.signal(signal.SIGINT, _sigint_handler)
  frame_id = 0
  try:
    viewer.setup()
    next_frame_time = time.perf_counter()
    while viewer.is_running() and not interrupted:
      _write_motion_frame(env, motion, frame_id)
      viewer.sync_env_to_viewer()

      frame_id += 1
      if frame_id >= motion.num_frames:
        if not cfg.loop:
          break
        frame_id = 0

      next_frame_time += frame_dt
      sleep_time = next_frame_time - time.perf_counter()
      if sleep_time > 0.0:
        time.sleep(sleep_time)
      else:
        next_frame_time = time.perf_counter()
  finally:
    viewer.close()
    env.close()
    signal.signal(signal.SIGINT, prev_handler)


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import frog_mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  args = tyro.cli(
    ReplayConfig,
    config=mjlab.TYRO_FLAGS,
    description=f"Available tasks include: {', '.join(all_tasks)}",
  )
  run_replay(args)


if __name__ == "__main__":
  main()
