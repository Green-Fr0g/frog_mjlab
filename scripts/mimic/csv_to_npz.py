"""Replay a CSV motion in mjlab and export it as an NPZ.

This version is config-driven, mirroring
``frog_lab/scripts/mimic/csv_to_npz_frog.py``:

- motion interpretation comes from ``motion_data/config/<robot>.yaml``
- robot selection comes from a small registry in this script

The exported NPZ stores the *same data contract* as frog_lab, so the two
repositories' NPZ files are interchangeable::

    fps, robot_name, joint_names, body_names, root_link_name,
    joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

Example:
    python scripts/mimic/csv_to_npz.py \
        --config motion_data/config/g1.yaml \
        --output_name source/frog_mjlab/frog_mjlab/tasks/mimic/config/g1/motions/dance1_subject2.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)

parser = argparse.ArgumentParser(
  description="Replay motion from csv file and output to npz file (mjlab backend)."
)
parser.add_argument(
  "--config",
  type=str,
  default="motion_data/config/g1.yaml",
  help="Motion config yaml.",
)
parser.add_argument(
  "--csv_path",
  type=str,
  default=None,
  help="Optional override for motion_data.csv_path. Used by the batch converter.",
)
parser.add_argument(
  "--frame_range",
  nargs=2,
  type=int,
  metavar=("START", "END"),
  help=(
    "Frame range: START END (both inclusive). The frame index starts from 1. "
    "If not provided, all frames will be loaded."
  ),
)
parser.add_argument(
  "--output_name",
  type=str,
  required=True,
  help="Output path (or file name) of the motion npz file.",
)
parser.add_argument(
  "--output_dir",
  type=str,
  default=None,
  help="Optional directory prepended when --output_name is not absolute.",
)
parser.add_argument("--output_fps", type=float, default=50.0, help="The fps of the output motion.")
parser.add_argument(
  "--device",
  type=str,
  default=None,
  help="Device to run on. Defaults to cuda:0 when available, otherwise cpu.",
)
args_cli = parser.parse_args()

# mjlab backend imports (kept after argparse so --help stays cheap).
from frog_mjlab.tasks.mimic.config.g1.env_cfgs import unitree_g1_flat_mimic_env_cfg
from frog_mjlab.tasks.mimic.config.g1_23dof.env_cfgs import (
  unitree_g1_23dof_flat_mimic_env_cfg,
)

# ---------------------------------------------------------------------------
# Robot registry
# ---------------------------------------------------------------------------
# Aligned with frog_lab's ``_get_robot_asset_cfg``: ``robot_name`` selects the
# robot. In mjlab the robot lives inside the mimic env cfg's scene, so each
# entry points at the env cfg factory and we take its ``.scene``.
_ROBOT_REGISTRY = {
  "g1": unitree_g1_flat_mimic_env_cfg,
  "g1_23": unitree_g1_23dof_flat_mimic_env_cfg,  # frog_lab alias
  "g1_23dof": unitree_g1_23dof_flat_mimic_env_cfg,  # frog_mjlab name
}


def _get_robot_scene_cfg(robot_name: str):
  """Resolve the scene configuration for ``robot_name`` via the registry."""
  try:
    factory = _ROBOT_REGISTRY[robot_name]
  except KeyError as exc:
    raise KeyError(
      f"Unknown robot_name '{robot_name}'. Supported: {sorted(_ROBOT_REGISTRY)}"
    ) from exc
  return factory().scene


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
  with path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict):
    raise ValueError(f"Invalid yaml structure in: {path}")
  return data


def _resolve_path(base_dir: Path, maybe_path: str) -> Path:
  path = Path(maybe_path)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def _resolve_output_path(output_name: str, output_dir: str | None) -> Path:
  path = Path(output_name)
  if not path.is_absolute() and output_dir is not None:
    path = Path(output_dir) / path
  if not str(path).endswith(".npz"):
    path = Path(str(path) + ".npz")
  path.parent.mkdir(parents=True, exist_ok=True)
  return path


def _root_quat_to_wxyz(quat: torch.Tensor, order: str) -> torch.Tensor:
  if order == "wxyz":
    return quat
  if order == "xyzw":
    return quat[:, [3, 0, 1, 2]]
  raise ValueError(f"Unsupported root quaternion order: {order}")


# ---------------------------------------------------------------------------
# Motion loading
# ---------------------------------------------------------------------------


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: float,
    output_fps: float,
    device: torch.device | str,
    frame_range: tuple[int, int] | None,
    root_quat_order: str,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.frame_range = frame_range
    self.root_quat_order = root_quat_order
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
    if self.frame_range is None:
      motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
    else:
      motion = torch.from_numpy(
        np.loadtxt(
          self.motion_file,
          delimiter=",",
          skiprows=self.frame_range[0] - 1,
          max_rows=self.frame_range[1] - self.frame_range[0] + 1,
        )
      )
    motion = motion.to(torch.float32).to(self.device)
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = _root_quat_to_wxyz(motion[:, 3:7], self.root_quat_order)
    self.motion_dof_poss_input = motion[:, 7:]

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt
    print(
      f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, "
      f"frames: {self.input_frames}"
    )

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B-2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B-2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """Gets the next state of the motion."""
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def run_simulator(
  sim: Simulation,
  scene: Scene,
  motion: MotionLoader,
  csv_joint_names: list[str],
  output_name: str,
  root_link_name: str,
  robot_name: str,
  output_fps: float,
):
  robot: Entity = scene["robot"]

  if root_link_name not in robot.body_names:
    raise ValueError(
      f"root_link_name '{root_link_name}' not found in robot bodies: {robot.body_names}"
    )

  robot_joint_indexes = robot.find_joints(csv_joint_names, preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": [output_fps],
    # Metadata makes the npz self-describing (and interchangeable with frog_lab).
    "robot_name": [robot_name],
    "joint_names": np.array(robot.joint_names),
    "body_names": np.array(robot.body_names),
    "root_link_name": [root_link_name],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False

  print(f"\nStarting simulation with {motion.output_frames} frames...")

  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam.
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)
        np.savez(output_name, **log)  # type: ignore[arg-type]
        print(f"[INFO]: Motion saved to: {output_name}")


def main():
  config_path = Path(args_cli.config).resolve()
  config = _load_yaml(config_path)
  motion_cfg = config["motion_data"]

  robot_name = str(motion_cfg["robot_name"])
  csv_path_value = args_cli.csv_path if args_cli.csv_path is not None else motion_cfg["csv_path"]
  if not isinstance(csv_path_value, str):
    raise TypeError(
      "csv_path must be a string. For multiple paths, use scripts/mimic/batch_csv_to_npz.py."
    )
  csv_path = _resolve_path(config_path.parent, csv_path_value)
  input_fps = float(motion_cfg.get("csv_fps", 30))
  root_quat_order = str(motion_cfg.get("root_quat_order", "xyzw"))
  root_link_name = str(motion_cfg["root_link_name"])
  csv_joint_names = list(motion_cfg["csv_joint_names"])

  output_fps = float(args_cli.output_fps)
  device = args_cli.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(_get_robot_scene_cfg(robot_name), device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  scene.reset()
  print("[INFO]: Setup complete...")

  motion = MotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device=device,
    frame_range=tuple(args_cli.frame_range) if args_cli.frame_range is not None else None,
    root_quat_order=root_quat_order,
  )

  output_path = _resolve_output_path(args_cli.output_name, args_cli.output_dir)
  run_simulator(
    sim,
    scene,
    motion,
    csv_joint_names,
    str(output_path),
    root_link_name,
    robot_name,
    output_fps,
  )


if __name__ == "__main__":
  main()
