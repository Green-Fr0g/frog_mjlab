from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class MotionResetManager:
    """Caches AMP motion frames and resets environments from sampled frames."""

    _instance: MotionResetManager | None = None

    def __init__(self) -> None:
        self._frames: dict[str, dict[str, torch.Tensor]] = {}

    @classmethod
    def get(cls) -> MotionResetManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(
        self,
        motion_dir: str,
        device: str | torch.device,
        root_name: str,
        all_body_names: tuple[str, ...],
    ) -> None:
        motion_dir = str(Path(motion_dir).expanduser().resolve())
        if root_name not in all_body_names:
            raise ValueError(f"AMP root body '{root_name}' is not in all_body_names.")
        root_index = all_body_names.index(root_name)
        cache_key = f"{motion_dir}:{root_index}:{tuple(all_body_names)}"
        if cache_key in self._frames:
            return

        files = self._collect_motion_files(motion_dir)
        if not files:
            raise FileNotFoundError(f"No AMP motion .npz files found in: {motion_dir}")

        frame_lists: dict[str, list[torch.Tensor]] = {
            "root_pos": [],
            "root_quat": [],
            "root_lin_vel": [],
            "root_ang_vel": [],
            "joint_pos": [],
            "joint_vel": [],
        }
        for file in files:
            data = np.load(file)
            for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel"):
                if key not in data:
                    raise KeyError(f"AMP motion file '{file}' is missing key '{key}'.")

            body_pos_w = data["body_pos_w"]
            if body_pos_w.shape[1] != len(all_body_names):
                raise ValueError(
                    f"AMP motion file '{file}' has {body_pos_w.shape[1]} bodies; "
                    f"expected {len(all_body_names)}."
                )
            frame_lists["root_pos"].append(
                torch.as_tensor(body_pos_w[:, root_index, :], device=device, dtype=torch.float32)
            )
            frame_lists["root_quat"].append(
                torch.as_tensor(data["body_quat_w"][:, root_index, :], device=device, dtype=torch.float32)
            )
            frame_lists["root_lin_vel"].append(
                torch.as_tensor(data["body_lin_vel_w"][:, root_index, :], device=device, dtype=torch.float32)
            )
            frame_lists["root_ang_vel"].append(
                torch.as_tensor(data["body_ang_vel_w"][:, root_index, :], device=device, dtype=torch.float32)
            )
            frame_lists["joint_pos"].append(torch.as_tensor(data["joint_pos"], device=device, dtype=torch.float32))
            frame_lists["joint_vel"].append(torch.as_tensor(data["joint_vel"], device=device, dtype=torch.float32))

        self._frames[cache_key] = {key: torch.cat(value, dim=0) for key, value in frame_lists.items()}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        motion_dir: str,
        root_name: str,
        all_body_names: tuple[str, ...],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        motion_dir = str(Path(motion_dir).expanduser().resolve())
        root_index = all_body_names.index(root_name)
        cache_key = f"{motion_dir}:{root_index}:{tuple(all_body_names)}"
        if cache_key not in self._frames:
            self.init(motion_dir, env.device, root_name, all_body_names)

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

        if len(env_ids) == 0:
            return

        self._write_reset_state(env, env_ids, self._frames[cache_key], asset_cfg)

    def _write_reset_state(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        frames: dict[str, torch.Tensor],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        total_frames = frames["root_pos"].shape[0]
        num_reset = len(env_ids)
        idx = torch.randint(0, total_frames, (num_reset,), device=env.device)

        asset: Entity = env.scene[asset_cfg.name]

        # --- Root pose ---
        root_pos = frames["root_pos"][idx]
        root_quat = frames["root_quat"][idx]
        root_pos = root_pos.clone()
        root_pos[:, :2] += env.scene.env_origins[env_ids, :2]
        root_pos[:, 2] += env.scene.env_origins[env_ids, 2]

        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

        # --- Root velocity ---
        root_vel = torch.cat([frames["root_lin_vel"][idx], frames["root_ang_vel"][idx]], dim=-1)
        asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        # --- Joint state ---
        joint_pos = frames["joint_pos"][idx]
        joint_vel = frames["joint_vel"][idx]

        soft_joint_pos_limits = asset.data.soft_joint_pos_limits
        assert soft_joint_pos_limits is not None
        joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
        joint_pos_clamped = joint_pos[:, asset_cfg.joint_ids].clamp_(
            joint_pos_limits[..., 0], joint_pos_limits[..., 1]
        )

        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, list):
            joint_ids = torch.tensor(joint_ids, device=env.device)

        asset.write_joint_state_to_sim(
            joint_pos_clamped,
            joint_vel[:, asset_cfg.joint_ids],
            env_ids=env_ids,
            joint_ids=joint_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_motion_files(motion_dir: str) -> list[Path]:
        path = Path(motion_dir)
        if path.is_file() and path.suffix == ".npz":
            return [path]
        return sorted(path.rglob("*.npz"))


# ------------------------------------------------------------------
# Event callback wrappers (thin delegates to singleton)
# ------------------------------------------------------------------

def init_motion_loader(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    root_name: str,
    all_body_names: tuple[str, ...],
) -> None:
    """Startup event: load normal AMP motion data."""
    del env_ids
    MotionResetManager.get().init(
        motion_dir=motion_dir,
        device=env.device,
        root_name=root_name,
        all_body_names=all_body_names,
    )


def reset_from_motion_data(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    root_name: str,
    all_body_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset event: reset envs from random normal motion frames."""
    MotionResetManager.get().reset(
        env=env,
        env_ids=env_ids,
        motion_dir=motion_dir,
        root_name=root_name,
        all_body_names=all_body_names,
        asset_cfg=asset_cfg,
    )
