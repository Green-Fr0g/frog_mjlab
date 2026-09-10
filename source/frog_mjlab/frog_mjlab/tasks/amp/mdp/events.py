from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _as_str_list(values) -> list[str]:
    """Convert npz name arrays to a list of strings."""
    return [str(name) for name in np.asarray(values).tolist()]


def _index_by_names(available: Sequence[str], requested: Sequence[str], kind: str) -> list[int]:
    """Map requested names onto a motion-file name list."""
    lookup = {name: index for index, name in enumerate(available)}
    missing = [name for name in requested if name not in lookup]
    if missing:
        raise ValueError(
            f"AMP motion {kind} names {missing} were not found. Available {kind} names: {list(available)}"
        )
    return [lookup[name] for name in requested]


def _select(array, indexes: Sequence[int]):
    """Reorder the columns of ``array`` by ``indexes``."""
    return np.asarray(array)[:, indexes]


def _resolve_robot_layout(
    env: ManagerBasedRlEnv,
    entity_name: str,
    all_body_names: Sequence[str],
    joint_names: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve the authoritative body/joint layout from the live robot entity.

    The robot entity is the source of truth, so no hard-coded body/joint list is
    required: motion data is reordered onto the entity's own layout.  When the
    cfg does provide names they are checked against the entity so a stale list
    fails loudly instead of silently mis-indexing the motions.
    """
    try:
        asset: Entity = env.scene[entity_name]
    except KeyError:
        return tuple(all_body_names), tuple(joint_names)

    actual_body_names = tuple(getattr(asset, "body_names", ()) or ())
    actual_joint_names = tuple(getattr(asset, "joint_names", ()) or ())

    if actual_body_names and tuple(all_body_names) and set(actual_body_names) != set(all_body_names):
        raise ValueError(
            "AMP robot/cfg body name mismatch. "
            f"Robot={actual_body_names}, cfg={tuple(all_body_names)}"
        )
    if actual_joint_names and tuple(joint_names) and set(actual_joint_names) != set(joint_names):
        raise ValueError(
            "AMP robot/cfg joint name mismatch. "
            f"Robot={actual_joint_names}, cfg={tuple(joint_names)}"
        )

    return (actual_body_names or tuple(all_body_names)), (actual_joint_names or tuple(joint_names))


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
        joint_names: tuple[str, ...],
    ) -> None:
        motion_dir = str(Path(motion_dir).expanduser().resolve())
        all_body_names = tuple(all_body_names)
        joint_names = tuple(joint_names)
        if root_name not in all_body_names:
            raise ValueError(f"AMP root body '{root_name}' is not in all_body_names.")
        if not joint_names:
            raise ValueError("AMP motion loader requires a non-empty joint layout (joint_names).")
        root_index = all_body_names.index(root_name)
        cache_key = f"{motion_dir}:{root_index}:{all_body_names}:{joint_names}"
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

            # The motion file must carry its own layout; the data is re-ordered
            # by name onto the requested layout (no positional assumptions).
            for key in ("body_names", "joint_names"):
                if key not in data:
                    raise KeyError(
                        f"AMP motion file '{file}' is missing '{key}'. "
                        "Re-export the motion with scripts/mimic/csv_to_npz.py."
                    )

            body_pos_w = np.asarray(data["body_pos_w"])
            motion_body_names = _as_str_list(data["body_names"])
            if len(motion_body_names) != body_pos_w.shape[1]:
                raise ValueError(
                    f"AMP motion file '{file}' has {len(motion_body_names)} body_names "
                    f"but body_pos_w has {body_pos_w.shape[1]} bodies."
                )
            body_indexes = _index_by_names(motion_body_names, all_body_names, "body")

            joint_pos = np.asarray(data["joint_pos"])
            joint_vel = np.asarray(data["joint_vel"])
            motion_joint_names = _as_str_list(data["joint_names"])
            if len(motion_joint_names) != joint_pos.shape[1]:
                raise ValueError(
                    f"AMP motion file '{file}' has {len(motion_joint_names)} joint_names "
                    f"but joint_pos has {joint_pos.shape[1]} joints."
                )
            joint_indexes = _index_by_names(motion_joint_names, joint_names, "joint")

            body_pos_w = _select(body_pos_w, body_indexes)
            frame_lists["root_pos"].append(
                torch.as_tensor(body_pos_w[:, root_index, :], device=device, dtype=torch.float32)
            )
            frame_lists["root_quat"].append(
                torch.as_tensor(
                    _select(data["body_quat_w"], body_indexes)[:, root_index, :], device=device, dtype=torch.float32
                )
            )
            frame_lists["root_lin_vel"].append(
                torch.as_tensor(
                    _select(data["body_lin_vel_w"], body_indexes)[:, root_index, :], device=device, dtype=torch.float32
                )
            )
            frame_lists["root_ang_vel"].append(
                torch.as_tensor(
                    _select(data["body_ang_vel_w"], body_indexes)[:, root_index, :], device=device, dtype=torch.float32
                )
            )
            frame_lists["joint_pos"].append(
                torch.as_tensor(_select(joint_pos, joint_indexes), device=device, dtype=torch.float32)
            )
            frame_lists["joint_vel"].append(
                torch.as_tensor(_select(joint_vel, joint_indexes), device=device, dtype=torch.float32)
            )

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
        joint_names: tuple[str, ...],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> None:
        motion_dir = str(Path(motion_dir).expanduser().resolve())
        all_body_names = tuple(all_body_names)
        joint_names = tuple(joint_names)
        root_index = all_body_names.index(root_name)
        cache_key = f"{motion_dir}:{root_index}:{all_body_names}:{joint_names}"
        if cache_key not in self._frames:
            self.init(motion_dir, env.device, root_name, all_body_names, joint_names)

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
    all_body_names: tuple[str, ...] = (),
    joint_names: tuple[str, ...] = (),
    entity_name: str = "robot",
) -> None:
    """Startup event: load normal AMP motion data."""
    del env_ids
    body_layout, joint_layout = _resolve_robot_layout(
        env, entity_name, all_body_names, joint_names
    )
    MotionResetManager.get().init(
        motion_dir=motion_dir,
        device=env.device,
        root_name=root_name,
        all_body_names=body_layout,
        joint_names=joint_layout,
    )


def reset_from_motion_data(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    root_name: str,
    all_body_names: tuple[str, ...] = (),
    joint_names: tuple[str, ...] = (),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset event: reset envs from random normal motion frames."""
    body_layout, joint_layout = _resolve_robot_layout(
        env, asset_cfg.name, all_body_names, joint_names
    )
    MotionResetManager.get().reset(
        env=env,
        env_ids=env_ids,
        motion_dir=motion_dir,
        root_name=root_name,
        all_body_names=body_layout,
        joint_names=joint_layout,
        asset_cfg=asset_cfg,
    )
