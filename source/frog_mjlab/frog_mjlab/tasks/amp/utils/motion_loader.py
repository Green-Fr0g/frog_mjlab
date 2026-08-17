from __future__ import annotations
import numpy as np
import torch
from collections.abc import Sequence
from pathlib import Path

from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_apply_inverse,
    subtract_frame_transforms,
)


class AMPBodyStateMotionLoader:
    """Loads AMP expert states from G1 body-state ``.npz`` motion files."""

    def __init__(
        self,
        motion_files: str | Sequence[str],
        body_names: Sequence[str],
        anchor_name: str,
        all_body_names: Sequence[str],
        device: str = "cpu",
        time_between_frames: float = 0.02,
        quat_order: str = "wxyz",
    ) -> None:
        del time_between_frames
        if quat_order != "wxyz":
            raise ValueError(f"Unsupported quat_order '{quat_order}'. Expected 'wxyz'.")

        self.device = torch.device(device)
        self.body_names = tuple(body_names)
        self.anchor_name = anchor_name
        self.all_body_names = tuple(all_body_names)
        self._body_indexes = [self.all_body_names.index(name) for name in self.body_names]
        self._anchor_index = self.all_body_names.index(self.anchor_name)
        self._num_bodies = len(self._body_indexes)

        self.motion_files = self._resolve_motion_files(motion_files)
        self._states: list[torch.Tensor] = []
        self._next_states: list[torch.Tensor] = []
        self.fps = None

        for motion_file in self.motion_files:
            state = self._load_motion_state(motion_file)
            self._states.append(state[:-1])
            self._next_states.append(state[1:])

        if not self._states:
            raise RuntimeError("No AMP expert states were loaded.")

    @property
    def observation_dim(self) -> int:
        return (3 + 6 + 3 + 3) * self._num_bodies

    @property
    def state_dim(self) -> int:
        return self.observation_dim

    def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
        for batch_idx in range(num_mini_batch):
            motion_idx = batch_idx % len(self._states)
            states = self._states[motion_idx]
            next_states = self._next_states[motion_idx]
            ids = torch.randint(0, states.shape[0], (mini_batch_size,), device=self.device)
            yield states[ids], next_states[ids]

    def _load_motion_state(self, motion_file: Path) -> torch.Tensor:
        data = np.load(motion_file)
        for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            if key not in data:
                raise KeyError(f"AMP motion file '{motion_file}' is missing key '{key}'.")

        if self.fps is None and "fps" in data:
            self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])

        body_pos_w = torch.as_tensor(data["body_pos_w"], dtype=torch.float32, device=self.device)
        body_quat_w = torch.as_tensor(data["body_quat_w"], dtype=torch.float32, device=self.device)
        body_lin_vel_w = torch.as_tensor(data["body_lin_vel_w"], dtype=torch.float32, device=self.device)
        body_ang_vel_w = torch.as_tensor(data["body_ang_vel_w"], dtype=torch.float32, device=self.device)

        anchor_pos_w = body_pos_w[:, self._anchor_index]
        anchor_quat_w = body_quat_w[:, self._anchor_index]
        target_pos_w = body_pos_w[:, self._body_indexes]
        target_quat_w = body_quat_w[:, self._body_indexes]
        target_lin_vel_w = body_lin_vel_w[:, self._body_indexes]
        target_ang_vel_w = body_ang_vel_w[:, self._body_indexes]

        num_frames = target_pos_w.shape[0]
        anchor_pos_w = anchor_pos_w[:, None, :].expand(-1, self._num_bodies, -1)
        anchor_quat_w = anchor_quat_w[:, None, :].expand(-1, self._num_bodies, -1)
        body_pos_b, body_quat_b = subtract_frame_transforms(
            anchor_pos_w, anchor_quat_w, target_pos_w, target_quat_w
        )
        body_ori_b = matrix_from_quat(body_quat_b)[..., :2].reshape(num_frames, self._num_bodies, 6)
        body_lin_vel_b = quat_apply_inverse(
            target_quat_w.reshape(-1, 4), target_lin_vel_w.reshape(-1, 3)
        ).reshape(num_frames, self._num_bodies, 3)
        body_ang_vel_b = quat_apply_inverse(
            target_quat_w.reshape(-1, 4), target_ang_vel_w.reshape(-1, 3)
        ).reshape(num_frames, self._num_bodies, 3)

        return torch.cat(
            (
                body_pos_b.reshape(num_frames, -1),
                body_ori_b.reshape(num_frames, -1),
                body_lin_vel_b.reshape(num_frames, -1),
                body_ang_vel_b.reshape(num_frames, -1),
            ),
            dim=-1,
        )

    @staticmethod
    def _resolve_motion_files(motion_files: str | Sequence[str]) -> list[Path]:
        if isinstance(motion_files, (str, Path)):
            paths = [Path(motion_files)]
        else:
            paths = [Path(path) for path in motion_files]

        resolved: list[Path] = []
        for path in paths:
            path = path.expanduser().resolve()
            if path.is_file():
                if path.suffix == ".npz":
                    resolved.append(path)
            elif path.is_dir():
                resolved.extend(sorted(path.rglob("*.npz")))
            else:
                raise FileNotFoundError(f"AMP motion path does not exist: {path}")

        if not resolved:
            raise FileNotFoundError(f"No AMP motion .npz files found in: {motion_files}")
        return sorted(resolved)
