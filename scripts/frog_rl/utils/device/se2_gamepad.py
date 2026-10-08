"""Gamepad controller for SE(2) velocity commands.

Ported from Isaac Lab's ``isaaclab.devices.gamepad.se2_gamepad.Se2Gamepad``,
replacing Omniverse ``carb.input`` with a pygame joystick backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .device_base import DeviceBase, DeviceCfg


class Se2Gamepad(DeviceBase):
  r"""Gamepad controller that emits SE(2) velocity commands.

  Command is base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

  Stick bindings (same as Isaac Lab)::

    ====================== ========================= ========================
    Command                Stick (+ve axis)          Stick (-ve axis)
    ====================== ========================= ========================
    Move along x-axis      left stick up             left stick down
    Move along y-axis      left stick right          left stick left
    Rotate along z-axis    right stick right         right stick left
    ====================== ========================= ========================

  Continuous pygame axes are converted into Isaac Lab's positive/negative raw
  buffer so ``advance()`` semantics match the Omniverse implementation.
  """

  def __init__(self, cfg: Se2GamepadCfg, backend: Any | None = None) -> None:
    self.v_x_sensitivity = cfg.v_x_sensitivity
    self.v_y_sensitivity = cfg.v_y_sensitivity
    self.omega_z_sensitivity = cfg.omega_z_sensitivity
    self.dead_zone = cfg.dead_zone
    self._sim_device = cfg.sim_device

    if backend is None:
      from .pygame_backend import WindowlessGamepadBackend

      backend = WindowlessGamepadBackend(dead_zone=self.dead_zone)
    self._backend = backend

    # (positive, negative) x (x, y, yaw) — same layout as Isaac Lab.
    self._base_command_raw = np.zeros([2, 3], dtype=np.float64)
    self._additional_callbacks: dict[str, Callable] = {}
    self._prev_buttons: dict[str, bool] = {}

  def __str__(self) -> str:
    msg = f"Gamepad Controller for SE(2): {self.__class__.__name__}\n"
    msg += "\t----------------------------------------------\n"
    msg += "\tMove in X-Y plane: left stick\n"
    msg += "\tRotate in Z-axis: right stick\n"
    return msg

  def reset(self) -> None:
    self._base_command_raw.fill(0.0)

  def add_callback(self, key: str, func: Callable) -> None:
    """Bind an extra callback to a named gamepad button (``A``, ``B``, ...)."""
    self._additional_callbacks[key] = func

  def advance(self) -> torch.Tensor:
    """Poll the stick axes and return ``[vx, vy, wz]``."""
    self._backend.pump()
    axes = self._backend.get_gamepad_axes()
    self._base_command_raw.fill(0.0)
    if axes is not None:
      left_x, left_y, right_x, _right_y = axes

      def _apply(value: float, axis: int, sensitivity: float) -> None:
        if abs(value) < self.dead_zone:
          return
        # direction 0 = positive contribution, 1 = negative (Isaac Lab layout).
        direction = 0 if value >= 0.0 else 1
        self._base_command_raw[direction, axis] = abs(value) * sensitivity

      # left_y (+up) -> +vx; left_x (+right) -> +vy; right_x (+right) -> +yaw.
      _apply(left_y, 0, self.v_x_sensitivity)
      _apply(left_x, 1, self.v_y_sensitivity)
      _apply(right_x, 2, self.omega_z_sensitivity)

    buttons = self._backend.get_gamepad_buttons()
    for name, pressed in buttons.items():
      was_pressed = self._prev_buttons.get(name, False)
      if pressed and not was_pressed:
        callback = self._additional_callbacks.get(name)
        if callback is not None:
          callback()
    self._prev_buttons = dict(buttons)

    return torch.tensor(
      self._resolve_command_buffer(self._base_command_raw),
      dtype=torch.float32,
      device=self._sim_device,
    )

  def close(self) -> None:
    close_fn = getattr(self._backend, "close", None)
    if close_fn is not None:
      close_fn()

  @staticmethod
  def _resolve_command_buffer(raw_command: np.ndarray) -> np.ndarray:
    """Resolve Isaac Lab's (2, 3) raw stick buffer into a (3,) command."""
    command_sign = raw_command[1, :] > raw_command[0, :]
    command = raw_command.max(axis=0)
    command[command_sign] *= -1
    return command


@dataclass
class Se2GamepadCfg(DeviceCfg):
  """Configuration for :class:`Se2Gamepad`."""

  v_x_sensitivity: float = 1.0
  v_y_sensitivity: float = 1.0
  omega_z_sensitivity: float = 1.0
  dead_zone: float = 0.01
  class_type: type[DeviceBase] = Se2Gamepad
