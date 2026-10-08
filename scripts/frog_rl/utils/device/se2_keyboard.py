"""Keyboard controller for SE(2) velocity commands.

Ported from Isaac Lab's ``isaaclab.devices.keyboard.se2_keyboard.Se2Keyboard``,
replacing Omniverse ``carb.input`` with the MuJoCo native viewer key callback.
No pygame dependency.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch
from mjlab.viewer.native.keys import (
  KEY_DOWN,
  KEY_KP_2,
  KEY_KP_4,
  KEY_KP_6,
  KEY_KP_7,
  KEY_KP_8,
  KEY_KP_9,
  KEY_L,
  KEY_LEFT,
  KEY_RIGHT,
  KEY_UP,
  KEY_X,
  KEY_Z,
)

from .device_base import DeviceBase, DeviceCfg

# MuJoCo viewer key code -> Isaac Lab SE(2) key name.
# Note: KEY_RIGHT doubles as the native viewer's "single step" shortcut, which
# only has an effect while the viewer is paused.
MUJOCO_KEY_TO_INPUT: dict[int, str] = {
  KEY_UP: "UP",
  KEY_KP_8: "NUMPAD_8",
  KEY_DOWN: "DOWN",
  KEY_KP_2: "NUMPAD_2",
  KEY_LEFT: "LEFT",
  KEY_KP_4: "NUMPAD_4",
  KEY_RIGHT: "RIGHT",
  KEY_KP_6: "NUMPAD_6",
  KEY_Z: "Z",
  KEY_KP_7: "NUMPAD_7",
  KEY_X: "X",
  KEY_KP_9: "NUMPAD_9",
  KEY_L: "L",
}


class Se2Keyboard(DeviceBase):
  r"""Keyboard controller that emits SE(2) velocity commands.

  Command is base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

  Key bindings (same as Isaac Lab)::

    ====================== ========================= ========================
    Command                Key (+ve axis)            Key (-ve axis)
    ====================== ========================= ========================
    Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
    Move along y-axis      Numpad 4 / Arrow Left     Numpad 6 / Arrow Right
    Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
    Reset all commands     L
    ====================== ========================= ========================

  Wire :meth:`on_key` to the MuJoCo native viewer's ``key_callback``. MuJoCo
  only reports key *presses* (no releases / repeats), so each press is held for
  ``hold_timeout`` seconds and refreshed by further presses.
  """

  def __init__(self, cfg: Se2KeyboardCfg) -> None:
    self.v_x_sensitivity = cfg.v_x_sensitivity
    self.v_y_sensitivity = cfg.v_y_sensitivity
    self.omega_z_sensitivity = cfg.omega_z_sensitivity
    self._sim_device = cfg.sim_device
    self.hold_timeout = max(float(cfg.hold_timeout), 0.0)

    self._lock = Lock()
    self._pending: list[str] = []
    self._active: dict[str, float] = {}

    self._create_key_bindings()
    self._base_command = np.zeros(3, dtype=np.float64)
    self._additional_callbacks: dict[str, Callable] = {}

  def __str__(self) -> str:
    msg = f"Keyboard Controller for SE(2): {self.__class__.__name__}\n"
    msg += "\tInput: MuJoCo native viewer key callback\n"
    msg += f"\tHold timeout: {self.hold_timeout:.2f}s\n"
    msg += "\t----------------------------------------------\n"
    msg += "\tReset all commands: L\n"
    msg += "\tMove forward   (along x-axis): Numpad 8 / Arrow Up\n"
    msg += "\tMove backward  (along x-axis): Numpad 2 / Arrow Down\n"
    msg += "\tMove left      (along y-axis): Numpad 4 / Arrow Left\n"
    msg += "\tMove right     (along y-axis): Numpad 6 / Arrow Right\n"
    msg += "\tYaw positively (along z-axis): Numpad 7 / Z\n"
    msg += "\tYaw negatively (along z-axis): Numpad 9 / X"
    return msg

  def on_key(self, key: int) -> None:
    """MuJoCo native viewer key callback. Safe to call from the render thread."""
    name = MUJOCO_KEY_TO_INPUT.get(key)
    if name is None:
      return
    with self._lock:
      self._pending.append(name)

  def reset(self) -> None:
    self._base_command.fill(0.0)

  def add_callback(self, key: str, func: Callable) -> None:
    """Bind an extra callback to a named key (``L``, ``UP``, ...)."""
    self._additional_callbacks[key] = func

  def advance(self) -> torch.Tensor:
    """Apply pending key edges and return ``[vx, vy, wz]``."""
    for name, pressed in self._consume_key_events():
      if pressed:
        if name == "L":
          self.reset()
        elif name in self._INPUT_KEY_MAPPING:
          self._base_command += self._INPUT_KEY_MAPPING[name]
        callback = self._additional_callbacks.get(name)
        if callback is not None:
          callback()
      elif name in self._INPUT_KEY_MAPPING:
        self._base_command -= self._INPUT_KEY_MAPPING[name]
    return torch.tensor(
      self._base_command, dtype=torch.float32, device=self._sim_device
    )

  def close(self) -> None:
    with self._lock:
      self._pending.clear()
      self._active.clear()

  def _consume_key_events(self) -> list[tuple[str, bool]]:
    """Return ``(key_name, pressed)`` edges due for this control step.

    A repeated press never yields a second press edge, so the accumulating
    command cannot drift upwards. Expiry of ``hold_timeout`` yields a release.
    """
    now = time.perf_counter()
    events: list[tuple[str, bool]] = []
    with self._lock:
      pending, self._pending = self._pending, []
      for name in pending:
        if name not in self._active:
          events.append((name, True))
        self._active[name] = now + self.hold_timeout
      for name in [n for n, expiry in self._active.items() if expiry <= now]:
        del self._active[name]
        events.append((name, False))
    return events

  def _create_key_bindings(self) -> None:
    """Create default key binding (Isaac Lab Se2Keyboard mapping)."""
    self._INPUT_KEY_MAPPING = {
      "NUMPAD_8": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
      "UP": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
      "NUMPAD_2": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
      "DOWN": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
      # +y is body-left (Isaac Lab convention).
      "NUMPAD_4": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
      "LEFT": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
      "NUMPAD_6": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
      "RIGHT": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
      "NUMPAD_7": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
      "Z": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
      "NUMPAD_9": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
      "X": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
    }


@dataclass
class Se2KeyboardCfg(DeviceCfg):
  """Configuration for :class:`Se2Keyboard`."""

  v_x_sensitivity: float = 0.8
  v_y_sensitivity: float = 0.4
  omega_z_sensitivity: float = 1.0
  hold_timeout: float = 0.5
  """Seconds a MuJoCo key press keeps its velocity contribution."""
  class_type: type[DeviceBase] = Se2Keyboard
