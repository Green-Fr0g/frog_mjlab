"""Play-only teleoperation helpers for frog_mjlab.

Drives the ``twist`` velocity command from a keyboard or a gamepad during
interactive playback. These classes deliberately live next to ``play.py``
instead of under ``frog_mjlab/tasks/`` because they are not part of the training
MDP and must stay out of the installed package.

They reuse mjlab's own ``Se2Keyboard`` / ``Se2Gamepad`` devices, so key and stick
bindings are identical to Isaac Lab's ``isaaclab.devices``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

import torch

from mjlab.devices import (
  Se2Gamepad,
  Se2GamepadCfg,
  Se2Keyboard,
  Se2KeyboardCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
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

try:  # Private module, but the only way to disable pygame's helper window.
  from mjlab.devices._pygame_backend import PygameInputBackend as _PygameInputBackend
except ImportError:  # pragma: no cover - guards a future mjlab layout change.
  _PygameInputBackend = object  # type: ignore[assignment,misc]


# MuJoCo viewer key code -> mjlab SE(2) device key name.
# Mirrors mjlab's pygame backend, which itself mirrors Isaac Lab's Se2Keyboard.
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


class MujocoKeyBackend:
  """Input backend fed by the MuJoCo native viewer's key callback.

  MuJoCo only ever reports key *presses*: ``PlatformUIAdapter::OnKey`` in
  ``simulate/platform_ui_adapter.cc`` returns immediately unless the GLFW action
  is ``GLFW_PRESS``, so key releases and auto-repeats never reach the callback.
  A press is therefore turned into a synthetic hold that lasts ``hold_timeout``
  seconds and is refreshed by each further press.

  Practical consequences:

  * Tapping a direction key moves for ``hold_timeout`` seconds.
  * Holding a key does not keep moving; re-tap to extend. A large
    ``hold_timeout`` behaves like a latch: press once to start, press ``L``
    (or the opposite key) to stop.
  """

  def __init__(self, hold_timeout: float = 0.5) -> None:
    self.hold_timeout = max(float(hold_timeout), 0.0)
    self._lock = Lock()
    self._pending: list[str] = []
    self._active: dict[str, float] = {}

  def push(self, name: str) -> None:
    """Queue a key press. Called on the MuJoCo viewer (render) thread."""
    with self._lock:
      self._pending.append(name)

  def __deepcopy__(self, memo: dict) -> MujocoKeyBackend:
    """Shared, never copied: the viewer callback and the command term must agree.

    Also stops a stray ``deepcopy`` of the env config from choking on the lock.
    """
    return self

  def pump(self) -> None:
    """Nothing to pump: MuJoCo owns the window and its event loop."""
    return

  def consume_key_events(self) -> list[tuple[str, bool]]:
    """Return the ``(key_name, pressed)`` edges due for this control step.

    A repeated press never yields a second press edge, so the device's
    accumulating key map cannot drift upwards.
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

  def get_gamepad_axes(self) -> None:
    return None

  def get_gamepad_buttons(self) -> dict[str, bool]:
    return {}

  def close(self) -> None:
    with self._lock:
      self._pending.clear()
      self._active.clear()


class WindowlessGamepadBackend(_PygameInputBackend):  # type: ignore[misc,valid-type]
  """pygame gamepad backend that opens no second window.

  The joystick subsystem works without a window; only keyboard events need a
  focused display surface. ``pump`` / ``consume_key_events`` touch pygame's
  event queue, which may require an initialised video system, so both are made
  non-fatal.
  """

  def __post_init__(self) -> None:
    self.create_window = False
    super().__post_init__()

  def pump(self) -> None:
    try:
      super().pump()
    except Exception:
      pass

  def consume_key_events(self) -> list[tuple[str, bool]]:
    try:
      return super().consume_key_events()
    except Exception:
      return []


def _gamepad_backend() -> Any:
  """Return a windowless gamepad backend, or ``None`` for mjlab's default."""
  if _PygameInputBackend is object:
    return None
  return WindowlessGamepadBackend()


class TeleopVelocityCommand(UniformVelocityCommand):
  """Velocity command term whose command tensor is written by a teleop device.

  Only ``_update_command`` is overridden -- the last step of
  ``CommandTerm.compute`` (``_update_metrics`` -> resample -> ``_update_command``).
  That tensor is the single source of truth read by the ``command`` observation,
  the ``phase`` observation and every command-driven reward term, so driving it
  here keeps all of them consistent. Swapping an observation term instead would
  desync the gait phase from the real command.

  Heading/standing sampling is intentionally bypassed: the operator is the
  command.
  """

  cfg: TeleopVelocityCommandCfg

  def __init__(self, cfg: TeleopVelocityCommandCfg, env) -> None:
    super().__init__(cfg, env)
    self._device = self._make_device(cfg)

  @staticmethod
  def _make_device(cfg: TeleopVelocityCommandCfg):
    # Use the upper command bound as sensitivity, exactly like frog_lab's play.py.
    sensitivities = {
      "v_x_sensitivity": cfg.ranges.lin_vel_x[1],
      "v_y_sensitivity": cfg.ranges.lin_vel_y[1],
      "omega_z_sensitivity": cfg.ranges.ang_vel_z[1],
      "sim_device": cfg.sim_device,
    }
    if cfg.device_type == "gamepad":
      return Se2Gamepad(Se2GamepadCfg(**sensitivities), backend=_gamepad_backend())
    return Se2Keyboard(Se2KeyboardCfg(**sensitivities), backend=cfg.input_backend)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids  # The command is a pure function of the device state.
    self.vel_command_b[:] = self._device.advance().to(self.device).unsqueeze(0)

  def close(self) -> None:
    self._device.close()


@dataclass(kw_only=True)
class TeleopVelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for :class:`TeleopVelocityCommand` (play only)."""

  device_type: Literal["keyboard", "gamepad"] = "keyboard"
  """Which SE(2) device drives the command."""

  sim_device: str = "cpu"
  """Torch device string for the device's output tensors."""

  input_backend: Any = field(default=None, repr=False, compare=False)
  """Runtime-only input backend injected by ``play.py``; not a user-facing knob.

  Only used by the keyboard device. ``None`` falls back to mjlab's default
  (pygame, which opens a small helper window).
  """

  def build(self, env) -> TeleopVelocityCommand:
    return TeleopVelocityCommand(self, env)
