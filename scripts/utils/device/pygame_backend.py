"""pygame-backed input for SE(2) teleop devices.

Isaac Lab talks to Omniverse ``carb.input``. Here the same logical events are
produced from pygame so the devices can run under MuJoCo without Isaac Sim.

Keyboard events use Isaac Lab's SE(2) key names (``UP``, ``NUMPAD_8``, ``L``,
...). Gamepad axes are returned in a robotics-friendly convention: +X right,
+Y up on each stick.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


def _require_pygame():
  try:
    import pygame
  except ImportError as exc:  # pragma: no cover - dependency hint
    raise ImportError(
      "Gamepad teleop requires pygame. Install it with: pip install pygame"
    ) from exc
  return pygame


# pygame keycode -> Isaac Lab Se2Keyboard key name.
_PYGAME_KEY_TO_INPUT: dict[int, str] | None = None


def _pygame_key_map(pygame: Any) -> dict[int, str]:
  global _PYGAME_KEY_TO_INPUT
  if _PYGAME_KEY_TO_INPUT is None:
    _PYGAME_KEY_TO_INPUT = {
      pygame.K_UP: "UP",
      pygame.K_KP8: "NUMPAD_8",
      pygame.K_DOWN: "DOWN",
      pygame.K_KP2: "NUMPAD_2",
      pygame.K_LEFT: "LEFT",
      pygame.K_KP4: "NUMPAD_4",
      pygame.K_RIGHT: "RIGHT",
      pygame.K_KP6: "NUMPAD_6",
      pygame.K_z: "Z",
      pygame.K_KP7: "NUMPAD_7",
      pygame.K_x: "X",
      pygame.K_KP9: "NUMPAD_9",
      pygame.K_l: "L",
    }
  return _PYGAME_KEY_TO_INPUT


@dataclass
class PygameInputBackend:
  """Poll keyboard and joystick events through pygame.

  Parameters:
    create_window: Open a small helper window so keyboard focus works. Gamepad-
      only use can set this to ``False``.
    window_size: Size of the helper window when ``create_window`` is True.
    joystick_index: Which connected joystick to use (0 = first).
    dead_zone: Absolute axis magnitude below which readings are zeroed.
  """

  create_window: bool = True
  window_size: tuple[int, int] = (360, 240)
  joystick_index: int = 0
  dead_zone: float = 0.01
  _pygame: Any = field(default=None, init=False, repr=False)
  _joystick: Any = field(default=None, init=False, repr=False)
  _closed: bool = field(default=False, init=False, repr=False)

  def __post_init__(self) -> None:
    pygame = _require_pygame()
    self._pygame = pygame

    # Headless / windowless joystick reads need a video driver; fall back to
    # dummy when the caller asked for no window and nothing is configured.
    if not self.create_window and not os.environ.get("DISPLAY") and not os.environ.get(
      "WAYLAND_DISPLAY"
    ):
      os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    pygame.joystick.init()

    if self.create_window:
      pygame.display.set_mode(self.window_size)
      pygame.display.set_caption("frog_mjlab teleop")
    else:
      # Joystick subsystem works without a window; ensure the event queue exists.
      try:
        pygame.display.init()
      except Exception:
        pass

    if pygame.joystick.get_count() > self.joystick_index:
      self._joystick = pygame.joystick.Joystick(self.joystick_index)
      self._joystick.init()

  def pump(self) -> None:
    """Advance pygame's event loop."""
    if self._closed:
      return
    self._pygame.event.pump()

  def consume_key_events(self) -> list[tuple[str, bool]]:
    """Return ``(key_name, pressed)`` edges since the last call."""
    if self._closed:
      return []
    pygame = self._pygame
    key_map = _pygame_key_map(pygame)
    events: list[tuple[str, bool]] = []
    for event in pygame.event.get([pygame.KEYDOWN, pygame.KEYUP]):
      name = key_map.get(event.key)
      if name is None:
        continue
      events.append((name, event.type == pygame.KEYDOWN))
    return events

  def get_gamepad_axes(self) -> tuple[float, float, float, float] | None:
    """Return ``(left_x, left_y, right_x, right_y)`` with +Y up, or ``None``."""
    joy = self._joystick
    if joy is None or self._closed:
      return None

    def _axis(index: int, invert: bool = False) -> float:
      if index >= joy.get_numaxes():
        return 0.0
      value = float(joy.get_axis(index))
      if invert:
        value = -value
      return 0.0 if abs(value) < self.dead_zone else value

    left_x = _axis(0)
    left_y = _axis(1, invert=True)  # pygame +Y is down
    naxes = joy.get_numaxes()
    # Linux Xbox-style: 0/1 left, 3/4 right (2/5 triggers). Others: 2/3 right.
    if naxes >= 5:
      right_x = _axis(3)
      right_y = _axis(4, invert=True)
    elif naxes >= 4:
      right_x = _axis(2)
      right_y = _axis(3, invert=True)
    else:
      right_x = 0.0
      right_y = 0.0
    return left_x, left_y, right_x, right_y

  def get_gamepad_buttons(self) -> dict[str, bool]:
    """Return a small named button map (A/B/X/Y/LB/RB/BACK/START when present)."""
    joy = self._joystick
    if joy is None or self._closed:
      return {}
    names = ("A", "B", "X", "Y", "LB", "RB", "BACK", "START")
    out: dict[str, bool] = {}
    for i, name in enumerate(names):
      if i < joy.get_numbuttons():
        out[name] = bool(joy.get_button(i))
    return out

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    pygame = self._pygame
    if self._joystick is not None:
      try:
        self._joystick.quit()
      except Exception:
        pass
      self._joystick = None
    try:
      pygame.joystick.quit()
    except Exception:
      pass
    try:
      pygame.display.quit()
    except Exception:
      pass


@dataclass
class WindowlessGamepadBackend(PygameInputBackend):
  """Gamepad backend that does not open a helper window.

  The joystick subsystem works without a focused display. ``pump`` /
  ``consume_key_events`` may still touch pygame's event queue, so both are
  made non-fatal for headless / minimal setups.
  """

  create_window: bool = False

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
