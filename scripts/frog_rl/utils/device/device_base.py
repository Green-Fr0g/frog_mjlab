"""Minimal teleoperation device base classes (Isaac Lab ``DeviceBase`` subset).

Only the pieces needed by the SE(2) keyboard / gamepad controllers used in
``play.py`` are kept here. Omniverse / retargeter machinery is intentionally
omitted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DeviceCfg:
  """Configuration for teleoperation devices."""

  sim_device: str = "cpu"
  """Torch device string for output tensors."""

  class_type: type[DeviceBase] | None = None
  """Concrete device class constructed for this config."""


class DeviceBase(ABC):
  """Interface for SE(2) teleoperation devices."""

  def __str__(self) -> str:
    return self.__class__.__name__

  @abstractmethod
  def reset(self) -> None:
    """Reset internal command state."""

  @abstractmethod
  def add_callback(self, key: Any, func: Callable) -> None:
    """Bind an extra callback to a device key / button."""

  @abstractmethod
  def advance(self) -> torch.Tensor:
    """Return the current command as a 1-D float tensor ``[vx, vy, wz]``."""

  def close(self) -> None:
    """Release OS resources owned by the device (optional)."""
    return
