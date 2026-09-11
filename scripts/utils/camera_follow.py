"""Native-viewer camera utilities for interactive frog_mjlab playback.

MuJoCo's tracking camera keeps its *lookat* pinned to a body but leaves
``azimuth``/``elevation`` in the world frame, so the robot turns in front of a
static camera. :class:`CameraFollower` turns the azimuth with the robot instead,
which is what frog_lab's ``CameraFollower`` achieves with a body-frame camera
offset.
"""

from __future__ import annotations

import math
from collections import deque

from mjlab.viewer import NativeMujocoViewer


class CameraFollower:
  """Keep the viewer camera at a fixed angle relative to the robot's heading.

  The heading is smoothed over a short window so that a jittery gait does not
  make the camera twitch. Averaging is done on the unit circle, which stays
  stable across the +/-180 degree wrap.
  """

  def __init__(self, base_azimuth: float = 90.0, window_size: int = 50) -> None:
    self._base_azimuth = float(base_azimuth)
    self._sin: deque[float] = deque(maxlen=window_size)
    self._cos: deque[float] = deque(maxlen=window_size)

  def reset(self) -> None:
    """Clear the smoothing state after an environment reset."""
    self._sin.clear()
    self._cos.clear()

  def update(self, env) -> float:
    """Return the azimuth (degrees) the viewer camera should use this frame."""
    robot = env.unwrapped.scene["robot"]
    heading = float(robot.data.heading_w[0])  # Radians, world frame.
    angle = math.radians(self._base_azimuth) + heading
    self._sin.append(math.sin(angle))
    self._cos.append(math.cos(angle))
    mean_sin = sum(self._sin) / len(self._sin)
    mean_cos = sum(self._cos) / len(self._cos)
    return math.degrees(math.atan2(mean_sin, mean_cos))


class FollowCameraViewer(NativeMujocoViewer):
  """``NativeMujocoViewer`` that lets a :class:`CameraFollower` steer the camera.

  ``sync_env_to_viewer`` runs once per rendered frame on the main thread and is
  the natural place to refresh the shared ``cam`` struct, which is exactly how
  the base viewer configures the camera in ``setup``.
  """

  def __init__(self, env, policy, *, follower: CameraFollower | None = None, **kwargs):
    super().__init__(env, policy, **kwargs)
    self._follower = follower

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    viewer = self.viewer
    if self._follower is None or viewer is None or not viewer.is_running():
      return
    viewer.cam.azimuth = self._follower.update(self.env)

  def reset_environment(self) -> None:
    super().reset_environment()
    if self._follower is not None:
      self._follower.reset()
