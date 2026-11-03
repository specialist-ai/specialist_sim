"""A controller binding a gripper: a squeeze of the grip binds it where it
stands, with no jump, and the next squeeze releases it. While bound, the
gripper moves rigidly with the controller and the trigger sets the opening,
from ``OPEN`` to ``SHUT``, past where the pads meet.
"""

from __future__ import annotations

import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.teleop.xr import compose, relative

__all__ = ["OPEN", "SHUT", "Grab"]

OPEN = 0.4
SHUT = -0.15
"""The opening at a released and at a pulled-through trigger, radians."""
PRESS, RELEASE = 0.6, 0.4
"""The squeeze at which the grip counts as pressed, and as released again:
apart, so a half-held grip does not count twice."""


class Grab:
  """One controller binding one gripper, a squeeze of its grip at a time."""

  def __init__(self) -> None:
    self.hold: Float64Array | None = None
    """The gripper in the controller's frame while bound, ``(7,)``, or ``None``."""
    self._pressed = False

  def update(
    self,
    controller: Float64Array | None,
    squeeze: float,
    trigger: float,
    target: Float64Array,
  ) -> Float64Array:
    """Where the gripper is sent next, ``(8,)``, from where it was, ``target``,
    and the controller's pose in the room frame ``(7,)`` (``None`` while
    untracked), its squeeze and its trigger. A controller the headset loses
    keeps its binding, and the gripper stands until it is found."""
    pressed = squeeze > RELEASE if self._pressed else squeeze >= PRESS
    toggled = pressed and not self._pressed and controller is not None
    self._pressed = pressed
    if toggled:
      assert controller is not None
      self.hold = relative(controller, target[:7]) if self.hold is None else None
    if self.hold is None or controller is None:
      return target
    pulled = float(np.clip(trigger, 0.0, 1.0))
    return np.concatenate(
      [compose(controller, self.hold), [(1.0 - pulled) * OPEN + pulled * SHUT]]
    )
