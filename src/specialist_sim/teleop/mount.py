"""A gripper mounted on a controller, held by its handle: it moves rigidly with
the controller whenever the headset tracks it, and the trigger sets the opening,
from ``OPEN`` to ``SHUT``, past where the pads meet. A gripper away from its
controller, as on entering a session or when a lost controller is found again,
moves onto it no faster than the embodiment closes a gap.
"""

from __future__ import annotations

import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.teleop.xr import compose

__all__ = ["HANDLE", "MOUNT", "OPEN", "SHUT", "target"]

OPEN = 0.4
SHUT = -0.15
"""The opening at a released and at a pulled-through trigger, radians."""
HANDLE = 0.075
"""How far below the base tag a fist closes on the gripper's handle, along the
tag's z, meters: the middle of the handle, which runs from 0.05 to 0.106."""
MOUNT = np.array([0.0, 0.0, -HANDLE, 0.0, np.sqrt(0.5), -np.sqrt(0.5), 0.0])
"""The base tag in a controller's grip space, the controller held as the
handle. WebXR's grip space has its origin in the curled fingers, -Z along the
held rod toward the thumb, +Y along the arm and +X out of the back of the right
hand, so the tag's x (forward) is -Y, its y (left) -X and its z (up the
handle) -Z."""


def target(controller: Float64Array, trigger: float) -> Float64Array:
  """Where the gripper on a controller is sent, ``(8,)``: the controller's pose
  in the room frame ``(7,)`` and its trigger, from 0 released to 1 pulled."""
  pulled = float(np.clip(trigger, 0.0, 1.0))
  return np.concatenate(
    [compose(controller, MOUNT), [(1.0 - pulled) * OPEN + pulled * SHUT]]
  )
