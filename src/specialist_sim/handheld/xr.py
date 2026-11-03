"""From a Quest headset's space to the room frame, and how the demonstrator moves
the scene around in it.

The headset reports in WebXR's ``local-floor`` space: y up, the floor at
y = 0, -z the way the demonstrator faced at the start, quaternions ``[w, x, y,
z]``. The room frame has z up and the demonstrator at +x facing -x. ``Anchor`` maps
one to the other. ``facing`` stands a scene's focus ``REACH`` ahead of the
head, the room's floor on the real one. ``steer`` and ``nudged`` move it from
there, turning it about the focus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array

__all__ = [
  "REACH",
  "Anchor",
  "ahead",
  "compose",
  "facing",
  "heading",
  "moved",
  "nudged",
  "steer",
]

REACH = 0.5
"""How far ahead of the demonstrator's head the focus stands, meters."""
TURN = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
"""The room's axes in the headset's, facing the way the session began: the
room's +y is the demonstrator's right (+x), its +z is up (+y), its +x is toward the
demonstrator (+z)."""
SLIDE = 0.25
RAISE = 0.15
SPIN = np.radians(30.0)
"""How fast a thumbstick pushed all the way moves the scene: m/s along the
floor, m/s up or down, rad/s about the vertical. The response is quadratic past
``DEADZONE``, so a small push is a fine adjustment."""
DEADZONE = 0.12


def compose(a: Float64Array, b: Float64Array) -> Float64Array:
  """Pose ``b`` ``(7,)`` given in the frame of pose ``a``, in ``a``'s parent frame."""
  return np.concatenate(
    [a[:3] + rotation.rotate(a[3:7], b[:3]), rotation.multiply(a[3:7], b[3:7])]
  )


def ahead(
  newer: Float64Array, older: Float64Array, apart: float, by: float
) -> Float64Array:
  """Pose ``newer`` ``(7,)`` carried ``by`` seconds on along its motion from
  ``older``, a pose ``apart`` seconds before it, at the same speed and turn."""
  rate = by / apart
  turn = rotation.log(rotation.multiply(newer[3:7], rotation.conjugate(older[3:7])))
  return np.concatenate(
    [
      newer[:3] + rate * (newer[:3] - older[:3]),
      rotation.multiply(rotation.exp(rate * turn), newer[3:7]),
    ]
  )


@dataclass(frozen=True)
class Anchor:
  """The room frame in the headset's space: ``xr = rotation @ room + offset``."""

  rotation: Float64Array
  offset: Float64Array

  def to_room(self, pose: Float64Array) -> Float64Array:
    """A headset-space pose ``(7,)`` in the room frame."""
    turn = rotation.from_matrix(self.rotation.T)
    return np.concatenate(
      [self.rotation.T @ (pose[:3] - self.offset), rotation.multiply(turn, pose[3:7])]
    )

  def matrix(self) -> Float64Array:
    """The ``(4, 4)`` homogeneous transform from the room frame to the
    headset's space."""
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = self.rotation, self.offset
    return out


def facing(head: Float64Array, focus: Float64Array) -> Anchor:
  """The room stood in front of a head at ``head`` ``(7,)``, in headset space:
  ``focus`` ``(3,)``, a room point, ``REACH`` ahead of it along the way it
  faces, and the room's floor, z = 0, on the real one."""
  ahead = rotation.rotate(head[3:7], np.array([0.0, 0.0, -1.0]))
  turn = np.arctan2(-ahead[0], -ahead[2])
  c, s = np.cos(turn), np.sin(turn)
  turned = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ TURN
  demonstrator = np.array([focus[0] + REACH, focus[1], 0.0])
  # The demonstrator's spot lies at z = 0, so ``turned`` leaves the height to the offset alone.
  offset = np.array([head[0], 0.0, head[2]]) - turned @ demonstrator
  return Anchor(turned, offset)


def heading(head: Float64Array) -> tuple[Float64Array, Float64Array]:
  """The way a head ``(7,)`` faces along the floor and its right, unit vectors
  in headset space."""
  ahead = rotation.rotate(head[3:7], np.array([0.0, 0.0, -1.0]))
  ahead = np.array([ahead[0], 0.0, ahead[2]]) / max(np.hypot(ahead[0], ahead[2]), 1e-9)
  return ahead, np.array([-ahead[2], 0.0, ahead[0]])


def moved(
  anchor: Anchor, step: Float64Array, turn: float, pivot: Float64Array
) -> Anchor:
  """``anchor`` with the room shifted ``step`` ``(3,)`` in headset space and
  turned ``turn`` radians about the vertical (headset +y, counterclockwise seen
  from above) through ``pivot``, a room point."""
  if turn == 0.0 and not np.any(step):
    return anchor  # untouched, not multiplied by the identity 150 times a second until it drifts
  c, s = np.cos(turn), np.sin(turn)
  spin = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
  center = anchor.rotation @ pivot + anchor.offset
  return Anchor(spin @ anchor.rotation, spin @ (anchor.offset - center) + center + step)


def nudged(
  anchor: Anchor,
  head: Float64Array | None,
  pivot: Float64Array,
  forward: float,
  right: float,
  up: float,
  turn: float,
) -> Anchor:
  """``anchor`` with the room moved ``forward`` and ``right`` along the floor
  the way ``head`` ``(7,)`` faces (the way the room's demonstrator faces, its -x,
  without a head), ``up``, meters, and turned ``turn`` radians counterclockwise
  seen from above about the vertical through ``pivot``."""
  if head is None:
    ahead = anchor.rotation @ np.array([-1.0, 0.0, 0.0])
    ahead = np.array([ahead[0], 0.0, ahead[2]]) / max(
      np.hypot(ahead[0], ahead[2]), 1e-9
    )
    side = np.array([-ahead[2], 0.0, ahead[0]])
  else:
    ahead, side = heading(head)
  return moved(
    anchor, forward * ahead + right * side + np.array([0.0, up, 0.0]), turn, pivot
  )


def _response(axis: float) -> float:
  if abs(axis) < DEADZONE:
    return 0.0
  return float(np.sign(axis) * ((abs(axis) - DEADZONE) / (1.0 - DEADZONE)) ** 2)


def steer(
  anchor: Anchor,
  head: Float64Array,
  pivot: Float64Array,
  left: tuple[float, float],
  right: tuple[float, float],
  dt: float,
) -> Anchor:
  """``anchor`` after ``dt`` seconds of the thumbsticks, each ``(x, y)`` as a
  gamepad reads them (y is negative pushed forward): the left one slides the
  scene forward, back and to the sides of the way ``head`` faces, the right one
  raises or lowers it (y) and turns it about ``pivot`` (x, pushed right turns
  it clockwise seen from above)."""
  return nudged(
    anchor,
    head,
    pivot,
    forward=SLIDE * dt * _response(-left[1]),
    right=SLIDE * dt * _response(left[0]),
    up=RAISE * dt * _response(-right[1]),
    turn=-SPIN * dt * _response(right[0]),
  )
