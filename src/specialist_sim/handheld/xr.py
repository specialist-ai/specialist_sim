"""From a Quest headset's space to the room frame, and the gripper each
controller carries.

The headset reports in WebXR's ``local-floor`` space: y up, the floor at y = 0, -z
the way the demonstrator faced at the start. The room frame has z up and the
demonstrator at -x facing +x, their left at +y. An anchor, the room frame's pose
``(7,)`` in headset space, maps one to the other. Quaternions are wxyz.
"""

from dataclasses import dataclass

import numpy as np

from specialist_sim import gripper, rotation
from specialist_sim.arrays import Float64Array

# The finger servo's stiffness and damping for a demonstrator's grip: critically
# damped at 25 Hz on the right finger's armature.
FINGER = (567.5, 7.23)
# How far ahead of the demonstrator's head the focus stands.
REACH = 0.5
# How far below the base tag a fist closes on the gripper's handle, along the
# tag's z.
HANDLE = 0.075
# The base tag in a controller's grip space, the controller held as the handle.
# WebXR's grip space has its origin in the curled fingers, -Z along the rod toward
# the thumb, +Y along the arm and +X out of the back of the right hand, so the
# tag's x (forward) is -Y, its y (left) -X and its z (up the handle) -Z.
MOUNT = np.array([0.0, 0.0, -HANDLE, 0.0, np.sqrt(0.5), -np.sqrt(0.5), 0.0])


@dataclass(frozen=True)
class Controller:
  pose: Float64Array
  """``(7,)`` in headset space."""
  trigger: float


@dataclass(frozen=True)
class Inputs:
  head: Float64Array
  """``(7,)`` in headset space."""
  hands: dict[str, Controller]
  """The controllers tracked, by hand."""
  at: float
  """When the headset's frame arrived, ``time.perf_counter()``."""
  drawn: float
  """When the headset drew the frame, seconds on the headset's own clock."""


def _heading(head: Float64Array) -> float:
  """The way a head ``(7,)`` faces along the floor: radians about headset +y,
  counterclockwise seen from above, from -z."""
  forward = rotation.rotate(head[3:7], np.array([0.0, 0.0, -1.0]))
  return float(np.arctan2(-forward[0], -forward[2]))


def facing(head: Float64Array, focus: Float64Array) -> Float64Array:
  """The anchor that stands the room before a head ``(7,)``: ``focus`` ``(3,)``,
  a room point, ``REACH`` ahead of it along the way it faces, and the room's
  floor, z = 0, on the real one."""
  # The room's axes in headset space before the turn: +x ahead (-z), +y left (-x),
  # +z up (+y).
  axes = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
  turned = rotation.multiply(
    rotation.exp(np.array([0.0, _heading(head), 0.0])), rotation.from_matrix(axes)
  )
  # The demonstrator's spot lies at z = 0, so the turn leaves its height alone.
  spot = np.array([focus[0] - REACH, focus[1], 0.0])
  standing = rotation.rotate(turned, spot)
  return np.concatenate([np.array([head[0], 0.0, head[2]]) - standing, turned])


def to_room(anchor: Float64Array, pose: Float64Array) -> Float64Array:
  """A headset-space pose ``(7,)`` in the room frame."""
  return rotation.compose(rotation.inverse(anchor), pose)


def target(controller: Float64Array, trigger: float) -> Float64Array:
  """Where the gripper on a controller is sent, ``(8,)``: the controller's pose
  in the room frame ``(7,)`` and its trigger, from 0 released to 1 pulled."""
  opening = (1.0 - trigger) * gripper.OPEN + trigger * gripper.SHUT
  return np.append(rotation.compose(controller, MOUNT), opening)


def carried(
  newest: Inputs, older: Inputs | None, hand: str, now: float
) -> Float64Array:
  """A controller's pose in headset space at ``now``, ``time.perf_counter()``:
  the newest frame's, carried on along its motion since the frame before, by the
  time since it arrived (at most a frame's)."""
  pose = newest.hands[hand].pose
  if older is None or hand not in older.hands:
    return pose
  apart = newest.drawn - older.drawn
  if apart <= 0.0:
    return pose
  by = min(now - newest.at, apart)
  return rotation.interpolate(older.hands[hand].pose, pose, 1.0 + by / apart)
