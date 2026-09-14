"""The task: which hands a policy drives, where an episode opens, and what
the cube must do for it to be won.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.gripper import SLOTS
from specialist_sim.pickplace.room import Room

__all__ = [
  "BIMANUAL",
  "HOME_ROTATION",
  "SETTLE_STEPS",
  "Fixed",
  "Goal",
  "Home",
  "Judgement",
  "Placed",
  "Region",
  "RestingIn",
  "Start",
  "Task",
  "home",
  "left_task",
  "look",
  "on_tape",
  "right_task",
  "task_for",
]

HOME_ACROSS = 0.16
"""How far to its side of the tape each hand rests, level with it along the
table; the operator's left is the room's +y."""
HOME_HEIGHT = 0.18
"""How far above the table a hand rests."""
_TILT = np.radians(40.0)
_MIRROR = np.diag([1.0, -1.0, 1.0])
_HELD = np.array(
  [
    [0.0, 1.0, 0.0],
    [-np.cos(_TILT), 0.0, -np.sin(_TILT)],
    [-np.sin(_TILT), 0.0, np.cos(_TILT)],
  ]
)
HOME_ROTATION = (_HELD, _MIRROR @ _HELD @ _MIRROR)
"""Each hand's resting orientation: fingers pointing across the table at the
other hand, 40 degrees nose down, mirrored across the table."""
HOME_OPENING = np.radians(20.0)
JITTER = 0.02
TURN = np.radians(10.0)
"""How far from home, and how far turned, a hand starts."""
QUARTER = np.pi / 2.0
"""A square cube shows every face it has within a quarter turn."""


def home(room: Room, side: int) -> Float64Array:
  """Where the hand on one side of the rig rests; ``side`` indexes ``SLOTS``."""
  across = HOME_ACROSS if side == 0 else -HOME_ACROSS
  return np.array([room.tape[0], room.tape[1] + across, room.table_top + HOME_HEIGHT])


@dataclass(frozen=True)
class Region:
  """A patch of table: a counter-clockwise convex ring, less a disc."""

  vertices: tuple[tuple[float, float], ...]
  hole: tuple[float, float, float] | None = None
  """A disc no draw lands in, as ``(x, y, radius)``."""

  def contains(self, points: Float64Array) -> BoolArray:
    """Whether each point, ``(..., 2)``, lies in the ring and out of the hole."""
    ring = np.asarray(self.vertices)
    edges = np.roll(ring, -1, axis=0) - ring
    toward = np.asarray(points)[..., None, :] - ring
    inside: BoolArray = (
      edges[:, 0] * toward[..., 1] - edges[:, 1] * toward[..., 0] >= 0.0
    ).all(axis=-1)
    if self.hole is not None:
      inside &= np.hypot(*(points - np.asarray(self.hole[:2])).T).T >= self.hole[2]
    return inside

  def draw(self, rng: np.random.Generator) -> Float64Array:
    """A point drawn uniformly, by rejection."""
    ring = np.asarray(self.vertices)
    while True:
      spot: Float64Array = rng.uniform(ring.min(axis=0), ring.max(axis=0))
      if self.contains(spot):
        return spot


class Start(Protocol):
  def draw(
    self, room: Room, rng: np.random.Generator
  ) -> tuple[Float64Array, Float64Array]:
    """Every hand's state, ``(hands, 8)``, and the cube's ``(x, y, yaw)``."""
    ...


@dataclass(frozen=True)
class Home:
  """Every hand near its home, turned a little and part open; the cube upright
  on the tape, or in a region, at a random yaw."""

  region: Region | None = None

  def draw(
    self, room: Room, rng: np.random.Generator
  ) -> tuple[Float64Array, Float64Array]:
    hands = np.empty((len(room.slots), 8))
    for index, slot in enumerate(room.slots):
      side = SLOTS.index(slot)
      axis = rng.normal(size=3)
      turned = rotation.exp(axis / np.linalg.norm(axis) * rng.uniform(0.0, TURN))
      hands[index, :3] = home(room, side) + rng.uniform(-JITTER, JITTER, size=3)
      hands[index, 3:7] = rotation.multiply(
        turned, rotation.from_matrix(HOME_ROTATION[side])
      )
      hands[index, 7] = HOME_OPENING
    spot = room.tape if self.region is None else self.region.draw(rng)
    return hands, np.array([*spot, rng.uniform(0.0, QUARTER)])


@dataclass(frozen=True)
class Fixed:
  """A start named outright."""

  hands: Float64Array
  cube: Float64Array

  def draw(
    self, room: Room, rng: np.random.Generator
  ) -> tuple[Float64Array, Float64Array]:
    return self.hands, self.cube


PLACE_TOLERANCE = 0.010
"""How far the cube's center may stand beyond the tape's half-width."""
REST_HEIGHT = 0.005
REST_SPEED = 0.02
REST_SPIN = 0.5
UPRIGHT = 0.966
"""What a resting cube is: seated on the table, still, and with a face up."""
DROP_DEPTH = 0.10
"""How far below the table top a cube has fallen off it."""
SETTLE_STEPS = 5
"""How many steps a goal must hold before the episode is won."""


def on_tape(room: Room, position: Float64Array) -> BoolArray:
  """Whether a cube at each position, ``(..., 3)``, stands on the tape."""
  out: BoolArray = (
    np.abs(position[..., :2] - room.tape) <= room.tape_half + PLACE_TOLERANCE
  ).all(axis=-1)
  return out


@dataclass(frozen=True)
class Judgement:
  """What every world's cube is doing, one entry per world."""

  cube: Float64Array
  """The cube's position, ``(worlds, 3)``."""
  on_tape: BoolArray
  resting: BoolArray
  held: BoolArray
  """Whether a gripping pad touches it."""
  dropped: BoolArray
  displaced: BoolArray
  """Whether it has been off the tape since the episode opened."""


def look(room: Room, data: mujoco.MjData) -> tuple[Float64Array, bool, bool, bool]:
  """One world's cube: its position, and whether it is resting, held and
  dropped."""
  cube = room.cube
  position = data.xpos[cube.body]
  velocity = data.cvel[cube.body]
  upright = float(np.abs(data.xmat[cube.body].reshape(3, 3)[2]).max())
  resting = bool(
    abs(position[2] - cube.rest) < REST_HEIGHT
    and np.linalg.norm(velocity[3:]) < REST_SPEED
    and np.linalg.norm(velocity[:3]) < REST_SPIN
    and upright > UPRIGHT
  )
  pairs = data.contact.geom[: data.ncon]
  touching = pairs[(pairs == cube.geom).any(axis=1)]
  held = bool(np.isin(touching, room.pads).any())
  dropped = bool(position[2] < room.table_top - DROP_DEPTH)
  return position.copy(), resting, held, dropped


class Goal(Protocol):
  def reached(self, judgement: Judgement) -> BoolArray:
    """Which worlds' cubes are doing what the task asks."""
    ...


@dataclass(frozen=True)
class Placed:
  """The cube back on the tape, at rest and let go of, having been off it."""

  def reached(self, judgement: Judgement) -> BoolArray:
    out: BoolArray = (
      judgement.displaced & judgement.on_tape & judgement.resting & ~judgement.held
    )
    return out


@dataclass(frozen=True)
class RestingIn:
  """The cube at rest off the tape, let go of, inside a region."""

  region: Region

  def reached(self, judgement: Judgement) -> BoolArray:
    out: BoolArray = (
      judgement.resting
      & ~judgement.on_tape
      & ~judgement.held
      & self.region.contains(judgement.cube[:, :2])
    )
    return out


@dataclass(frozen=True)
class Task:
  driven: tuple[str, ...]
  """The hands the action commands and the observation carries states for."""
  start: Start = Home()
  goal: Goal = Placed()
  seen: tuple[str, ...] = ()
  """Lenses the observation carries beyond the driven hands' own."""


BIMANUAL = Task(SLOTS)
"""Both hands, from the tape, until the cube is put back on it."""


def right_task(region: Region) -> Task:
  """The right hand alone, from a cube kicked into ``region``."""
  return Task((SLOTS[1],), Home(region))


def left_task(region: Region) -> Task:
  """The left hand alone, until the cube rests in ``region``."""
  return Task((SLOTS[0],), goal=RestingIn(region))


def task_for(driven: tuple[str, ...], region: Region) -> Task:
  """The task a policy driving ``driven`` is evaluated on."""
  return {
    SLOTS: BIMANUAL,
    (SLOTS[1],): right_task(region),
    (SLOTS[0],): left_task(region),
  }[driven]
