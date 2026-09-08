"""Where an episode opens: the cube, and every hand, drawn from one stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np
import numpy.typing as npt

from specialist_sim import pose as se3
from specialist_sim.action import turn
from specialist_sim.arrays import Float64Array
from specialist_sim.scene import Scene
from specialist_sim.spaces import GRIPPERS, PER_HAND

__all__ = [
  "Homes",
  "Kicked",
  "OnTape",
  "Region",
  "Start",
  "home",
  "place_cube",
  "place_hand",
  "place_hands",
]

HOME_XY = ((0.47, 0.16), (0.47, -0.16))
"""Where each hand rests. The operator's left is the room's +y, which is the
side the left anchor stands on."""
HOME_HEIGHT = 0.18
"""How far above the table a hand rests."""
# The fingers point across the table at the other hand, 40 degrees nose down.
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
"""Each hand's resting orientation, mirrored across the table from each other."""
FINGER_HOME = np.radians(20.0)
CLEARANCE = 1e-4
"""How far above the table a placed cube starts, so it never begins inside it."""
QUARTER = np.pi / 2.0
"""A square cube shows every face it has within a quarter turn."""


def place_cube(
  scene: Scene, data: mujoco.MjData, spot: npt.ArrayLike, yaw: float = 0.0
) -> None:
  """Rest the cube upright at a spot on the table, turned about its own axis."""
  quaternion = np.empty(4)
  mujoco.mju_axisAngle2Quat(quaternion, np.array([0.0, 0.0, 1.0]), yaw)
  address = scene.cube.qpos
  place = np.asarray(spot, np.float64)
  data.qpos[address : address + 3] = (place[0], place[1], scene.cube.rest + CLEARANCE)
  data.qpos[address + 3 : address + 7] = quaternion


def place_hand(
  scene: Scene,
  data: mujoco.MjData,
  hand: int,
  position: Float64Array,
  quaternion: Float64Array,
  opening: float,
) -> None:
  """Put one hand at a pose, its two fingers mirrored to an opening."""
  one = scene.hands[hand]
  data.qpos[list(one.fingers)] = opening, -opening
  scene.embodiment.place(data, one, position, quaternion)


def place_hands(scene: Scene, data: mujoco.MjData, state: Float64Array) -> None:
  """Put every hand exactly where a state vector says."""
  hands = len(scene.hands)
  if state.shape != (hands * PER_HAND,):
    raise ValueError(
      f"a state names {hands} hands as {hands * PER_HAND} numbers, not {state.shape}"
    )
  for hand in range(hands):
    one = state[hand * PER_HAND : (hand + 1) * PER_HAND]
    place_hand(scene, data, hand, one[:3], turn(one[3:7]), float(one[7]))


def home(scene: Scene, side: int) -> Float64Array:
  """Where the hand on one side of the rig rests, standing off above the
  table; ``side`` indexes ``GRIPPERS``."""
  return np.array([*HOME_XY[side], scene.table_top + HOME_HEIGHT])


class HandStart(Protocol):
  def place(
    self, scene: Scene, data: mujoco.MjData, hand: int, rng: np.random.Generator
  ) -> None: ...


@dataclass(frozen=True)
class Homes:
  """A hand near its home, turned a little, held part open."""

  jitter: float = 0.02
  turn: float = np.radians(10.0)
  opening: float = FINGER_HOME

  def place(
    self, scene: Scene, data: mujoco.MjData, hand: int, rng: np.random.Generator
  ) -> None:
    side = GRIPPERS.index(scene.slots[hand])
    rotation = _turned(rng, self.turn) @ HOME_ROTATION[side]
    jitter = rng.uniform(-self.jitter, self.jitter, size=3)
    place_hand(
      scene, data, hand, home(scene, side) + jitter, se3.wxyz(rotation), self.opening
    )


class Start(Protocol):
  """Where an episode opens: the cube and every hand, from the episode's stream."""

  def place(
    self, scene: Scene, data: mujoco.MjData, rng: np.random.Generator
  ) -> None: ...


@dataclass(frozen=True)
class OnTape:
  """The cube on the tape, as every recorded episode opens."""

  turn: float = QUARTER
  hands: tuple[HandStart, ...] = (Homes(), Homes())

  def place(self, scene: Scene, data: mujoco.MjData, rng: np.random.Generator) -> None:
    for hand in range(len(scene.hands)):
      self.hands[hand].place(scene, data, hand, rng)
    place_cube(scene, data, scene.tape, rng.uniform(0.0, self.turn))


@dataclass(frozen=True)
class Region:
  """A patch of table, as the counter-clockwise convex ring around it, with
  the hole the ring fills in.

  Drawn from uniformly by rejection, so the distribution is the region and
  nothing else. A dataset's picks are an annulus around the tape — the kick
  always drives the cube away — and a convex hull of an annulus contains the
  tape and the disc around it, which the dataset never picks from; ``hole`` is
  that disc, as ``(x, y, radius)``, and no draw lands inside it. Both are
  derived from the dataset a policy was trained on by
  ``specialist_sim.dataset.region``, at evaluation time, never shipped.
  """

  vertices: tuple[tuple[float, float], ...]
  hole: tuple[float, float, float] | None = None

  def __post_init__(self) -> None:
    ring = np.asarray(self.vertices, np.float64)
    if len(ring) < 3:
      raise ValueError(f"a region is a ring of at least 3 vertices, not {len(ring)}")
    ahead = np.roll(ring, -1, axis=0) - ring
    bend = np.roll(ahead, -1, axis=0)
    if not (ahead[:, 0] * bend[:, 1] - ahead[:, 1] * bend[:, 0] > 0.0).all():
      raise ValueError("a region's ring is convex and counter-clockwise")
    if self.hole is not None and self.hole[2] < 0.0:
      raise ValueError(f"a hole's radius is not negative: {self.hole[2]}")

  def contains(self, spot: npt.ArrayLike) -> bool:
    """Whether a point lies in the ring and out of the hole."""
    point = np.asarray(spot, np.float64)
    ring = np.asarray(self.vertices, np.float64)
    edges = np.roll(ring, -1, axis=0) - ring
    toward = point - ring
    if not (edges[:, 0] * toward[:, 1] - edges[:, 1] * toward[:, 0] >= 0.0).all():
      return False
    return self.hole is None or bool(np.hypot(*(point - self.hole[:2])) >= self.hole[2])

  def draw(self, rng: np.random.Generator) -> Float64Array:
    ring = np.asarray(self.vertices, np.float64)
    low, high = ring.min(axis=0), ring.max(axis=0)
    for _ in range(10_000):
      spot: Float64Array = rng.uniform(low, high)
      if self.contains(spot):
        return spot
    raise ValueError("the region leaves nothing to draw from")


@dataclass(frozen=True)
class Kicked:
  """The cube where a kick would have left it: at rest in a region, turned."""

  region: Region
  turn: float = QUARTER
  hands: tuple[HandStart, ...] = (Homes(), Homes())

  def place(self, scene: Scene, data: mujoco.MjData, rng: np.random.Generator) -> None:
    for hand in range(len(scene.hands)):
      self.hands[hand].place(scene, data, hand, rng)
    place_cube(scene, data, self.region.draw(rng), rng.uniform(0.0, self.turn))


def _turned(rng: np.random.Generator, limit: float) -> Float64Array:
  """A rotation of up to ``limit`` radians about a uniformly random axis."""
  axis = rng.normal(size=3)
  angle = rng.uniform(0.0, limit)
  quaternion, matrix = np.empty(4), np.empty(9)
  mujoco.mju_axisAngle2Quat(quaternion, axis / np.linalg.norm(axis), angle)
  mujoco.mju_quat2Mat(matrix, quaternion)
  return matrix.reshape(3, 3)
