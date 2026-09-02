"""What varies between episodes.

A reset writes state before the environment checks it, so a bad placement is
drawn again.
"""

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
  "Region",
  "Scatter",
  "Start",
  "scattered",
  "RandomiseCfg",
  "home",
  "nominal",
  "place_cube",
  "place_hands",
  "sample_cube",
  "sample_hands",
]

HOME_XY = ((0.47, 0.16), (0.47, -0.16))
"""Where each hand rests. The operator's left is the room's +y, which is the
side the wall of that name stands on."""
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


@dataclass(frozen=True)
class RandomiseCfg:
  cube_turn: float = np.pi / 2.0
  """How far the cube may be turned about its own axis when it is placed. It is
  square, so a quarter turn covers every pose it can be seen in."""
  hand_jitter: float = 0.02
  hand_turn: float = np.radians(10.0)


def place_cube(
  scene: Scene,
  data: mujoco.MjData,
  spot: npt.ArrayLike,
  yaw: float = 0.0,
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
  position: npt.ArrayLike,
  rotation: Float64Array,
  finger: float,
) -> None:
  """Put one hand at a pose, its two fingers mirrored to an angle."""
  grip = scene.grippers[hand]
  data.qpos[grip.qpos : grip.qpos + 3] = position
  data.qpos[grip.qpos + 3 : grip.qpos + 7] = se3.wxyz(rotation)
  for address, sign in zip(grip.fingers, (1.0, -1.0), strict=True):
    data.qpos[address] = sign * finger


def place_hands(scene: Scene, data: mujoco.MjData, state: Float64Array) -> None:
  """Put every hand exactly where a state vector says."""
  for hand in range(len(GRIPPERS)):
    one = state[hand * PER_HAND : (hand + 1) * PER_HAND]
    place_hand(
      scene,
      data,
      hand,
      one[:3],
      se3.rotation(turn(one[3:7])),
      float(one[7]),
    )


def sample_cube(
  scene: Scene, data: mujoco.MjData, rng: np.random.Generator, cfg: RandomiseCfg
) -> None:
  """On the tape, at any quarter turn.

  An episode opens where the last one left off, so the spread of places the cube
  is picked up from is the spread a kick produces rather than one drawn from a
  patch that no episode ever ends in.
  """
  place_cube(scene, data, scene.tape, rng.uniform(0.0, cfg.cube_turn))


def sample_hands(
  scene: Scene, data: mujoco.MjData, rng: np.random.Generator, cfg: RandomiseCfg
) -> None:
  """Both hands near home, held part open."""
  for hand in range(len(GRIPPERS)):
    rotation = _turn(rng, cfg.hand_turn) @ HOME_ROTATION[hand]
    jitter = rng.uniform(-cfg.hand_jitter, cfg.hand_jitter, size=3)
    place_hand(scene, data, hand, home(scene, hand) + jitter, rotation, FINGER_HOME)


class Start(Protocol):
  """Where an episode opens: anything that can draw the cube a spot."""

  def draw(self, scene: Scene, rng: np.random.Generator) -> Float64Array: ...


@dataclass(frozen=True)
class Scatter:
  """A polar sector of table where a cube may start an episode off the tape.

  The distances are where corpus_0821's picks begin — the kick leaves the cube
  0.086 to 0.105 m from the tape — but a sector holds the corpus's distances,
  not its shape: measured against the closings of that corpus, the true patch
  is a 148-degree-wide, non-radial hull of which this sector covers 66%. An
  evaluation that must match what a policy was trained on derives a ``Region``
  from that corpus instead; these numbers are one corpus's, and accepting them
  is itself a benchmark decision.
  """

  distance: tuple[float, float] = (0.086, 0.105)
  turn: float = np.radians(35.0)
  """How far either side of the line toward the picking hand the cube may lie."""
  hand: int = 1
  """The hand that picks, whose side the cube is scattered on."""

  def draw(self, scene: Scene, rng: np.random.Generator) -> Float64Array:
    return scattered(scene, rng, self)


@dataclass(frozen=True)
class Region:
  """A patch of table, as the counter-clockwise convex ring around it, with
  the hole the ring fills in.

  Drawn from uniformly by rejection, so the distribution is the region and
  nothing else. A corpus's picks are an annulus around the tape — the kick
  always drives the cube away — and a convex hull of an annulus contains the
  tape and the disc around it, which the corpus never picks from; ``hole`` is
  that disc, as ``(x, y, radius)``, and no draw lands inside it. Both the ring
  and the hole are derived from the corpus a policy was trained on by
  ``specialist_sim.corpus.region``, at evaluation time, so no repo evaluates
  against another corpus's geometry by silently accepting shipped numbers.
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

  def draw(self, scene: Scene, rng: np.random.Generator) -> Float64Array:
    ring = np.asarray(self.vertices, np.float64)
    low, high = ring.min(axis=0), ring.max(axis=0)
    edges = np.roll(ring, -1, axis=0) - ring
    while True:
      spot: Float64Array = rng.uniform(low, high)
      toward = spot - ring
      if not (edges[:, 0] * toward[:, 1] - edges[:, 1] * toward[:, 0] >= 0.0).all():
        continue
      if self.hole is not None and np.hypot(*(spot - self.hole[:2])) < self.hole[2]:
        continue
      return spot


def scattered(scene: Scene, rng: np.random.Generator, cfg: Scatter) -> Float64Array:
  """Where a kick would have left the cube, drawn but not placed."""
  tape = np.asarray(scene.tape, np.float64)
  toward = home(scene, cfg.hand)[:2] - tape
  bearing = np.arctan2(toward[1], toward[0]) + rng.uniform(-cfg.turn, cfg.turn)
  spot: Float64Array = tape + rng.uniform(*cfg.distance) * np.array(
    [np.cos(bearing), np.sin(bearing)]
  )
  return spot


def nominal(scene: Scene, data: mujoco.MjData) -> None:
  """A fixed valid state, for when sampling keeps failing."""
  mujoco.mj_resetData(scene.model, data)
  place_cube(scene, data, scene.tape)
  for hand in range(len(GRIPPERS)):
    place_hand(scene, data, hand, home(scene, hand), HOME_ROTATION[hand], FINGER_HOME)


def home(scene: Scene, hand: int) -> Float64Array:
  """A hand's resting position, standing off above the table."""
  return np.array([*HOME_XY[hand], scene.table_top + HOME_HEIGHT])


def _turn(rng: np.random.Generator, limit: float) -> Float64Array:
  """A rotation of up to ``limit`` radians about a uniformly random axis."""
  axis = rng.normal(size=3)
  angle = rng.uniform(0.0, limit)
  quaternion, matrix = np.empty(4), np.empty(9)
  mujoco.mju_axisAngle2Quat(quaternion, axis / np.linalg.norm(axis), angle)
  mujoco.mju_quat2Mat(matrix, quaternion)
  return matrix.reshape(3, 3)
