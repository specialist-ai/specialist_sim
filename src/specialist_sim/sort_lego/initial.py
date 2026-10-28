"""Where an episode starts: the bricks' layout, drawn from a seed, and the hands.

A layout puts some bricks on the tray and parks the rest in their bins:

* **drop**: the collection task. One to eight bricks, either color, anywhere in
  the well at any turn and a little tilted, each dropped from a random 1 to
  4 cm. A brick that overlaps one already placed starts above it, so no two
  start inside each other.
* **single**: one brick lying flat anywhere in the well.
* **pile**: the bricks heaped near the middle, one above the other, the most
  contacts a tray makes.

A layout is a function of ``(mode, seed, bricks)``, which each episode records,
so a later run lays the same tray again. ``bricks`` sets how many lie on the
tray, of any colors.

A driven hand starts at ``home``, over the tray's near corner on its side; a
hand nobody drives waits ``aside``, out of every lens.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.gripper import SLOTS
from specialist_sim.sort_lego import scene

__all__ = [
  "MODES",
  "OPEN",
  "SIDES",
  "Layout",
  "aside",
  "draw",
  "home",
  "parked",
  "starts",
]

REST_Z = scene.TRAY_FLOOR + scene.BRICK[2] / 2 + 5e-4
WALLS = (0.103, 0.143)
"""How far from the tray's middle a brick's middle can lie and the brick still
lie in the well, meters."""
MODES = ("drop", "single", "pile")
SETTLE = {"drop": 1.0, "single": 0.1, "pile": 0.6}
"""Seconds a layout rests before the hands move: a drop and its tumble take
under a second."""
DROP = (1, 8)
"""How many bricks a ``drop`` or ``pile`` tray holds, fewest and most."""
DROP_HEIGHT = (0.01, 0.04)
"""How far above resting a dropped brick starts before any stacking, meters."""
DROP_TILT = 0.25
"""The most a dropped brick is tilted from flat, radians."""
DROP_CLEAR = 0.04
"""How near two bricks' middles may lie in plan before one starts above the
other: twice the half diagonal of a brick tilted ``DROP_TILT``, with a margin,
meters."""
DROP_STACK = 0.025
"""How far above an overlapped brick a brick starts: taller than a tilted brick
stands (20 mm)."""

SIDES = {"left_gripper": -1.0, "right_gripper": 1.0}
"""Which side of the tray each hand works from, along y: the operator stands at
+x facing -x."""
OPEN = 0.4
"""The opening a hand starts at: a released Quest trigger, radians."""
HOME_HEIGHT = 0.22
"""How far over the tray floor a hand starts: the right hand's median start over
the S and H Quest episodes, meters."""


@dataclass(frozen=True)
class Layout:
  mode: str
  seed: int
  bricks: int | None
  """How many bricks were asked for; ``None`` lets the mode draw its count."""
  poses: Float64Array
  """Every brick, ``(k, 7)`` position and wxyz."""
  active: BoolArray
  """Which bricks start on the tray, ``(k,)``."""

  @property
  def settle(self) -> float:
    return SETTLE[self.mode]


def home(slot: str) -> Float64Array:
  """Where a driven hand starts, ``(8,)``: ``HOME_HEIGHT`` over the tray's near
  corner on its side, fingers aimed at the middle of the tray floor, open."""
  side = SIDES[slot]
  corner = np.array(
    [
      scene.TRAY_CENTER[0] + scene.TRAY_SIZE[0] / 2,
      scene.TRAY_CENTER[1] + scene.TRAY_SIZE[1] / 2 * side,
      scene.TRAY_FLOOR + HOME_HEIGHT,
    ]
  )
  middle = np.array([*scene.TRAY_CENTER, scene.TRAY_FLOOR])
  return np.array([*corner, *rotation.pointing(middle - corner), OPEN])


def aside(slot: str) -> Float64Array:
  """Where a hand nobody drives waits, ``(8,)``: a meter out on its side and
  below the desk top, out of every lens, as the operator puts a free gripper
  away."""
  position = [
    scene.DESK_CENTER[0],
    scene.DESK_CENTER[1] + SIDES[slot],
    scene.DESK_TOP - 0.3,
  ]
  return np.array([*position, *rotation.pointing(np.array([0.0, 0.0, -1.0])), OPEN])


def starts(driven: tuple[str, ...]) -> Float64Array:
  """Every hand's start, ``(hands, 8)``: driven hands at home, others aside."""
  return np.stack([home(slot) if slot in driven else aside(slot) for slot in SLOTS])


def _yaw(yaw: Float64Array) -> Float64Array:
  zero = np.zeros_like(yaw)
  return np.stack([np.cos(yaw / 2), zero, zero, np.sin(yaw / 2)], -1)


Placement = tuple[Float64Array, Float64Array, Float64Array]
"""Bricks on the tray: their ``(k, 2)`` plan positions, ``(k,)`` heights and
``(k, 4)`` quaternions."""


def _single(rng: np.random.Generator, count: int) -> Placement:
  xy = scene.TRAY_CENTER + rng.uniform(-1, 1, (1, 2)) * WALLS
  return xy, np.full(1, REST_Z), _yaw(rng.uniform(0, np.pi, 1))


def _pile(rng: np.random.Generator, count: int) -> Placement:
  center = scene.TRAY_CENTER + rng.uniform(-1, 1, 2) * np.array([0.05, 0.08])
  xy = center + rng.normal(0, 0.015, (count, 2))
  z = np.asarray(
    REST_Z + 0.003 + rng.permutation(count) * (scene.BRICK[2] + 0.006), np.float64
  )
  tilt = rng.uniform(-0.15, 0.15, (count, 3)) * np.array([1.0, 1.0, 0.0])
  return (
    xy,
    z,
    rotation.multiply(rotation.exp(tilt), _yaw(rng.uniform(0, np.pi, count))),
  )


def _drop(rng: np.random.Generator, count: int) -> Placement:
  xy = scene.TRAY_CENTER + rng.uniform(-1, 1, (count, 2)) * WALLS
  z = REST_Z + rng.uniform(*DROP_HEIGHT, count)
  for j in range(count):
    under = np.linalg.norm(xy[:j] - xy[j], axis=1) < DROP_CLEAR
    if under.any():
      z[j] = max(z[j], z[:j][under].max() + DROP_STACK)
  tilt = rng.uniform(-DROP_TILT, DROP_TILT, (count, 3)) * np.array([1.0, 1.0, 0.0])
  return (
    xy,
    z,
    rotation.multiply(rotation.exp(tilt), _yaw(rng.uniform(0, np.pi, count))),
  )


MAKERS: dict[str, Callable[[np.random.Generator, int], Placement]] = {
  "drop": _drop,
  "single": _single,
  "pile": _pile,
}


def parked(colors: tuple[str, ...]) -> Float64Array:
  """A spot on its own bin's floor for every brick, lying flat, ``(k, 7)``."""
  out = np.zeros((len(colors), 7))
  for index, color in enumerate(colors):
    out[index, :3] = scene.parked(color, colors[:index].count(color))
    out[index, 3] = 1.0
  return out


def draw(
  mode: str, seed: int, colors: tuple[str, ...], bricks: int | None = None
) -> Layout:
  """One tray laid out ``mode``'s way from ``seed``, ``bricks`` of ``colors`` on it."""
  if mode not in MAKERS:
    raise ValueError(f"no layout {mode!r}; one of {MODES}")
  rng = np.random.default_rng(seed)
  if mode == "single":
    count = 1
  else:
    count = int(rng.integers(DROP[0], DROP[1] + 1)) if bricks is None else bricks
  if not 1 <= count <= len(colors):
    raise ValueError(f"{count} bricks on a tray of {len(colors)}")
  active = np.zeros(len(colors), np.bool_)
  active[rng.choice(len(colors), count, replace=False)] = True
  poses = parked(colors)
  poses[active, :2], poses[active, 2], poses[active, 3:] = MAKERS[mode](rng, count)
  return Layout(mode, seed, bricks, poses, active)
