"""Where an episode starts: the hands, and the bricks' layout drawn from a seed."""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array
from specialist_sim.gripper import ACROSS, HANDS
from specialist_sim.tasks import desk
from specialist_sim.tasks.sort_lego.measures import (
  BRICK,
  BRICK_COLORS,
  TRAY_CENTER,
  TRAY_FLOOR,
  TRAY_RIM,
  TRAY_SIZE,
  parked,
)

# The height a brick's middle is laid at on the tray.
LAID = TRAY_FLOOR + BRICK[2] / 2 + desk.GAP
# How far a brick's middle can lie from the tray's, along x and y, with the brick
# in the well.
WALLS = np.asarray(TRAY_SIZE) / 2 - TRAY_RIM - BRICK[0] / 2
# How far over the tray floor a hand starts.
HOME_OVER_TRAY = 0.22


def home(hand: str) -> Float64Array:
  """Where a driven hand starts, ``(8,)``: over the tray's near corner on its
  side, fingers aimed at the middle of the tray floor."""
  corner = np.array(
    [
      TRAY_CENTER[0] - TRAY_SIZE[0] / 2,
      TRAY_CENTER[1] + TRAY_SIZE[1] / 2 * ACROSS[hand],
      TRAY_FLOOR + HOME_OVER_TRAY,
    ]
  )
  middle = np.array([*TRAY_CENTER, TRAY_FLOOR])
  return np.array([*corner, *rotation.pointing(middle - corner), 0.0])


def aside(hand: str) -> Float64Array:
  """Where a hand nobody drives waits, ``(8,)``: a meter out on its side and
  below the desk top, out of every lens."""
  down = rotation.pointing(np.array([0.0, 0.0, -1.0]))
  return np.array([0.0, ACROSS[hand], desk.TOP - 0.3, *down, 0.0])


def starts(driven: tuple[str, ...]) -> Float64Array:
  """Every hand's start, ``(hands, 8)``: driven hands at home, others aside."""
  return np.stack([home(hand) if hand in driven else aside(hand) for hand in HANDS])


# Bricks on the tray: their (k, 2) plan positions, (k,) heights and (k, 4) quaternions.
Placement = tuple[Float64Array, Float64Array, Float64Array]


def _tilted(rng: np.random.Generator, count: int, tilt: float) -> Float64Array:
  """``count`` turns about z at random, each then tipped up to ``tilt`` about x and
  about y."""
  tipped = rng.uniform(-tilt, tilt, (count, 3)) * np.array([1.0, 1.0, 0.0])
  yaw = np.outer(rng.uniform(0.0, np.pi, count), [0.0, 0.0, 1.0])
  return rotation.multiply(rotation.exp(tipped), rotation.exp(yaw))


def _pile(rng: np.random.Generator, count: int) -> Placement:
  """The bricks heaped near the middle, one above the other."""
  center = TRAY_CENTER + rng.uniform(-1, 1, 2) * np.array([0.05, 0.08])
  xy = center + rng.normal(0, 0.015, (count, 2))
  z = np.asarray(LAID + 0.003 + rng.permutation(count) * (BRICK[2] + 0.006), np.float64)
  return xy, z, _tilted(rng, count, 0.15)


def _drop(rng: np.random.Generator, count: int) -> Placement:
  """The collection task's layout: bricks anywhere in the well, a little tilted,
  each dropped from 1 to 4 cm, higher over any earlier brick it overlaps."""
  xy = TRAY_CENTER + rng.uniform(-1, 1, (count, 2)) * WALLS
  z = LAID + rng.uniform(0.01, 0.04, count)
  for j in range(count):
    # Twice the half diagonal of a brick tilted 0.25 rad, with a margin.
    under = np.linalg.norm(xy[:j] - xy[j], axis=1) < 0.04
    if under.any():
      # Over a brick tilted 0.25 rad, which stands 20 mm tall.
      z[j] = max(z[j], z[:j][under].max() + 0.025)
  return xy, z, _tilted(rng, count, 0.25)


class Mode(NamedTuple):
  """A tray layout: its maker, and the seconds it rests before the hands move."""

  place: Callable[[np.random.Generator, int], Placement]
  settle: float


# A drop and its tumble take under a second.
MODES = {"drop": Mode(_drop, 1.0), "pile": Mode(_pile, 0.6)}


def _waiting() -> Float64Array:
  """A spot on its own bin's floor for every brick, lying flat, ``(k, 7)``."""
  out = np.zeros((len(BRICK_COLORS), 7))
  for index, color in enumerate(BRICK_COLORS):
    out[index, :3] = parked(color, BRICK_COLORS[:index].count(color))
    out[index, 3] = 1.0
  return out


def draw(mode: str, rng: np.random.Generator, bricks: int | None) -> Float64Array:
  """Every brick's pose, ``(k, 7)``: ``bricks`` of them on the tray, laid
  out ``mode``'s way, 1 to 8 when ``bricks`` is ``None``, and the rest parked."""
  count = int(rng.integers(1, 8 + 1)) if bricks is None else bricks
  active = np.zeros(len(BRICK_COLORS), np.bool_)
  active[rng.choice(len(BRICK_COLORS), count, replace=False)] = True
  poses = _waiting()
  poses[active, :2], poses[active, 2], poses[active, 3:] = MODES[mode].place(rng, count)
  return poses
