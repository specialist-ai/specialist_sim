"""Where the LEGO setup stands, in :mod:`specialist_sim.tasks.desk`'s frame.

The bins are AkroBins 30210 (Akro-Mils), a red and a blue one, backed onto the
desk's far edge at +x and open toward the operator.
"""

import numpy as np

from specialist_sim.arrays import BoolArray, Float64Array, Int64Array
from specialist_sim.tasks import desk

TRAY_FLOOR_THICKNESS = 0.005
# The tray floor's top, where bricks rest.
TRAY_FLOOR = desk.TOP + TRAY_FLOOR_THICKNESS

TRAY_CENTER = np.array([-0.08, 0.0])
# Outer size along x and y; the TILLGANG tray's short side runs along x.
TRAY_SIZE = (0.29, 0.37)
# How far in from the tray's outer edge its flat floor begins.
TRAY_RIM = 0.025
# The rim's top, above the desk.
TRAY_LIP = 0.015
# The rim's inner wall's slope from the horizontal.
TRAY_WALL_ANGLE = np.pi / 4

# A 2 x 4 brick's collision box: long, wide, and tall to the studs' tops.
BRICK = (0.03175, 0.015875, 0.009525 + 0.00185)
BRICK_MASS = 0.0023

# The x of the desk's far edge.
DESK_EDGE = desk.SIZE[0] / 2
# Each bin's middle across the desk, by the color it takes.
BIN_Y = {"red": -0.0565, "blue": 0.0565}
COLORS = tuple(BIN_Y)

# A bin's own frame: x from its back toward the operator, z up, the origin on the
# desk under its middle. BIN_TURN, a half turn about z, stands it in the room.
BIN_TURN = (0.0, 0.0, 0.0, 1.0)
# Overall length along x, width and height.
BIN_SIZE = (0.1379, 0.1047, 0.0762)
# The flat floor from the back wall's inner face to the foot of the 45 degree
# front ramp: its back and front x, and its top above the desk.
BIN_FLOOR = (-0.0616, 0.0333, 0.00485)
# Where a sorted brick lies, up the ramp to the label lip's front face: back and
# front x, half the width, and the top above the desk.
BIN_INSIDE = (BIN_FLOOR[0], 0.0660, 0.04365, 0.0717)
# Every brick's color, by its index.
BRICK_COLORS = ("red",) * 6 + ("blue",) * 6
# Parked bricks, long side along x: how many along x and across y (room for
# BRICK_COLORS's six of a color), and their pitch along x and across y.
PARKING = (2, 3, 0.045, 0.027)


def bin_middle(color: str) -> Float64Array:
  """The bin's origin in plan, in the room, ``(2,)``."""
  return np.array([DESK_EDGE - BIN_SIZE[0] / 2, BIN_Y[color]])


def parked(color: str, slot: int) -> tuple[float, float, float]:
  """The ``slot``-th spot on the bin's floor, where a brick not in play waits."""
  x, y = bin_middle(color)
  along, across, pitch_along, pitch_across = PARKING
  # Along the bin's own x.
  own = (BIN_FLOOR[0] + BIN_FLOOR[1]) / 2 + (
    slot % along - (along - 1) / 2
  ) * pitch_along
  return (
    float(x - own),
    float(y - (slot // along - (across - 1) / 2) * pitch_across),
    desk.TOP + BIN_FLOOR[2] + 0.0025 + BRICK[2] / 2,
  )


def in_bin(position: Float64Array) -> Int64Array:
  """The bin each point, ``(..., 3)``, lies inside below its rim, as an index into
  ``COLORS``, ``(...,)``; ``-1`` for none."""
  back, front, half_width, top = BIN_INSIDE
  out = np.full(position.shape[:-1], -1)
  for index, color in enumerate(COLORS):
    offset = position[..., :2] - bin_middle(color)
    along = -offset[..., 0]
    inside = (
      (back < along)
      & (along < front)
      & (np.abs(offset[..., 1]) < half_width)
      & (desk.TOP < position[..., 2])
      & (position[..., 2] < desk.TOP + top)
    )
    out[inside] = index
  return out


def on_tray(position: Float64Array) -> BoolArray:
  """Whether each point, ``(..., 3)``, lies over the tray, low enough to be on it."""
  half = np.asarray(TRAY_SIZE) / 2
  out: BoolArray = (np.abs(position[..., :2] - TRAY_CENTER) < half).all(-1) & (
    position[..., 2] < desk.TOP + 0.05
  )
  return out
