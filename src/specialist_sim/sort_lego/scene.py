"""Where the 0927 LEGO setup stands, in the room frame of its takes.

The room frame is the corpus's: the midpoint of the two anchor tags is the origin, +y runs
from tag 4 to tag 5, +z is up. In it the operator stands at +x and the bins back onto the far
edge of the desk at -x. Every number here was measured off the two takes of
``0927_sort_lego_simple`` by ``specialist_rlft``'s ``scripts/build_demos.py`` (see its report),
none is a guess from a product page except the articles' own sizes. This file is that repo's
``scene.py`` unchanged, so a sim episode here stands where its demonstrations and experts do.

* The tray floor, from the grasps: a held brick's middle, read off the gripper at every
  closing, averages z = -0.397 and -0.400 in the two takes, so the floor is half a brick
  below -0.398. The vision layouts put it at -0.406 and -0.397 -- the 9 mm between those is
  the first frame's camera poses, which the grasps, spread over the whole take, do not share.
* The tray, from its dark floor seen by the left lens: centered at (-0.338, -0.063) and
  (-0.341, -0.052), 37 x 29 cm with its long side along y.
* The bins, from their cards: the near faces at x = -0.567, the blue card at y = -0.111 and
  the red at y = +0.002.
"""

import numpy as np

TRAY_FLOOR = -0.404
"""The height bricks rest at on the tray, m."""
TRAY_FLOOR_THICKNESS = 0.005
DESK_TOP = TRAY_FLOOR - TRAY_FLOOR_THICKNESS
"""The desk's surface: the tray and bins stand on it."""
DESK_HEIGHT = 0.73

TRAY_CENTER = np.array([-0.3395, -0.0575])
TRAY_SIZE = (0.29, 0.37)
"""Outer size across x and across y: the TILLGANG's short side runs along x here."""
TRAY_RIM = 0.025
"""How far in from the tray's outer edge its flat floor begins, m."""
TRAY_LIP = 0.015
"""The rim's top, above the desk, m: measured on the tray."""
TRAY_WALL_ANGLE = np.radians(45.0)
"""The rim's inner wall slopes up from the floor at this angle to the horizontal, measured on
the tray: a brick pushed against it rides up the slope instead of wedging in a corner."""

BIN_SIZE = (0.152, 0.106, 0.082)
"""A bin's outside, along x (its long side, toward the backdrop), along y, and tall."""
BIN_WALL = 0.002
BIN_NEAR_FACE = -0.567
BINS = {
  "blue": np.array([BIN_NEAR_FACE - BIN_SIZE[0] / 2, -0.111]),
  "red": np.array([BIN_NEAR_FACE - BIN_SIZE[0] / 2, 0.002]),
}
"""Each bin's middle in plan, by the color it takes."""
COLORS = ("red", "blue")

DESK_SIZE = (0.60, 1.20)
DESK_CENTER = np.array(
  [
    BIN_NEAR_FACE - BIN_SIZE[0] + DESK_SIZE[0] / 2,
    float(np.mean([b[1] for b in BINS.values()])),
  ]
)
"""The bins back onto the desk's far edge with their shared wall on its middle."""

BRICK = (0.03175, 0.015875, 0.009525 + 0.00185)
"""A 2 x 4 brick's collision box: long, wide, and tall to the studs' tops, m."""
BRICK_MASS = 0.0023


def bin_center(color: str) -> np.ndarray:
  """A bin's middle, on its floor, in the room."""
  return np.array([*BINS[color], DESK_TOP + BIN_WALL])


def parked(color: str, slot: int) -> tuple[float, float, float]:
  """The ``slot``-th spot on a bin's floor, three along and two across: where a brick not in
  play waits, as if already sorted."""
  x, y = BINS[color]
  return (
    float(x + (slot % 3 - 1) * 0.045),
    float(y + (slot // 3 - 0.5) * 0.05),
    DESK_TOP + 2 * BIN_WALL + BRICK[2] / 2 + 5e-4,
  )
