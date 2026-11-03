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

The rig's bins are AkroBins 30210 (Akro-Mils), a red and a blue one; the clear bins are the
ones the 0927 takes recorded. The AkroBins' sizes are the product's: the CAD is MiniTec's
32.9540, the same bin, whose STEP matches the catalog's 5-3/8 x 4-1/8 x 3 in and 4-3/4 x
3-7/16 in inside. Where they stand is not measured yet: like the clear bins they back onto
the desk's far edge, open front toward the operator, on the clear bins' middles across y.
"""

from dataclasses import dataclass

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

BRICK = (0.03175, 0.015875, 0.009525 + 0.00185)
"""A 2 x 4 brick's collision box: long, wide, and tall to the studs' tops, m."""
BRICK_MASS = 0.0023

COLORS = ("red", "blue")
BIN_Y = {"blue": -0.111, "red": 0.002}
"""Each bin's middle across the desk, by the color it takes."""
PARK_DROP = 0.002
"""How far above its bin's floor a parked brick is laid, to settle onto it, m."""


@dataclass(frozen=True)
class Bins:
  """A kind of bin, one of each color standing side by side on the desk.

  Lengths are in a bin's own frame: x from its back toward the operator, y across it, z
  up from the desk, the origin on the desk under the middle of its overall length and width.
  """

  near_face: float
  """The room x of the face toward the operator."""
  size: tuple[float, float, float]
  """Overall length along x, width and height, m."""
  floor: tuple[float, float, float]
  """The flat floor: its back and front x, and its top above the desk."""
  inside: tuple[float, float, float, float]
  """Where a sorted brick lies: back and front x, half the width, and the top above the desk."""
  parking: tuple[int, int, float, float]
  """Parked bricks, long side along x: how many along x and across y, and their pitch along
  x and across y."""

  def middle(self, color: str) -> np.ndarray:
    """The bin's origin in plan, in the room."""
    return np.array([self.near_face - self.size[0] / 2, BIN_Y[color]])

  def floor_middle(self, color: str) -> np.ndarray:
    """The middle of the bin's flat floor, on it, in the room."""
    x, y = self.middle(color)
    return np.array(
      [x + (self.floor[0] + self.floor[1]) / 2, y, DESK_TOP + self.floor[2]]
    )

  def parked(self, color: str, slot: int) -> tuple[float, float, float]:
    """The ``slot``-th spot on the bin's floor: where a brick not in play waits, as if
    already sorted."""
    x, y = self.middle(color)
    along, across, pitch_along, pitch_across = self.parking
    return (
      float(
        x
        + (self.floor[0] + self.floor[1]) / 2
        + (slot % along - (along - 1) / 2) * pitch_along
      ),
      float(y + (slot // along - (across - 1) / 2) * pitch_across),
      DESK_TOP + (self.floor[2] + PARK_DROP) + BRICK[2] / 2 + 5e-4,
    )

  def holding(self, position: np.ndarray) -> str | None:
    """The color of the bin a point lies inside, below its rim, or None."""
    back, front, half_width, top = self.inside
    for color in BIN_Y:
      x, y = self.middle(color)
      if (
        back < position[0] - x < front
        and abs(position[1] - y) < half_width
        and DESK_TOP < position[2] < DESK_TOP + top
      ):
        return color
    return None


CLEAR_WALL = 0.002
BINS = {
  # The 0927 takes' clear bins: a 2 mm shell with the color's card on its near face.
  "clear": Bins(
    near_face=-0.567,
    size=(0.152, 0.106, 0.082),
    floor=(-(0.152 / 2 - CLEAR_WALL), 0.152 / 2 - CLEAR_WALL, CLEAR_WALL),
    inside=(
      -(0.152 / 2 - CLEAR_WALL),
      0.152 / 2 - CLEAR_WALL,
      0.106 / 2 - CLEAR_WALL,
      0.082,
    ),
    parking=(3, 2, 0.045, 0.05),
  ),
  # AkroBins 30210, measured off the STEP: the floor runs from the back wall's inner face to
  # the foot of the 45 degree front ramp, 4.85 mm up on the runners; inside, a brick may lie
  # on that ramp up to the label lip's front face, between the side walls, below their top.
  "akrobins": Bins(
    near_face=-0.567 - 0.152 + 0.1379,
    size=(0.1379, 0.1047, 0.0762),
    floor=(-0.0616, 0.0333, 0.00485),
    inside=(-0.0616, 0.0660, 0.04365, 0.0717),
    parking=(2, 3, 0.045, 0.027),
  ),
}
"""Every kind of bin the scene can stand, by name. Both kinds back onto the desk's far edge."""

DESK_SIZE = (0.60, 1.20)
DESK_CENTER = np.array(
  [
    BINS["clear"].near_face - BINS["clear"].size[0] + DESK_SIZE[0] / 2,
    float(np.mean(list(BIN_Y.values()))),
  ]
)
"""The clear bins backed onto the desk's far edge with their shared wall on its middle."""
DESK_EDGE = BINS["clear"].near_face - BINS["clear"].size[0]
"""The desk's far edge, toward the backdrop."""
