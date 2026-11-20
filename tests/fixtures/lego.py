"""Where the LEGO room's bricks are, and how a hand takes one."""

from typing import Any

import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.env import Env
from specialist_sim.tasks.sort_lego import initial
from specialist_sim.tasks.sort_lego.measures import BRICK_COLORS


def brick_q(env: Env[Any], brick: int) -> int:
  """Where one brick's pose starts in ``qpos``."""
  return int(env.model.joint(f"brick_{brick}_joint").qposadr[0])


def brick_geoms(env: Env[Any]) -> BoolArray:
  """Whether each geom is a brick's, ``(ngeom,)``."""
  model = env.model
  bodies = [model.body(f"brick_{i}").id for i in range(len(BRICK_COLORS))]
  return np.isin(model.geom_bodyid, bodies)


def grasp(brick: Float64Array, hand: str) -> Float64Array:
  """The hand state that holds ``brick`` between its pads, fingers along its
  long side and pointing away from the hand's home."""
  long_side = rotation.rotate(brick[3:7], np.array([1.0, 0.0, 0.0]))
  along = np.array([long_side[0], long_side[1], 0.0]) / np.linalg.norm(long_side[:2])
  if np.dot(along[:2], brick[:2] - initial.home(hand)[:2]) < 0.0:
    along = -along
  pitch = np.radians(48.0)  # the fingers tip down this far to reach a flat brick
  turn = rotation.pointing(np.cos(pitch) * along - np.array([0.0, 0.0, np.sin(pitch)]))
  # A held brick's middle in the base tag's frame: z at the pads' centers, y mid-jaw.
  at = brick[:3] - rotation.rotate(turn, np.array([0.098, 0.0, -0.0177]))
  return np.array([*at, *turn, 0.3])  # open just wide of the brick


def eased(plan: list[tuple[float, Float64Array]], at: float) -> Float64Array:
  """A hand state along ``plan``, smoothstepped between its arrival times."""
  if at <= plan[0][0]:
    return plan[0][1]
  for (first, a), (second, b) in zip(plan, plan[1:], strict=False):
    if at <= second:
      part = (at - first) / (second - first)
      smooth = part * part * (3.0 - 2.0 * part)
      out = a + smooth * (b - a)
      out[:7] = rotation.interpolate(a, b, smooth)
      return out
  return plan[-1][1]


def lift(
  brick: Float64Array, hand: str
) -> tuple[Float64Array, Float64Array, Float64Array, Float64Array]:
  """The hand states that take ``brick`` up 10 cm: over it, around it, shut on
  it and lifted."""
  around = grasp(brick, hand)
  above, shut = around.copy(), around.copy()
  above[2] += 0.06
  shut[7] = -0.15  # the commanded squeeze on the brick
  lifted = shut.copy()
  lifted[2] += 0.10
  return above, around, shut, lifted


def in_hand(state: Float64Array, brick: Float64Array) -> Float64Array:
  """Where ``brick`` stands in the frame of a hand at ``state``."""
  return rotation.compose(rotation.inverse(state), brick)
