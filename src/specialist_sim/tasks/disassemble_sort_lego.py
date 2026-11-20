"""The LEGO disassembly and sorting task: pairs of bricks clutched together on
the tray of the LEGO sorting room, to be pulled apart and sorted into their
colors' bins."""

from dataclasses import dataclass, replace
from typing import ClassVar

import mujoco
import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array
from specialist_sim.env import Bond, Lighting, Objects, Task, Verdict
from specialist_sim.tasks.sort_lego import SortLego, initial, scene
from specialist_sim.tasks.sort_lego.measures import (
  BRICK,
  BRICK_COLORS,
  NESTED,
  STUD_PITCH,
  STUD_RISE,
  TRAY_CENTER,
  TRAY_FLOOR,
  TRAY_RIM,
  TRAY_SIZE,
  in_bin,
)

# How far a clutched brick stands clear of the body under it, so a pair touches
# nowhere and its bond bears every load between its bricks.
CLEAR = 5e-5
# A brick clutched two studs along the one under it, as in a stair: its pose in
# the lower brick's frame; the face where the two overlap, on the lower brick's
# body, by its middle and its half sizes; and the newtons its four studs take
# together before they give, drawn between the two for each episode.
CLUTCH = (2 * STUD_PITCH, 0.0, NESTED + CLEAR, 1.0, 0.0, 0.0, 0.0)
ANCHOR = (STUD_PITCH, 0.0, BRICK[2] / 2 - STUD_RISE + CLEAR / 2)
FACE = (BRICK[0] / 2 - STUD_PITCH, BRICK[1] / 2)
PULL = (30.0, 40.0)
# Each pair's red and blue brick, by their index, the same rank in each color.
PAIRS = tuple(
  zip(
    [i for i, color in enumerate(BRICK_COLORS) if color == "red"],
    [i for i, color in enumerate(BRICK_COLORS) if color == "blue"],
    strict=True,
  )
)
# The pair each brick belongs to.
PAIR_OF = np.array(
  [
    next(k for k, pair in enumerate(PAIRS) if i in pair)
    for i in range(len(BRICK_COLORS))
  ]
)
# The middle of a pair, between its bricks' middles, in its lower brick's frame,
# and every corner of its bricks' boxes about it.
MIDDLE = np.array(CLUTCH[:3]) / 2
CORNERS = np.array(
  [
    [x, y, z] + shift - MIDDLE
    for shift in (np.zeros(3), np.array(CLUTCH[:3]))
    for x in (-BRICK[0] / 2, BRICK[0] / 2)
    for y in (-BRICK[1] / 2, BRICK[1] / 2)
    for z in (-BRICK[2] / 2, BRICK[2] / 2)
  ]
)
# How far a pair reaches from its middle, however turned, and how far apart
# pairs' middles are laid, so none starts in another.
REACH = float(np.linalg.norm(CORNERS, axis=1).max())
SPACING = 2 * REACH + 0.005


def _bond(base: int, part: int) -> Bond:
  return Bond(f"brick_{base}", f"brick_{part}", CLUTCH, ANCHOR, FACE, PULL)


# Every pair's red brick clutched on its blue one, then its blue on its red.
BONDS = tuple(
  bond for red, blue in PAIRS for bond in (_bond(blue, red), _bond(red, blue))
)


def lay(rng: np.random.Generator, pairs: int | None) -> Float64Array:
  """Every brick's pose, ``(k, 7)``: ``pairs`` pairs, 1 to 3 when ``None``,
  either color on top, each turned at random and dropped from 1 to 3 cm onto
  the tray's floor, the rest parked in their bins."""
  count = int(rng.integers(1, 3 + 1)) if pairs is None else pairs
  poses = initial.waiting()
  within = np.asarray(TRAY_SIZE) / 2 - TRAY_RIM - REACH
  middles: list[Float64Array] = []
  while len(middles) < count:
    xy = TRAY_CENTER + rng.uniform(-1, 1, 2) * within
    if all(np.linalg.norm(xy - other) >= SPACING for other in middles):
      middles.append(xy)
  for k, xy in zip(rng.choice(len(PAIRS), count, replace=False), middles, strict=True):
    red, blue = PAIRS[k]
    base, part = (blue, red) if rng.integers(2) else (red, blue)
    turn = rng.normal(size=4)
    turn /= np.linalg.norm(turn)
    lowest = rotation.rotate(turn, CORNERS)[:, 2].min()
    middle = np.array([*xy, TRAY_FLOOR - lowest + rng.uniform(0.01, 0.03)])
    poses[base] = [*(middle - rotation.rotate(turn, MIDDLE)), *turn]
    poses[part] = rotation.compose(poses[base], np.asarray(CLUTCH))
  return poses


@dataclass(frozen=True)
class DisassembleSortLego(Task):
  """Pull apart every pair of bricks that starts clutched on the tray, and sort
  each brick into the bin of its color. It is won once every brick rests in
  its bin, apart and let go of, and lost as :class:`SortLego` is lost."""

  pairs: int | None = None
  """How many pairs a tray starts with. ``None`` draws 1 to 3."""

  lighting: ClassVar[Lighting] = SortLego.lighting
  fails: ClassVar[dict[str, tuple[str, ...]]] = SortLego.fails
  bonds: ClassVar[tuple[Bond, ...]] = BONDS
  focus: ClassVar[Float64Array] = SortLego.focus

  @property
  def settle(self) -> float:
    """A dropped pair's tumble."""
    return 1.0

  def spec(self) -> mujoco.MjSpec:
    return scene.spec()

  def draw(
    self, rng: np.random.Generator, driven: tuple[str, ...]
  ) -> tuple[Float64Array, Float64Array]:
    return initial.starts(driven), lay(rng, self.pairs)

  def judge(self, objects: Objects) -> Verdict:
    # Whether each pair still holds, by either bond; a brick its pair holds is
    # not let go of.
    joined = objects.bonded[:, 0::2] | objects.bonded[:, 1::2]
    sorting = SortLego().judge(replace(objects, held=objects.held | joined[:, PAIR_OF]))
    started = in_bin(objects.laid[:, [red for red, _ in PAIRS], :3]) < 0
    counts = {
      **sorting.counts,
      "pairs": started.sum(-1),
      "apart": (started & ~joined).sum(-1),
    }
    return Verdict(
      progress=(counts["apart"] + counts["sorted"])
      / (counts["pairs"] + counts["started"]),
      failed=sorting.failed,
      labels={"apart": counts["apart"] == counts["pairs"]},
      counts=counts,
    )
