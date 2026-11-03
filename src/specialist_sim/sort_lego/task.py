"""Where each brick is, read off the physics: sorted in its own bin, missorted
in the other, lost off the desk, or on the tray. A brick in a hand over its bin
is none of these.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.sort_lego import scene

__all__ = ["Score", "in_bin", "on_tray", "score"]

LOST = 0.05
"""How far below the desk top a brick has fallen off the desk, meters."""


def in_bin(position: Float64Array, bins: str) -> str | None:
  """The bin of kind ``bins`` a point lies inside, below its rim, or None."""
  return scene.BINS[bins].holding(position)


def on_tray(position: Float64Array) -> bool:
  half = np.asarray(scene.TRAY_SIZE) / 2
  return bool(
    np.all(np.abs(position[:2] - scene.TRAY_CENTER) < half)
    and position[2] < scene.DESK_TOP + 0.05
  )


@dataclass(frozen=True)
class Score:
  """How a tray stands, counted over the bricks that started on it."""

  started: int
  sorted: int
  missorted: int
  on_tray: int
  lost: int

  @property
  def cleared(self) -> bool:
    """Whether every brick that started on the tray is sorted: success."""
    return self.sorted == self.started

  @property
  def progress(self) -> float:
    """The share of the started bricks that are sorted."""
    return self.sorted / self.started

  def as_dict(self) -> dict[str, int | float | bool]:
    return {**asdict(self), "cleared": self.cleared, "progress": self.progress}


def score(
  bricks: Float64Array, colors: tuple[str, ...], active: BoolArray, bins: str
) -> Score:
  """The tray's standing from every brick's pose ``(k, 7)``, their ``colors``,
  and which of them started on the tray, ``active`` ``(k,)``, sorting into
  bins of kind ``bins``."""
  counts = {"sorted": 0, "missorted": 0, "on_tray": 0, "lost": 0}
  for pose, color in zip(bricks[active], np.array(colors)[active], strict=True):
    landed = in_bin(pose[:3], bins)
    if landed is not None:
      counts["sorted" if landed == color else "missorted"] += 1
    elif pose[2] < scene.DESK_TOP - LOST:
      counts["lost"] += 1
    elif on_tray(pose[:3]):
      counts["on_tray"] += 1
  return Score(started=int(active.sum()), **counts)
