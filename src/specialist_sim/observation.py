"""What the policy sees: each camera's image, and where its own hand has been.

A row is one tick of the row lattice, ``f_row``. A control step spans several of
them and reports them all, so a policy reads a hand at the rate a corpus
records one rather than at the rate it is asked to act.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.spaces import GRIPPERS

__all__ = ["flat_row", "flat_rows"]


def flat_row(
  observation: dict[str, Any], row: int = -1, world: int = 0
) -> Float64Array:
  """One world's hands at one row, laid out like an action."""
  one: Float64Array = flat_rows(observation, row)[world]
  return one


def flat_rows(observation: dict[str, Any], row: int = -1) -> Float64Array:
  """Every world's hands at one row, laid out like an action: the hands the
  observation carries, in the rig's order."""
  out: Float64Array = np.concatenate(
    [
      np.concatenate(
        [
          observation[slot]["pose"][:, row],
          observation[slot]["opening"][:, row, None],
        ],
        axis=1,
      )
      for slot in GRIPPERS
      if "pose" in observation.get(slot, {})
    ],
    axis=1,
  ).astype(np.float64)
  return out
