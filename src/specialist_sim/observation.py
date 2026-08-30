"""What the policy sees: each camera's image, and where its own hand has been.

A row is one tick of the step timeline. A control step spans several of them and
reports them all, so a policy reads a hand at the rate a dataset records one
rather than at the rate it is asked to act.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from specialist_sim.arrays import Float32Array, Float64Array
from specialist_sim.scene import Scene
from specialist_sim.spaces import GRIPPERS

__all__ = ["flat_row", "read_hands"]


def read_hands(scene: Scene) -> tuple[Float32Array, Float32Array]:
  """Every hand's pose and finger angle, as the physics leaves them.

  A rig is a free body, so its pose is its joint coordinates and reading it
  needs no kinematics pass. One taken between physics steps would refresh the
  frames the impedance is still reading and move the plant.
  """
  data, hands = scene.data, len(scene.grippers)
  poses, openings = np.empty((hands, 7)), np.empty(hands)
  for hand, grip in enumerate(scene.grippers):
    poses[hand] = data.qpos[grip.qpos : grip.qpos + 7]
    openings[hand] = data.qpos[grip.fingers[0]]
  return poses.astype(np.float32), openings.astype(np.float32)


def flat_row(observation: dict[str, Any], row: int = -1) -> Float64Array:
  """Every hand's pose and opening at one row, laid out like an action."""
  out: Float64Array = np.concatenate(
    [
      np.concatenate(
        [observation[slot]["pose"][row], [observation[slot]["opening"][row]]]
      )
      for slot in GRIPPERS
    ]
  ).astype(np.float64)
  return out
