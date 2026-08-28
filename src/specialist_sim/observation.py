"""What the policy sees: each camera's image, and where its own hand is."""

from __future__ import annotations

import numpy as np

from specialist_sim import pose as se3
from specialist_sim.action import opening_from_gripper
from specialist_sim.arrays import Float32Array
from specialist_sim.scene import Scene

__all__ = ["hand_state"]


def hand_state(scene: Scene) -> Float32Array:
  """Every hand's pose and opening, laid out like the action."""
  data = scene.data
  out = []
  for grip in scene.grippers:
    out.append(
      np.concatenate(
        [
          data.xpos[grip.rig],
          se3.wxyz(data.xmat[grip.rig].reshape(3, 3)),
          [opening_from_gripper(float(data.qpos[grip.fingers[0]]))],
        ]
      )
    )
  return np.concatenate(out).astype(np.float32)
