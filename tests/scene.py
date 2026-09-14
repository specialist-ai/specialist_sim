"""A bare scene for the rig's own tests: a slab far below the hands, and
nothing else. The hands' own homes come with it."""

import mujoco
import numpy as np

SLAB = """
<mujoco>
  <worldbody>
    <geom name="slab" type="box" pos="0.5 0 -0.5" size="1 1 0.02" group="1" rgba="0.8 0.8 0.75 1"/>
  </worldbody>
</mujoco>
"""
HOME = np.array(
  [
    [0.45, 0.15, -0.10, 1.0, 0.0, 0.0, 0.0, 0.2],
    [0.45, -0.15, -0.10, 0.0, 0.0, 0.0, 1.0, 0.2],
  ]
)
"""Where the hands rest over the slab."""


def bare() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_string(SLAB)
