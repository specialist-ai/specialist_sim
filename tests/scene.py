"""A bare scene for the rig's own tests: a slab far below the hands, and
nothing else."""

import mujoco

SLAB = """
<mujoco>
  <worldbody>
    <geom name="slab" type="box" pos="0.5 0 -0.5" size="1 1 0.02" rgba="0.8 0.8 0.75 1"/>
  </worldbody>
</mujoco>
"""


def bare() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_string(SLAB)
