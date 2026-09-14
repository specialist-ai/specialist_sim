"""A bare world for the rig's own tests: a lit slab far below the hands, and
nothing else."""

import mujoco

SLAB = """
<mujoco>
  <worldbody>
    <light dir="0.3 0.2 -0.9" diffuse="0.5 0.5 0.5" type="directional"/>
    <light castshadow="false" dir="-0.6 -0.5 -0.5" ambient="0.5 0.5 0.5" diffuse="0.3 0.3 0.3" type="directional"/>
    <geom name="slab" type="box" pos="0.5 0 -0.5" size="1 1 0.02" group="1" rgba="0.8 0.8 0.75 1"/>
  </worldbody>
</mujoco>
"""


def bare() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_string(SLAB)
