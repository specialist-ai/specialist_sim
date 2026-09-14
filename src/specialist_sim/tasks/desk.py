"""The desk every task stands on. The origin is on the floor under its middle,
+z up. The operator stands at -x facing +x, the left hand at +y."""

import mujoco

from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.gripper import DRAWN

# The top along x and y.
SIZE = (0.60, 1.20)
THICKNESS = 0.025
# The top's height over the floor.
TOP = 0.70
FRICTION = (0.26, 0.005, 0.0001)
# How far below TOP an object has fallen off the desk.
FALLEN = 0.05
# How far above its rest an object is laid.
GAP = 5e-4
# The floor's top is at z = 0.
FLOOR_HALF_THICK = 0.01


def stand(body: mujoco.MjsBody) -> None:
  """The top, four legs that collide with nothing, and an undrawn floor that
  catches whatever leaves the top."""
  body.add_geom(
    name="desk_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[0.0, 0.0, TOP - THICKNESS / 2],
    size=[SIZE[0] / 2, SIZE[1] / 2, THICKNESS / 2],
    rgba=[0.93, 0.93, 0.92, 1.0],
    group=DRAWN,
    friction=list(FRICTION),
  )
  body.add_geom(
    name="room_floor",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[0.0, 0.0, -FLOOR_HALF_THICK],
    size=[1.5, 1.5, FLOOR_HALF_THICK],
    group=3,
  )
  for along in (-1, 1):
    for left in (-1, 1):
      body.add_geom(
        name=f"desk_leg_{'n' if along < 0 else 'f'}{'l' if left > 0 else 'r'}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[along * (SIZE[0] / 2 - 0.05), left * (SIZE[1] / 2 - 0.05), TOP / 2],
        size=[0.02, 0.02, TOP / 2 - THICKNESS],
        rgba=[0.6, 0.6, 0.6, 1.0],
        group=DRAWN,
        contype=0,
        conaffinity=0,
      )


def fallen(z: Float64Array) -> BoolArray:
  """Whether an object at each height has fallen off the desk."""
  return z < TOP - FALLEN
