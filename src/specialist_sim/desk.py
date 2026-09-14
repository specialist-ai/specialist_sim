"""The desk every task stands on. The origin is on the floor under its middle,
+z up, the near side at +x."""

from __future__ import annotations

import mujoco

__all__ = ["CENTER", "FRICTION", "HEIGHT", "SIZE", "THICKNESS", "TOP", "stand"]

SIZE = (0.60, 1.20)
"""The top, across x and across y, meters."""
THICKNESS = 0.025
HEIGHT = 0.70
"""How far the top stands over the floor, meters."""
TOP = HEIGHT
"""The top's height: the origin is on the floor under it."""
CENTER = (0.0, 0.0)
"""Where the top's middle stands, in plan."""
FRICTION = (0.26, 0.005, 0.0001)
"""The top's laminate."""
LEG = 0.02
"""Half a leg's width, meters."""
INSET = 0.05
"""How far a leg's middle stands in from the top's corner, meters."""
FLOOR = 1.5
"""Half the floor's side, meters."""
FLOOR_THICKNESS = 0.01


def stand(body: mujoco.MjsBody) -> None:
  """The top, four legs that collide with nothing, and a floor that catches
  whatever leaves the top and is never drawn."""
  body.add_geom(
    name="desk_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[CENTER[0], CENTER[1], TOP - THICKNESS / 2],
    size=[SIZE[0] / 2, SIZE[1] / 2, THICKNESS / 2],
    rgba=[0.93, 0.93, 0.92, 1.0],
    group=2,
    friction=list(FRICTION),
  )
  body.add_geom(
    name="room_floor",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[CENTER[0], CENTER[1], -FLOOR_THICKNESS],
    size=[FLOOR, FLOOR, FLOOR_THICKNESS],
    group=3,
  )
  for near in (-1, 1):
    for left in (-1, 1):
      body.add_geom(
        name=f"desk_leg_{'n' if near > 0 else 'f'}{'l' if left > 0 else 'r'}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[
          CENTER[0] + near * (SIZE[0] / 2 - INSET),
          CENTER[1] + left * (SIZE[1] / 2 - INSET),
          TOP - HEIGHT / 2,
        ],
        size=[LEG, LEG, HEIGHT / 2 - THICKNESS],
        rgba=[0.6, 0.6, 0.6, 1.0],
        group=2,
        contype=0,
        conaffinity=0,
      )
