"""The camera a gripper carries: how its lens projects, and where it sits.

Intrinsics are held at the width they were measured at and scaled to whatever
the caller renders.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from specialist_sim.mei import MeiParams, rescale
from specialist_sim.spaces import MASK_RADIUS

__all__ = ["SCENE_VIEW", "CameraCfg"]


@dataclass(frozen=True)
class CameraCfg:
  """One gripper's lens, defaulting to a nominal camera."""

  params: MeiParams = MeiParams(
    fx=2250.0,
    fy=2250.0,
    cx=1456.9354838709678,
    cy=1450.1612903225807,
    k1=0.61060095,
    k2=0.17060375,
    k3=-2.69401383,
    p1=-0.00210447,
    p2=-0.00060749,
    xi=1.94817,
  )

  width: int = 2880
  """Width the intrinsics were measured at."""

  pos: tuple[float, float, float] = (
    -0.014537817405,
    -0.000731615839,
    0.036328428298,
  )
  """Where the lens sits in the frame of the body carrying it."""

  quat: tuple[float, float, float, float] = (
    -0.4290814654677777,
    0.5837380175668424,
    -0.5669007222884379,
    0.392125737370049,
  )
  """How the lens is turned: x right, y down, z forward."""

  mask_radius: float | None = MASK_RADIUS
  """How much of the frame this camera keeps, or all of it when there is none."""

  def at(self, width: int) -> MeiParams:
    """The same lens rendered ``width`` pixels across."""
    return rescale(self.params, width / self.width)

  def mount(self, model: mujoco.MjModel, body: int) -> None:
    """Move a lens body onto this camera's seat."""
    model.body_pos[body] = self.pos
    model.body_quat[body] = self.quat


SCENE_VIEW = CameraCfg(
  params=MeiParams(fx=520.0, fy=520.0, cx=240.0, cy=240.0),
  width=480,
  pos=(0.02, 0.0, 0.06),
  quat=(0.312046526844, -0.634528931637, 0.634528931637, -0.312046526844),
  mask_radius=None,
)
"""A pinhole across the room, for watching an episode."""
