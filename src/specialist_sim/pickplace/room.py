"""The room the recordings were made in, and what an env reads off it: the
table, the tape target, the cube, and how a hand takes hold of the cube."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from specialist_sim import gripper
from specialist_sim.arrays import Float64Array, Int64Array
from specialist_sim.camera import Camera, Mei
from specialist_sim.gripper import Hand

__all__ = ["ASSETS", "SCENE_VIEW", "Cube", "Room", "spec"]

ASSETS = Path(__file__).parent / "assets"

FIDUCIALS = ("tag_0", "tag_1")
"""The tag stands the room frame is measured from."""

CLEARANCE = 1e-4
"""How far above the table a placed cube starts, so it never begins inside it."""

SCENE_VIEW = Camera(
  mei=Mei(fx=520.0, fy=520.0, cx=240.0, cy=240.0),
  width=480,
  pos=(0.02, 0.0, 0.06),
  quat=(0.312046526844, -0.634528931637, 0.634528931637, -0.312046526844),
  mask=None,
)
"""A pinhole just above the tag line's midpoint, looking across the table,
for watching an episode."""


def spec(fiducials: bool = False) -> mujoco.MjSpec:
  """The room, its tag stands in it only when asked."""
  room = mujoco.MjSpec.from_file(str(ASSETS / "room.xml"))
  if not fiducials:
    for name in FIDUCIALS:
      room.delete(room.body(name))
  return room


@dataclass(frozen=True)
class Cube:
  body: int
  geom: int
  qpos: int
  """Where its free joint's position and quaternion start in ``qpos``."""
  side: float
  rest: float
  """The height of its center when it sits on the table."""


class Room:
  """The room's ids and measures in a compiled model the hands stand in."""

  def __init__(self, model: mujoco.MjModel, hands: tuple[Hand, ...]):
    self.model = model
    self.slots = tuple(hand.slot for hand in hands)
    self.table_top = float(model.body("table").pos[2] + model.geom("table").size[2])
    self.tape: Float64Array = np.array(model.body("tape").pos[:2])
    self.tape_half: Float64Array = np.array(model.geom("tape").size[:2])
    side = 2.0 * float(model.geom("cube").size[0])
    self.cube = Cube(
      body=model.body("cube").id,
      geom=model.geom("cube").id,
      qpos=int(model.jnt_qposadr[model.body("cube").jntadr[0]]),
      side=side,
      rest=self.table_top + side / 2.0,
    )
    self.pads: Int64Array = np.concatenate(
      [gripper.pads(model, hand) for hand in hands]
    )
    """Every gripping pad in the room, which is what a grasp touches with."""
    self.grasp = gripper.grasp(model, hands[0])
    """Where a hand's pads meet when shut, in its base frame."""
    self.pinch = self._pinch(hands[0])
    """The opening at which the pads just touch a cube seated between them."""

  def place_cube(self, data: mujoco.MjData, cube: Float64Array) -> None:
    """Rest the cube upright on the table at ``(x, y, yaw)``."""
    at = self.cube.qpos
    data.qpos[at : at + 3] = cube[0], cube[1], self.cube.rest + CLEARANCE
    data.qpos[at + 3 : at + 7] = np.cos(cube[2] / 2.0), 0.0, 0.0, np.sin(cube[2] / 2.0)

  def _pinch(self, hand: Hand) -> float:
    """Bisect the finger angle for the one at which the nearest pad just clears
    a cube seated where the pads meet."""
    model, data = self.model, mujoco.MjData(self.model)
    mujoco.mj_forward(model, data)
    seat = data.xpos[hand.base] + data.xmat[hand.base].reshape(3, 3) @ self.grasp
    data.qpos[self.cube.qpos : self.cube.qpos + 3] = seat
    pads = gripper.pads(model, hand)

    def gap(angle: float) -> float:
      data.qpos[list(hand.fingers)] = angle, -angle
      mujoco.mj_forward(model, data)
      return min(
        float(mujoco.mj_geomDistance(model, data, pad, self.cube.geom, 1.0, None))
        for pad in pads
      )

    low, high = gripper.SHUT, gripper.OPEN
    for _ in range(40):
      middle = 0.5 * (low + high)
      low, high = (low, middle) if gap(middle) > 0.0 else (middle, high)
    return float(high)
