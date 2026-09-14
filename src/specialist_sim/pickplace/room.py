"""The room an episode of pick-and-place runs in: the desk, the tape target
stuck on it, and the cube, and what an env reads off them."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from specialist_sim import desk, gripper
from specialist_sim.arrays import Float64Array, Int64Array
from specialist_sim.gripper import Hand

__all__ = ["CUBE_HALF", "TAPE", "TAPE_HALF", "Cube", "Room", "spec"]

TAPE = (0.0, 0.0)
"""Where the tape target sticks, on the desk's middle."""
TAPE_HALF = 0.013
"""Half the tape square's side, meters."""
TAPE_THICKNESS = 0.0002
CUBE_HALF = 0.015
"""Half the cube's side, meters."""
CUBE_MASS = 0.012
CUBE_FRICTION = (0.26, 0.005, 0.0001)
"""The cube's plastic."""
CLEARANCE = 1e-4
"""How far above the desk a placed cube starts, meters."""


def spec() -> mujoco.MjSpec:
  """The desk, the tape stuck on it, and the cube on its free joint."""
  spec = mujoco.MjSpec()
  room = spec.worldbody.add_body(name="room")
  desk.stand(room)
  room.add_geom(
    name="tape",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[TAPE[0], TAPE[1], desk.TOP + TAPE_THICKNESS],
    size=[TAPE_HALF, TAPE_HALF, TAPE_THICKNESS],
    rgba=[0.15, 0.32, 0.70, 1.0],
    group=1,
    contype=0,
    conaffinity=0,
    mass=0.0,
  )
  cube = spec.worldbody.add_body(
    name="cube", pos=[TAPE[0], TAPE[1], desk.TOP + CUBE_HALF + CLEARANCE]
  )
  cube.add_joint(name="cube", type=mujoco.mjtJoint.mjJNT_FREE)
  cube.add_geom(
    name="cube",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=[CUBE_HALF] * 3,
    rgba=[0.75, 0.12, 0.1, 1.0],
    mass=CUBE_MASS,
    friction=list(CUBE_FRICTION),
    group=1,
  )
  return spec


@dataclass(frozen=True)
class Cube:
  body: int
  geom: int
  qpos: int
  """Where its free joint's position and quaternion start in ``qpos``."""
  rest: float
  """The height of its center when it sits on the desk."""


class Room:
  """The room's ids and measures in a compiled model the hands stand in."""

  def __init__(self, model: mujoco.MjModel, hands: tuple[Hand, ...]):
    self.model = model
    self.slots = tuple(hand.slot for hand in hands)
    self.table_top = desk.TOP
    self.tape: Float64Array = np.array(TAPE)
    self.tape_half: Float64Array = np.array([TAPE_HALF, TAPE_HALF])
    self.cube = Cube(
      body=model.body("cube").id,
      geom=model.geom("cube").id,
      qpos=int(model.jnt_qposadr[model.body("cube").jntadr[0]]),
      rest=desk.TOP + CUBE_HALF,
    )
    self.pads: Int64Array = np.concatenate(
      [gripper.pads(model, hand) for hand in hands]
    )
    """Every gripping pad in the room, which is what a grasp touches with."""
    self.grasp = gripper.grasp(model, hands[0])
    """Where a hand's pads meet when shut, in its base tag's frame."""

  def place_cube(self, data: mujoco.MjData, cube: Float64Array) -> None:
    """Rest the cube upright on the desk at ``(x, y, yaw)``."""
    at = self.cube.qpos
    data.qpos[at : at + 3] = cube[0], cube[1], self.cube.rest + CLEARANCE
    data.qpos[at + 3 : at + 7] = np.cos(cube[2] / 2.0), 0.0, 0.0, np.sin(cube[2] / 2.0)
