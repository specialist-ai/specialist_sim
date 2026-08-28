"""The compiled scene: a table, a cube on a tape target, and the hands an
embodiment stands in the room."""

from __future__ import annotations

import re
from dataclasses import dataclass

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.embodiment import ASSETS, Embodiment, Hand
from specialist_sim.spaces import FINGER_SHUT, PER_HAND, gripper_from_opening

__all__ = ["SceneCfg", "Scene", "Cube"]


@dataclass(frozen=True)
class SceneCfg:
  physics_dt: float = 1.0 / 2000.0


@dataclass(frozen=True)
class Cube:
  """The block being moved."""

  body: int
  geom: int
  qpos: int
  side: float
  rest: float
  """The height of its centre when it sits on the table."""


def _compiled(embodiment: Embodiment) -> mujoco.MjModel:
  """The room with the embodiment's hands standing in it."""
  room = mujoco.MjSpec.from_file(str(ASSETS / "room.xml"))
  embodiment.attach(room)
  return room.compile()


class Scene:
  """The model, plus the ids and lengths everything else asks for."""

  def __init__(self, cfg: SceneCfg, embodiment: Embodiment):
    self.embodiment = embodiment
    self.slots = embodiment.slots
    """The hands the room holds, in ``GRIPPERS``' order."""
    self.model = _compiled(embodiment)
    self.model.opt.timestep = cfg.physics_dt
    self.data = mujoco.MjData(self.model)
    self._scratch = mujoco.MjData(self.model)
    self.table_top = float(
      self.model.body("table").pos[2] + self.model.geom("table").size[2]
    )
    self.table = tuple(float(value) for value in self.model.body("table").pos[:2])
    self.tape = tuple(float(value) for value in self.model.body("tape").pos[:2])
    self.tape_half = tuple(float(value) for value in self.model.geom("tape").size[:2])
    side = 2.0 * float(self.model.geom("cube").size[0])
    self.cube = Cube(
      body=self.model.body("cube").id,
      geom=self.model.geom("cube").id,
      qpos=int(self.model.jnt_qposadr[self.model.body("cube").jntadr[0]]),
      side=side,
      rest=self.table_top + side / 2.0,
    )
    self.hands: tuple[Hand, ...] = embodiment.bind(self.model)
    """Each hand's ids, in slot order."""
    self.pads = np.array(
      sorted(
        geom
        for geom in range(self.model.ngeom)
        if re.search(
          r"_pad\d+$",
          mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "",
        )
      ),
      np.int64,
    )
    """Every finger pad's geom, which is what a grasp touches with."""
    self.grasp = self._grasp()
    """Where a hand's pads meet when shut, in its base frame."""
    self.pinch = self._pinch()
    """The opening at which the pads just touch a seated cube."""

  def observe(self, qpos: Float64Array) -> Float64Array:
    """Every world's hands read out of ``qpos``, ``(worlds, nq)``, by forward
    kinematics, laid out like a state: each base's position and quaternion,
    then its left finger's angle, ``(worlds, hands, PER_HAND)``."""
    out = np.empty((len(qpos), len(self.hands), PER_HAND))
    for world, state in enumerate(qpos):
      self._scratch.qpos[:] = state
      mujoco.mj_kinematics(self.model, self._scratch)
      for index, hand in enumerate(self.hands):
        out[world, index, :3] = self._scratch.xpos[hand.base]
        out[world, index, 3:7] = self._scratch.xquat[hand.base]
        out[world, index, 7] = state[hand.fingers[0]]
    return out

  def twists(self, qpos: Float64Array, qvel: Float64Array) -> Float64Array:
    """Every world's hands read out of ``qpos`` and ``qvel``, by forward
    kinematics: each base's linear then angular velocity in the room's frame,
    ``(worlds, hands, 6)``."""
    out = np.empty((len(qpos), len(self.hands), 6))
    twist = np.empty(6)
    for world, (state, speed) in enumerate(zip(qpos, qvel, strict=True)):
      self._scratch.qpos[:] = state
      self._scratch.qvel[:] = speed
      mujoco.mj_kinematics(self.model, self._scratch)
      mujoco.mj_comPos(self.model, self._scratch)
      mujoco.mj_comVel(self.model, self._scratch)
      for index, hand in enumerate(self.hands):
        mujoco.mj_objectVelocity(
          self.model, self._scratch, mujoco.mjtObj.mjOBJ_XBODY, hand.base, twist, 0
        )
        out[world, index, :3] = twist[3:]
        out[world, index, 3:] = twist[:3]
    return out

  def _pinch(self) -> float:
    """The opening at which the pads just touch a cube seated between them.

    The finger servo holds a commanded angle, so closing past this point presses
    on the cube with whatever force the error asks for and stores it in the
    contact; the fingers then fire the cube out as they open. A grasp closes to
    just inside this instead.
    """
    hand, data = self.hands[0], mujoco.MjData(self.model)
    mujoco.mj_forward(self.model, data)
    seat = data.xpos[hand.base] + data.xmat[hand.base].reshape(3, 3) @ self.grasp
    data.qpos[self.cube.qpos : self.cube.qpos + 3] = seat
    bodies = {
      self.model.body(f"{hand.slot}_{side}_finger").id for side in ("left", "right")
    }
    pads = [geom for geom in self.pads if self.model.geom_bodyid[geom] in bodies]

    def gap(opening: float) -> float:
      angle = gripper_from_opening(opening)
      data.qpos[list(hand.fingers)] = angle, -angle
      mujoco.mj_forward(self.model, data)
      return min(
        float(mujoco.mj_geomDistance(self.model, data, pad, self.cube.geom, 1.0, None))
        for pad in pads
      )

    if gap(0.0) >= 0.0:
      raise ValueError("the pads never reach the cube, so nothing can be grasped")
    low, high = 0.0, 1.0
    for _ in range(40):
      middle = 0.5 * (low + high)
      low, high = (low, middle) if gap(middle) > 0.0 else (middle, high)
    return high

  def _grasp(self) -> Float64Array:
    """The middle of the pads that meet, in a hand's base frame.

    The pads first touch at their very end; an object seats further in.
    """
    hand, data = self.hands[0], self._scratch
    data.qpos[list(hand.fingers)] = FINGER_SHUT, -FINGER_SHUT
    mujoco.mj_forward(self.model, data)
    hinge = data.xanchor[self.model.joint(f"{hand.slot}_left_finger").id]
    tips = [
      max(
        (
          geom
          for geom in self.pads
          if self.model.geom_bodyid[geom]
          == self.model.body(f"{hand.slot}_{side}_finger").id
        ),
        key=lambda geom: float(np.linalg.norm(data.geom_xpos[geom] - hinge)),
      )
      for side in ("left", "right")
    ]
    middle = 0.5 * (data.geom_xpos[tips[0]] + data.geom_xpos[tips[1]])
    local: Float64Array = data.xmat[hand.base].reshape(3, 3).T @ (
      middle - data.xpos[hand.base]
    )
    return local
