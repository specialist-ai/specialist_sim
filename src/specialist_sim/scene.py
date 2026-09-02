"""The compiled scene: a table, a cube on a tape target, and two grippers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.spaces import (
  FINGER_HIGH,
  FINGER_LOW,
  FINGER_OPEN,
  FINGER_SHUT,
  GRIPPERS,
)

__all__ = ["SceneCfg", "Scene", "Gripper", "Cube"]

ASSETS = Path(__file__).parent / "assets"


@dataclass(frozen=True)
class SceneCfg:
  physics_dt: float = 1.0 / 2000.0
  damping: float = 60.0
  rotational_damping: float = 0.5
  """What slows a free rig, which the impedance's feedforward exists to cancel:
  the two must be the same numbers, and the environment holds them to it."""


@dataclass(frozen=True)
class Gripper:
  """One hand, by the role each of its ids plays."""

  rig: int
  """The free body a wrench pulls on."""
  qpos: int
  """Where that body's position and orientation start."""
  fingers: tuple[int, int]
  """Where the left and right finger angles live."""
  servos: tuple[int, int]
  """The left and right position servos."""


@dataclass(frozen=True)
class Cube:
  """The block being moved."""

  body: int
  geom: int
  qpos: int
  side: float
  rest: float
  """The height of its centre when it sits on the table."""


def _compiled() -> mujoco.MjModel:
  """The room with a gripper attached under each hand's name.

  One gripper is described once and attached twice, so the two differ only in
  the prefix every name they own carries.
  """
  room = mujoco.MjSpec.from_file(str(ASSETS / "room.xml"))
  for slot in GRIPPERS:
    gripper = mujoco.MjSpec.from_file(str(ASSETS / "gripper.xml"))
    room.attach(gripper, prefix=f"{slot}_", frame=room.worldbody.add_frame())
  return room.compile()


class Scene:
  """The model, plus the ids and lengths everything else asks for."""

  def __init__(self, cfg: SceneCfg | None = None):
    cfg = cfg or SceneCfg()
    self.model = _compiled()
    self.model.opt.timestep = cfg.physics_dt
    self.data = mujoco.MjData(self.model)
    self.table_top = float(
      self.model.body("table").pos[2] + self.model.geom("table").size[2]
    )
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
    self.grippers = tuple(self._gripper(slot, cfg) for slot in GRIPPERS)
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

  def _gripper(self, slot: str, cfg: SceneCfg) -> Gripper:
    rig = self.model.body(f"{slot}_rig")
    damping = self.model.dof_damping[rig.dofadr[0] : rig.dofadr[0] + 6]
    damping[:3], damping[3:] = cfg.damping, cfg.rotational_damping
    joints = [self.model.joint(f"{slot}_{side}_finger") for side in ("left", "right")]
    for joint, sign in zip(joints, (1.0, -1.0), strict=True):
      wanted = sorted(sign * angle for angle in (FINGER_SHUT, FINGER_OPEN))
      low, high = joint.range
      if wanted[0] < low - 1e-9 or wanted[1] > high + 1e-9:
        raise ValueError(
          f"{joint.name} travels {low:.4f}..{high:.4f}, which does not hold "
          f"the commanded {wanted[0]:.4f}..{wanted[1]:.4f}"
        )
      bounds = sorted(sign * angle for angle in (FINGER_LOW, FINGER_HIGH))
      if not np.allclose([low, high], bounds, atol=1e-6):
        raise ValueError(
          f"{joint.name} travels {low:.6f}..{high:.6f}, and an action is "
          f"bounded to {bounds[0]:.6f}..{bounds[1]:.6f}"
        )
    return Gripper(
      rig=rig.id,
      qpos=int(self.model.jnt_qposadr[rig.jntadr[0]]),
      fingers=(int(joints[0].qposadr[0]), int(joints[1].qposadr[0])),
      servos=tuple(self.model.actuator(joint.name).id for joint in joints),
    )

  def _grasp(self) -> Float64Array:
    """The middle of the pads that meet, in a hand's base frame.

    The pads first touch at their very end; an object seats further in.
    """
    data = mujoco.MjData(self.model)
    for address, sign in zip(self.grippers[0].fingers, (1.0, -1.0), strict=True):
      data.qpos[address] = sign * FINGER_SHUT
    mujoco.mj_forward(self.model, data)
    hinge = data.xanchor[self.model.joint(f"{GRIPPERS[0]}_left_finger").id]
    tips = [
      max(
        (
          geom
          for geom in self.pads
          if self.model.geom_bodyid[geom]
          == self.model.body(f"{GRIPPERS[0]}_{side}_finger").id
        ),
        key=lambda geom: float(np.linalg.norm(data.geom_xpos[geom] - hinge)),
      )
      for side in ("left", "right")
    ]
    base = self.model.body(f"{GRIPPERS[0]}_base").id
    middle = 0.5 * (data.geom_xpos[tips[0]] + data.geom_xpos[tips[1]])
    local: Float64Array = data.xmat[base].reshape(3, 3).T @ (middle - data.xpos[base])
    return local
