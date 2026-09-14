"""The described gripper: a base, two servoed fingers, and the pads they grip
with. A hand's state is its base's pose and its left finger's angle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
import specialist_description

from specialist_sim.arrays import Float64Array, Int64Array

__all__ = [
  "GRIPPING",
  "OPEN",
  "SHUT",
  "SLOTS",
  "VARIANT",
  "Hand",
  "bind",
  "grasp",
  "grip",
  "pads",
  "spec",
  "states",
  "twists",
]

VARIANT = "specialist_actuated"
SLOTS = ("left_gripper", "right_gripper")
"""The hands, named for the operator's side each is on. Every body a hand owns
carries its slot as a prefix."""

SHUT = np.radians(-9.1175)
"""The left finger's angle when the pads meet."""
OPEN = np.radians(40.0)
"""The left finger's angle at its open limit."""

GRIPPING = 4
"""The contact dimensionality the description marks a gripping pad with."""

FINGER_HZ = 50.0
FINGER_RATIO = 1.0
"""How fast the finger servo answers a commanded angle, and how it damps."""
PAD_SOLREF = 0.010
"""How long a pad takes to answer a contact."""


@dataclass(frozen=True)
class Hand:
  slot: str
  base: int
  """The body a pose names."""
  fingers: tuple[int, int]
  """Where the left and right finger angles sit in ``qpos``."""
  servos: tuple[int, int]
  """The left and right finger position servos."""


def spec() -> mujoco.MjSpec:
  """The gripper as described, with this rig's finger servo and pad compliance."""
  described = mujoco.MjSpec.from_file(str(specialist_description.mjcf_path(VARIANT)))
  turn = 2.0 * np.pi * FINGER_HZ
  for side in ("left", "right"):
    servo = described.actuator(f"{side}_finger")
    swung = described.joint(f"{side}_finger").armature
    servo.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    servo.gainprm[0] = swung * turn * turn
    servo.biasprm[1] = -swung * turn * turn
    servo.biasprm[2] = -2.0 * FINGER_RATIO * swung * turn
  for geom in described.geoms:
    if geom.condim == GRIPPING:
      geom.solref[0] = PAD_SOLREF
  return described


def bind(model: mujoco.MjModel, slot: str) -> Hand:
  """One slot's ids in a compiled model."""
  left, right = (model.joint(f"{slot}_{side}_finger") for side in ("left", "right"))
  return Hand(
    slot=slot,
    base=model.body(f"{slot}_base").id,
    fingers=(int(left.qposadr[0]), int(right.qposadr[0])),
    servos=(int(model.actuator(left.name).id), int(model.actuator(right.name).id)),
  )


def grip(ctrl: Float64Array, servos: Int64Array, opening: Float64Array) -> None:
  """Send every hand's finger servos, ``(hands, 2)``, an opening, ``(worlds,
  hands)``: the left finger to it and the right mirrored."""
  ctrl[:, servos] = opening[..., None] * np.array([1.0, -1.0])


def pads(model: mujoco.MjModel, hand: Hand) -> Int64Array:
  """The hand's gripping pads: every geom of its fingers the description marks
  as one."""
  fingers = [model.body(f"{hand.slot}_{side}_finger").id for side in ("left", "right")]
  out: Int64Array = np.flatnonzero(
    (model.geom_condim == GRIPPING) & np.isin(model.geom_bodyid, fingers)
  )
  return out


def grasp(model: mujoco.MjModel, hand: Hand) -> Float64Array:
  """Where the pads meet when shut, in the base frame: the middle of the two
  pads farthest from the finger hinge."""
  data = mujoco.MjData(model)
  data.qpos[list(hand.fingers)] = SHUT, -SHUT
  mujoco.mj_forward(model, data)
  hinge = data.xanchor[model.body(f"{hand.slot}_left_finger").jntadr[0]]
  all_pads = pads(model, hand)
  tips = []
  for side in ("left", "right"):
    body = model.body(f"{hand.slot}_{side}_finger").id
    own = all_pads[model.geom_bodyid[all_pads] == body]
    reach = np.linalg.norm(data.geom_xpos[own] - hinge, axis=1)
    tips.append(data.geom_xpos[own[np.argmax(reach)]])
  middle = 0.5 * (tips[0] + tips[1])
  out: Float64Array = data.xmat[hand.base].reshape(3, 3).T @ (
    middle - data.xpos[hand.base]
  )
  return out


def states(
  model: mujoco.MjModel,
  scratch: mujoco.MjData,
  hands: Sequence[Hand],
  qpos: Float64Array,
) -> Float64Array:
  """Every hand read out of ``qpos``, ``(n, nq)``: its base's position and
  quaternion, then its left finger's angle, ``(n, hands, 8)``."""
  out = np.empty((len(qpos), len(hands), 8))
  for row, state in enumerate(qpos):
    scratch.qpos[:] = state
    mujoco.mj_kinematics(model, scratch)
    for index, hand in enumerate(hands):
      out[row, index, :3] = scratch.xpos[hand.base]
      out[row, index, 3:7] = scratch.xquat[hand.base]
      out[row, index, 7] = state[hand.fingers[0]]
  return out


def twists(
  model: mujoco.MjModel,
  scratch: mujoco.MjData,
  hands: Sequence[Hand],
  qpos: Float64Array,
  qvel: Float64Array,
) -> Float64Array:
  """Every hand's base velocity in the room frame, linear then angular,
  ``(n, hands, 6)``."""
  out = np.empty((len(qpos), len(hands), 6))
  twist = np.empty(6)
  for row, (state, speed) in enumerate(zip(qpos, qvel, strict=True)):
    scratch.qpos[:] = state
    scratch.qvel[:] = speed
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    mujoco.mj_comVel(model, scratch)
    for index, hand in enumerate(hands):
      mujoco.mj_objectVelocity(
        model, scratch, mujoco.mjtObj.mjOBJ_XBODY, hand.base, twist, 0
      )
      out[row, index, :3] = twist[3:]
      out[row, index, 3:] = twist[:3]
  return out
