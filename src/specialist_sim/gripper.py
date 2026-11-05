"""The described gripper: a base, two fingers, and the pads they grip with.
One servo turns the right finger, and the left follows it through a 1:1 gear.
A hand's state is the pose of its base tag and its opening: how far each finger
has turned out from parallel, in radians, which is minus the right finger's angle.

The base tag is the frame a capture tracks a hand by, the dataset's hand frame.
The description puts it at the base as a placeholder. It sits ``TAG_OFFSET``
behind the base along the base's x, with the base's rotation.
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
  "TAG_OFFSET",
  "VARIANT",
  "Hand",
  "attach",
  "bind",
  "frame",
  "grasp",
  "grip",
  "pads",
  "spec",
  "states",
  "twists",
]

VARIANT = "specialist_actuated"
SLOTS = ("left_gripper", "right_gripper")
"""The hands, named for the near side each is on. Every body a hand owns
carries its slot as a prefix."""

SHUT = np.radians(-9.1175)
"""The opening when the pads meet."""
OPEN = np.radians(40.0)
"""The opening at the fingers' open limit."""

GRIPPING = 4
"""The contact dimensionality the description marks a gripping pad with."""

FINGER_HZ = 50.0
FINGER_RATIO = 1.0
"""How fast the finger servo answers a commanded angle, and how it damps."""
DRIVEN = "right_finger"
"""The finger the servo turns. The described gear makes the other follow it."""
TAG_OFFSET = 0.025
"""How far behind the described base the base tag sits, along the base's x,
meters: where the fiducial is mounted on the gripper."""


@dataclass(frozen=True)
class Hand:
  slot: str
  base: int
  """The body a hand is carried by. Its state is the tag ``TAG_OFFSET`` behind it."""
  fingers: tuple[int, int]
  """Where the left and right finger angles sit in ``qpos``."""
  servo: int
  """The right finger's position servo, the hand's one drive."""


def spec(side: int) -> mujoco.MjSpec:
  """The gripper as described, with this rig's finger servo. The description
  lets two grippers pass through each other, so the hand on each ``side``
  (its index in ``SLOTS``) gets a contact bit of its own that meets the other's."""
  described = mujoco.MjSpec.from_file(
    str(specialist_description.get_mjcf_path(VARIANT))
  )
  turn = 2.0 * np.pi * FINGER_HZ
  servo = described.actuator(DRIVEN)
  swung = described.joint(DRIVEN).armature
  servo.biastype = mujoco.mjtBias.mjBIAS_AFFINE
  servo.gainprm[0] = swung * turn * turn
  servo.biasprm[1] = -swung * turn * turn
  servo.biasprm[2] = -2.0 * FINGER_RATIO * swung * turn
  for geom in described.geoms:
    if geom.contype == 2:
      geom.contype = 2 << side
      geom.conaffinity = 1 | (2 << (1 - side))
  return described


def attach(scene: mujoco.MjSpec, body: mujoco.MjsBody, slot: str) -> None:
  """Seat a slot's gripper on the embodiment's ``body`` that carries it, whose
  frame is the hand's base tag: the base ``TAG_OFFSET`` ahead along its x."""
  scene.attach(
    spec(SLOTS.index(slot)),
    prefix=f"{slot}_",
    frame=body.add_frame(pos=[TAG_OFFSET, 0.0, 0.0]),
  )


def bind(model: mujoco.MjModel, slot: str) -> Hand:
  """One slot's ids in a compiled model."""
  left, right = (model.joint(f"{slot}_{side}_finger") for side in ("left", "right"))
  return Hand(
    slot=slot,
    base=model.body(f"{slot}_base").id,
    fingers=(int(left.qposadr[0]), int(right.qposadr[0])),
    servo=int(model.actuator(f"{slot}_{DRIVEN}").id),
  )


def grip(ctrl: Float64Array, servos: Int64Array, opening: Float64Array) -> None:
  """Send every hand's servo, ``(hands,)``, an opening, ``(num_envs, hands)``:
  the right finger opens toward negative angles."""
  ctrl[:, servos] = -opening


def pads(model: mujoco.MjModel, hand: Hand) -> Int64Array:
  """The hand's gripping pads: every geom of its fingers the description marks
  as one."""
  fingers = [model.body(f"{hand.slot}_{side}_finger").id for side in ("left", "right")]
  out: Int64Array = np.flatnonzero(
    (model.geom_condim == GRIPPING) & np.isin(model.geom_bodyid, fingers)
  )
  return out


def frame(data: mujoco.MjData, hand: Hand) -> tuple[Float64Array, Float64Array]:
  """Where a hand's base tag stands in the room: its position and its
  ``(3, 3)`` rotation, read off ``data``'s kinematics."""
  rotation = data.xmat[hand.base].reshape(3, 3)
  position: Float64Array = data.xpos[hand.base] - TAG_OFFSET * rotation[:, 0]
  return position, rotation


def grasp(model: mujoco.MjModel, hand: Hand) -> Float64Array:
  """Where the pads meet when shut, in the base tag's frame: the middle of
  the two pads farthest from the finger hinge."""
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
  position, rotation = frame(data, hand)
  out: Float64Array = rotation.T @ (middle - position)
  return out


def states(
  model: mujoco.MjModel,
  scratch: mujoco.MjData,
  hands: Sequence[Hand],
  qpos: Float64Array,
) -> Float64Array:
  """Every hand read out of ``qpos``, ``(n, nq)``: its base tag's position and
  quaternion, then its opening, ``(n, hands, 8)``: what the servo reads."""
  out = np.empty((len(qpos), len(hands), 8))
  for row, state in enumerate(qpos):
    scratch.qpos[:] = state
    mujoco.mj_kinematics(model, scratch)
    for index, hand in enumerate(hands):
      out[row, index, :3] = frame(scratch, hand)[0]
      out[row, index, 3:7] = scratch.xquat[hand.base]
      out[row, index, 7] = -state[hand.fingers[1]]
  return out


def twists(
  model: mujoco.MjModel,
  scratch: mujoco.MjData,
  hands: Sequence[Hand],
  qpos: Float64Array,
  qvel: Float64Array,
) -> Float64Array:
  """Every hand's base tag velocity in the room frame, linear then angular,
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
      behind = frame(scratch, hand)[0] - scratch.xpos[hand.base]
      out[row, index, :3] = twist[3:] + np.cross(twist[:3], behind)
      out[row, index, 3:] = twist[:3]
  return out
