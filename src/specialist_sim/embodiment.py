"""What holds the hands up: how it stands in the room, and how a hand is put
at a pose when an episode opens.

The gripper is one description, attached under each slot's name whatever holds
it, so a hand's state is always its gripper base's pose and its finger angle,
and a policy is not told what carries the base. The slots an embodiment fills
are some of ``GRIPPERS``, in that order, and the width of a state follows from
how many there are.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import mujoco
import numpy as np
import specialist_description

from specialist_sim.arrays import Float64Array
from specialist_sim.spaces import (
  FINGER_HIGH,
  FINGER_LOW,
  FINGER_OPEN,
  FINGER_SHUT,
  GRIPPERS,
)

if TYPE_CHECKING:
  from specialist_sim.action import ActionCfg
  from specialist_sim.control import ControllerCfg

__all__ = [
  "ASSETS",
  "ChainHand",
  "Embodiment",
  "EmbodimentCfg",
  "Floating",
  "FloatingCfg",
  "FreeHand",
  "Hand",
  "OutOfReach",
  "bind_chain",
  "bind_fingers",
  "carried",
  "check_slots",
  "gripper_spec",
  "welded_gripper",
]

ASSETS = Path(__file__).parent / "assets"

VARIANT = "specialist_actuated"

GRIPPING = 4
"""The contact dimensionality the description marks a gripping surface with."""

FINGER_STIFFNESS = 650.0
FINGER_DAMPING = 1.97
"""What the finger servo holds a commanded angle with."""

PAD_SOLREF = 0.010
"""How long a pad takes to answer a contact, until the real stiffness is
measured on the hardware."""

RIG_MASS = 1e-06
RIG_INERTIA = 1e-09


class OutOfReach(ValueError):
  """A pose the hand's carrier cannot put it at."""


@dataclass(frozen=True)
class Hand:
  """One slot's ids, by the role each plays."""

  slot: str
  base: int
  """The body ``{slot}_base``: what a pose names, and what a lens rides."""
  fingers: tuple[int, int]
  """Where the left and right finger angles live in ``qpos``."""
  servos: tuple[int, int]
  """The left and right finger position servos."""


@dataclass(frozen=True)
class FreeHand(Hand):
  """A hand hanging on a free joint."""

  rig: int
  """The free body a wrench pulls on."""
  qpos: int
  """Where that joint's position and orientation start in ``qpos``."""


@dataclass(frozen=True)
class ChainHand(Hand):
  """A hand a joint chain carries."""

  joints: tuple[int, ...]
  """Where each of the chain's joints lives in ``qpos``, base to wrist."""
  dofs: tuple[int, ...]
  """Where each joint's velocity lives in ``qvel``, in the same order."""
  motors: tuple[int, ...]
  """Each joint's actuator, in the same order."""


class Embodiment(Protocol):
  """What carries the hands."""

  @property
  def slots(self) -> tuple[str, ...]:
    """The hands it stands up, as some of ``GRIPPERS`` in that order."""
    ...

  def attach(self, room: mujoco.MjSpec) -> None:
    """Stand the hands in the room, every body of a hand prefixed ``{slot}_``."""
    ...

  def bind(self, model: mujoco.MjModel) -> tuple[Hand, ...]:
    """Each slot's ids in the compiled model, in slot order."""
    ...

  def place(
    self,
    data: mujoco.MjData,
    hand: Hand,
    position: Float64Array,
    quaternion: Float64Array,
  ) -> None:
    """Write ``qpos`` so the hand's base is at a pose, or raise ``OutOfReach``."""
    ...

  def default_controller(self, action: ActionCfg) -> ControllerCfg:
    """What drives these hands when the environment is not told otherwise."""
    ...


class EmbodimentCfg(Protocol):
  def build(self) -> Embodiment: ...


@dataclass(frozen=True)
class FloatingCfg:
  """Free-flying grippers, one per slot, each pulled about by a wrench."""

  slots: tuple[str, ...] = GRIPPERS
  damping: float = 60.0
  rotational_damping: float = 0.5
  """What slows a free rig."""

  def build(self) -> Floating:
    return Floating(self)


class Floating:
  """Grippers hanging on free joints, placed straight through ``qpos``."""

  def __init__(self, cfg: FloatingCfg):
    self.cfg = cfg
    self.slots = check_slots(cfg.slots)

  def attach(self, room: mujoco.MjSpec) -> None:
    for slot in self.slots:
      room.attach(gripper_spec(), prefix=f"{slot}_", frame=room.worldbody.add_frame())

  def bind(self, model: mujoco.MjModel) -> tuple[Hand, ...]:
    hands = []
    for slot in self.slots:
      rig = model.body(f"{slot}_rig")
      damping = model.dof_damping[rig.dofadr[0] : rig.dofadr[0] + 6]
      damping[:3], damping[3:] = self.cfg.damping, self.cfg.rotational_damping
      hands.append(
        FreeHand(
          **vars(bind_fingers(model, slot)),
          rig=rig.id,
          qpos=int(model.jnt_qposadr[rig.jntadr[0]]),
        )
      )
    return tuple(hands)

  def place(
    self,
    data: mujoco.MjData,
    hand: Hand,
    position: Float64Array,
    quaternion: Float64Array,
  ) -> None:
    at = carried(hand, FreeHand).qpos
    data.qpos[at : at + 3] = position
    data.qpos[at + 3 : at + 7] = quaternion

  def default_controller(self, action: ActionCfg) -> ControllerCfg:
    return action


def carried[H: Hand](hand: Hand, by: type[H]) -> H:
  """``hand`` as the kind its carrier made it."""
  if not isinstance(hand, by):
    raise TypeError(f"{hand.slot} is not a {by.__name__}")
  return hand


def gripper_spec() -> mujoco.MjSpec:
  """The described gripper, hung on a free joint and given this rig's physics."""
  described = mujoco.MjSpec.from_file(str(specialist_description.mjcf_path(VARIANT)))
  described.body("base").add_body(name="lens")
  for side in ("left", "right"):
    servo = described.actuator(f"{side}_finger")
    servo.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    servo.gainprm[0] = FINGER_STIFFNESS
    servo.biasprm[1] = -FINGER_STIFFNESS
    servo.biasprm[2] = -FINGER_DAMPING
  for geom in described.geoms:
    if geom.condim == GRIPPING:
      geom.solref[0] = PAD_SOLREF
  spec = mujoco.MjSpec()
  rig = spec.worldbody.add_body(name="rig")
  rig.add_freejoint(name="rig")
  rig.mass = RIG_MASS
  rig.inertia = [RIG_INERTIA] * 3
  spec.attach(described, prefix="", frame=rig.add_frame())
  return spec


def welded_gripper() -> mujoco.MjSpec:
  """The gripper without its free joint, to weld to a wrist."""
  spec = gripper_spec()
  spec.delete(spec.joint("rig"))
  return spec


def check_slots(slots: Sequence[str]) -> tuple[str, ...]:
  """``slots`` as a tuple, which must be some of ``GRIPPERS`` in that order."""
  chosen = tuple(slots)
  order = tuple(slot for slot in GRIPPERS if slot in chosen)
  if not chosen or order != chosen:
    raise ValueError(f"the hands are some of {GRIPPERS}, in that order, not {chosen}")
  return chosen


def bind_fingers(model: mujoco.MjModel, slot: str) -> Hand:
  """A slot's base and fingers, once the fingers' travel is known to be the
  travel an opening names."""
  joints = [model.joint(f"{slot}_{side}_finger") for side in ("left", "right")]
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
        f"{joint.name} travels {low:.6f}..{high:.6f}, and an opening names "
        f"{bounds[0]:.6f}..{bounds[1]:.6f}"
      )
  left, right = joints
  return Hand(
    slot=slot,
    base=model.body(f"{slot}_base").id,
    fingers=(int(left.qposadr[0]), int(right.qposadr[0])),
    servos=(int(model.actuator(left.name).id), int(model.actuator(right.name).id)),
  )


def bind_chain(
  model: mujoco.MjModel, slots: Sequence[str], joints: Sequence[str]
) -> tuple[Hand, ...]:
  """Each slot's hand on a chain of ``joints``, every joint named
  ``{slot}_{joint}`` and driven by a servo of the same name."""
  hands = []
  for slot in slots:
    chain = [model.joint(f"{slot}_{name}") for name in joints]
    hands.append(
      ChainHand(
        **vars(bind_fingers(model, slot)),
        joints=tuple(int(joint.qposadr[0]) for joint in chain),
        dofs=tuple(int(joint.dofadr[0]) for joint in chain),
        motors=tuple(int(model.actuator(joint.name).id) for joint in chain),
      )
    )
  return tuple(hands)
