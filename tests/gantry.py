"""A gantry: the gripper on six servoed joints, a carrier that is a joint chain.

Three slides along the room's axes carry three hinges, yaw then pitch then
roll, and the gripper base sits at the wrist's centre. Every pose in the box
has a joint configuration in closed form, so placement and tracking need no
solver. The chain carries no geometry of its own, so a lens sees exactly what
it sees on the free rig.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.control import CTRL, ControllerCfg, FingerServos
from specialist_sim.embodiment import (
  ChainHand,
  Hand,
  bind_chain,
  carried,
  check_slots,
  welded_gripper,
)
from specialist_sim.spaces import GRIPPERS

if TYPE_CHECKING:
  from specialist_sim.action import ActionCfg, Ramp
  from specialist_sim.scene import Scene

JOINTS = ("slide_x", "slide_y", "slide_z", "yaw", "pitch", "roll")
_KIND = (mujoco.mjtJoint.mjJNT_SLIDE,) * 3 + (mujoco.mjtJoint.mjJNT_HINGE,) * 3
_AXIS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1), (0, 1, 0), (1, 0, 0))


@dataclass(frozen=True)
class GantryCfg:
  slots: tuple[str, ...] = GRIPPERS
  stiffness: float = 2000.0
  damping: float = 120.0
  """Enough that a slide settles inside the box its setpoint is clipped to."""
  rotational_stiffness: float = 40.0
  rotational_damping: float = 0.5

  def build(self) -> Gantry:
    return Gantry(self)


class Gantry:
  def __init__(self, cfg: GantryCfg):
    self.cfg = cfg
    self.slots = check_slots(cfg.slots)

  def attach(self, room: mujoco.MjSpec) -> None:
    for slot in self.slots:
      room.attach(self._chain(), prefix=f"{slot}_", frame=room.worldbody.add_frame())

  def _chain(self) -> mujoco.MjSpec:
    """Six links, each on one servoed joint, the gripper welded to the last."""
    cfg = self.cfg
    gains = [(cfg.stiffness, cfg.damping)] * 3 + [
      (cfg.rotational_stiffness, cfg.rotational_damping)
    ] * 3
    spec = mujoco.MjSpec()
    parent = spec.worldbody
    for name, kind, axis, (kp, kv) in zip(JOINTS, _KIND, _AXIS, gains, strict=True):
      body = parent.add_body(name=f"carriage_{name}", gravcomp=1.0)
      body.mass, body.inertia, body.explicitinertial = 0.5, [1e-3] * 3, True
      body.add_joint(name=name, type=kind, axis=axis)
      spec.add_actuator(
        name=name,
        target=name,
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gainprm=[kp] + [0.0] * 9,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -kp, -kv] + [0.0] * 7,
      )
      parent = body
    spec.attach(welded_gripper(), prefix="", frame=parent.add_frame())
    return spec

  def bind(self, model: mujoco.MjModel) -> tuple[Hand, ...]:
    return bind_chain(model, self.slots, JOINTS)

  def place(
    self,
    data: mujoco.MjData,
    hand: Hand,
    position: Float64Array,
    quaternion: Float64Array,
  ) -> None:
    one = carried(hand, ChainHand)
    data.qpos[list(one.joints[:3])] = position
    data.qpos[list(one.joints[3:])] = euler_zyx(quaternion)

  def default_controller(self, action: ActionCfg) -> ControllerCfg:
    return JointServoCfg()


@dataclass(frozen=True)
class JointServoCfg:
  """Each joint's servo sent the closed-form configuration of the ramp's setpoint."""

  def build(self, scene: Scene, ramp: Ramp) -> JointServo:
    return JointServo(scene.hands)


class JointServo:
  drives = CTRL

  def __init__(self, hands: Sequence[Hand]):
    chains = [carried(hand, ChainHand) for hand in hands]
    self._slides = np.asarray([hand.motors[:3] for hand in chains], np.int64)
    self._hinges = np.asarray([hand.motors[3:] for hand in chains], np.int64)
    self.servos = FingerServos(hands)

  def reset(self, datas: Sequence[mujoco.MjData], worlds: Sequence[int]) -> None:
    """A servo carries no state between steps."""

  def command(
    self,
    ramp: Ramp,
    substep: int,
    qpos: Float64Array,
    qvel: Float64Array,
    ctrl: Float64Array,
    xfrc: Float64Array,
  ) -> None:
    ctrl[:, self._slides] = ramp.position[:, substep]
    ctrl[:, self._hinges] = euler_zyx(ramp.quaternion[:, substep])
    self.servos.command(ctrl, ramp.opening[:, substep])


def euler_zyx(quaternion: Float64Array) -> Float64Array:
  """Yaw, pitch and roll such that ``Rz(yaw) Ry(pitch) Rx(roll)`` is the
  rotation each ``wxyz`` quaternion names, over any leading axes."""
  w, x, y, z = (quaternion[..., i] for i in range(4))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  out: Float64Array = np.stack([yaw, pitch, roll], axis=-1)
  return out
