"""A gantry: the gripper on six servoed joints, an embodiment that is a joint chain.

Three slides along the room's axes carry three hinges, yaw then pitch then
roll, and the gripper's base tag sits at the wrist's center. Every pose has a joint
configuration in closed form, so placement and tracking need no solver. The
chain carries no geometry of its own, so a lens sees what it sees on the free
rig.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from specialist_sim import gripper
from specialist_sim.action import Interpolator
from specialist_sim.arrays import Float64Array
from specialist_sim.gripper import SLOTS, Hand

JOINTS = ("slide_x", "slide_y", "slide_z", "yaw", "pitch", "roll")
_KIND = (mujoco.mjtJoint.mjJNT_SLIDE,) * 3 + (mujoco.mjtJoint.mjJNT_HINGE,) * 3
_AXIS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1), (0, 1, 0), (1, 0, 0))


@dataclass
class Gantry:
  slots: tuple[str, ...] = SLOTS
  stiffness: float = 2000.0
  damping: float = 120.0
  rotational_stiffness: float = 40.0
  rotational_damping: float = 0.5

  def attach(self, scene: mujoco.MjSpec) -> None:
    gains = [(self.stiffness, self.damping)] * 3 + [
      (self.rotational_stiffness, self.rotational_damping)
    ] * 3
    for slot in self.slots:
      parent = scene.worldbody
      for name, kind, axis, (kp, kv) in zip(JOINTS, _KIND, _AXIS, gains, strict=True):
        body = parent.add_body(name=f"{slot}_carriage_{name}", gravcomp=1.0)
        body.mass, body.inertia, body.explicitinertial = 0.5, [1e-3] * 3, True
        body.add_joint(name=f"{slot}_{name}", type=kind, axis=axis)
        scene.add_actuator(
          name=f"{slot}_{name}",
          target=f"{slot}_{name}",
          trntype=mujoco.mjtTrn.mjTRN_JOINT,
          gainprm=[kp] + [0.0] * 9,
          biastype=mujoco.mjtBias.mjBIAS_AFFINE,
          biasprm=[0.0, -kp, -kv] + [0.0] * 7,
        )
        parent = body
      gripper.attach(scene, parent, slot)

  def bind(self, model: mujoco.MjModel) -> tuple[Hand, ...]:
    hands = tuple(gripper.bind(model, slot) for slot in self.slots)
    self._joints = np.array(
      [
        [model.joint(f"{slot}_{name}").qposadr[0] for name in JOINTS]
        for slot in self.slots
      ]
    )
    self._motors = np.array(
      [[model.actuator(f"{slot}_{name}").id for name in JOINTS] for slot in self.slots]
    )
    weight = -float(model.opt.gravity[2])
    self._lift = np.array(
      [weight * model.body_subtreemass[hand.base] / self.stiffness for hand in hands]
    )
    """How far above its setpoint the standing slide is sent to hold the
    gripper's weight."""
    self._servos = np.array([hand.servos for hand in hands])
    return hands

  def place(self, data: mujoco.MjData, hand: int, pose: Float64Array) -> None:
    data.qpos[self._joints[hand, :3]] = pose[:3]
    data.qpos[self._joints[hand, 3:]] = euler_zyx(pose[3:7])

  def reset(self, env_ids: Sequence[int], qpos: Float64Array) -> None:
    """A servo carries no state between steps."""

  def command(
    self,
    interpolator: Interpolator,
    substep: int,
    qpos: Float64Array,
    ctrl: Float64Array,
    xfrc: Float64Array,
  ) -> None:
    ctrl[:, self._motors[:, :3]] = interpolator.position[:, substep]
    ctrl[:, self._motors[:, 2]] += self._lift
    ctrl[:, self._motors[:, 3:]] = euler_zyx(interpolator.quaternion[:, substep])
    gripper.grip(ctrl, self._servos, interpolator.opening[:, substep])


def euler_zyx(quaternion: Float64Array) -> Float64Array:
  """Yaw, pitch and roll such that ``Rz(yaw) Ry(pitch) Rx(roll)`` is the
  rotation each ``wxyz`` quaternion names, over any leading axes."""
  w, x, y, z = (quaternion[..., i] for i in range(4))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  out: Float64Array = np.stack([yaw, pitch, roll], axis=-1)
  return out
