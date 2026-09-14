"""A gantry: the gripper on six servoed joints, an embodiment that is a joint chain.

Three slides along the room's axes carry yaw, pitch and roll hinges, with the
base tag at the wrist's center, so every pose has a closed-form joint
configuration. The chain has no geometry, so a lens sees what it sees on the
free rig.
"""

import mujoco
import numpy as np
from mujoco.batch import Batch

from specialist_sim.arrays import Float64Array
from specialist_sim.gripper import HANDS

JOINTS = ("slide_x", "slide_y", "slide_z", "yaw", "pitch", "roll")
_AXIS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1), (0, 1, 0), (1, 0, 0))
# Each joint's servo stiffness and damping.
_GAINS = ((2000.0, 120.0),) * 3 + ((40.0, 0.5),) * 3


class Gantry:
  @classmethod
  def attach(cls, spec: mujoco.MjSpec, hand: str) -> mujoco.MjsBody:
    parent = spec.worldbody
    for name, axis, (kp, kv) in zip(JOINTS, _AXIS, _GAINS, strict=True):
      body = parent.add_body(name=f"{hand}_gripper_carriage_{name}", gravcomp=1.0)
      body.mass = 0.5
      body.inertia = [1e-3] * 3
      body.explicitinertial = True
      slides = name.startswith("slide")
      body.add_joint(
        name=f"{hand}_gripper_{name}",
        type=mujoco.mjtJoint.mjJNT_SLIDE if slides else mujoco.mjtJoint.mjJNT_HINGE,
        axis=axis,
      )
      spec.add_actuator(
        name=f"{hand}_gripper_{name}",
        target=f"{hand}_gripper_{name}",
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        gainprm=[kp] + [0.0] * 9,
        biastype=mujoco.mjtBias.mjBIAS_AFFINE,
        biasprm=[0.0, -kp, -kv] + [0.0] * 7,
      )
      parent = body
    return parent

  def __init__(self, model: mujoco.MjModel) -> None:
    self._joints = np.array(
      [
        [model.joint(f"{hand}_gripper_{name}").qposadr[0] for name in JOINTS]
        for hand in HANDS
      ]
    )
    self._motors = np.array(
      [
        [model.actuator(f"{hand}_gripper_{name}").id for name in JOINTS]
        for hand in HANDS
      ]
    )
    weight = -float(model.opt.gravity[2])
    bases = [model.body(f"{hand}_gripper_base").id for hand in HANDS]
    # How far above its setpoint the standing slide is sent to hold the
    # gripper's weight.
    self._lift = weight * model.body_subtreemass[bases] / _GAINS[2][0]

  def place(self, qpos: Float64Array, index: int, pose: Float64Array) -> None:
    qpos[self._joints[index, :3]] = pose[:3]
    qpos[self._joints[index, 3:]] = euler_zyx(pose[3:7])

  def command(
    self, setpoint: Float64Array, twist: Float64Array, seconds: float, sim: Batch
  ) -> None:
    ctrl = sim.ctrl
    ctrl[:, self._motors[:, :3]] = setpoint[..., :3]
    ctrl[:, self._motors[:, 2]] += self._lift
    ctrl[:, self._motors[:, 3:]] = euler_zyx(setpoint[..., 3:])


def euler_zyx(quaternion: Float64Array) -> Float64Array:
  """Yaw, pitch and roll such that ``Rz(yaw) Ry(pitch) Rx(roll)`` is the
  rotation each ``wxyz`` quaternion names, over any leading axes."""
  w, x, y, z = (quaternion[..., i] for i in range(4))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  return np.stack([yaw, pitch, roll], axis=-1)
