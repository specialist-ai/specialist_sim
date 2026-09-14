"""The embodiment that carries the hands: what each stands on in the scene, how
one is put at a pose, and how every hand is driven toward its setpoint."""

from typing import Protocol

import mujoco
import numpy as np
from mujoco.batch import Batch

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array
from specialist_sim.gripper import HANDS


class Embodiment(Protocol):
  @classmethod
  def attach(cls, spec: mujoco.MjSpec, hand: str) -> mujoco.MjsBody:
    """Stand one hand's carrier in the scene and return the body its gripper
    is attached to, every body it adds prefixed ``{hand}_gripper_``."""

  def __init__(self, model: mujoco.MjModel) -> None:
    """Bind the carriers in a compiled model and set their model fields."""

  def place(self, qpos: Float64Array, index: int, pose: Float64Array) -> None:
    """Write one env's ``qpos``, ``(nq,)``, so the hand at ``index`` of
    ``HANDS`` has its base tag at ``pose``, ``(7,)``."""

  def command(
    self, setpoint: Float64Array, twist: Float64Array, seconds: float, sim: Batch
  ) -> None:
    """Drive every env's hands along paths that reach their setpoints,
    ``(num_envs, hands, 7)``, ``seconds`` from now, moving at their twists,
    ``(num_envs, hands, 6)`` linear then angular in the room frame, by writing
    ``sim``'s controls and state."""


# The stiffness of each axis of the setpoint's frame, linear then angular,
# critically damped.
STIFFNESS = (16000.0,) * 3 + (160.0,) * 3
# The most the impedance pushes and twists with along each axis.
CAP = (50.0,) * 3 + (10.0,) * 3
AXES = len(STIFFNESS)
# The fastest a rig closes a gap to its setpoint, linear then angular.
CLOSING = (1.5, 10.0)
# The fastest setpoint motion a rig feeds forward, linear then angular.
FEED_LIMIT = (3.0, 20.0)


class Floating:
  """Grippers hanging on free joints, held up against gravity, each pulled
  toward its setpoint by an impedance per axis of the setpoint's frame, a mocap
  body. An axis's actuator integrates the setpoint's twist, so the physics
  moves the setpoint every step and feeds its velocity forward; a hand far from
  its path aims only as far ahead as closes the gap at CLOSING."""

  @classmethod
  def attach(cls, spec: mujoco.MjSpec, hand: str) -> mujoco.MjsBody:
    prefix = f"{hand}_gripper_"
    setpoint = spec.worldbody.add_body(name=f"{prefix}setpoint", mocap=True)
    setpoint.add_site(name=f"{prefix}setpoint")
    # A body compiled with gravcomp makes the model compute gravity compensation,
    # which __init__ extends to the gripper bodies.
    rig = spec.worldbody.add_body(name=f"{prefix}rig", gravcomp=1.0)
    rig.add_freejoint(name=f"{prefix}rig")
    rig.add_site(name=f"{prefix}rig")
    for axis, kp, cap in zip(range(AXES), STIFFNESS, CAP, strict=True):
      actuator = spec.add_actuator(
        name=f"{prefix}rig_{axis}",
        target=f"{prefix}rig",
        refsite=f"{prefix}setpoint",
        trntype=mujoco.mjtTrn.mjTRN_SITE,
        gear=np.eye(AXES)[axis],
        forcelimited=True,
        forcerange=[-cap, cap],
      )
      actuator.set_to_position(kp=kp, dampratio=1.0)
      actuator.dyntype = mujoco.mjtDyn.mjDYN_INTEGRATOR
      actuator.actearly = True
    return rig

  def __init__(self, model: mujoco.MjModel) -> None:
    rigs = [model.body(f"{hand}_gripper_rig") for hand in HANDS]
    at = np.array([model.jnt_qposadr[rig.jntadr[0]] for rig in rigs])
    self._pose = at[:, None] + np.arange(7)
    self._mocap = model.body_mocapid[
      [model.body(f"{hand}_gripper_setpoint").id for hand in HANDS]
    ]
    first = np.array([model.actuator(f"{hand}_gripper_rig_0").id for hand in HANDS])
    self._axes = first[:, None] + np.arange(AXES)
    self._act = model.actuator_actadr[self._axes]
    # How far ahead of the setpoint each axis aims, in seconds of its twist, so
    # the stiffness cancels the damping at that twist.
    first_axes = self._axes[0]
    self._lead = (
      -model.actuator_biasprm[first_axes, 2] / model.actuator_gainprm[first_axes, 0]
    )
    model.body_gravcomp[np.isin(model.body_rootid, [rig.id for rig in rigs])] = 1.0

  def place(self, qpos: Float64Array, index: int, pose: Float64Array) -> None:
    qpos[self._pose[index]] = pose

  def command(
    self, setpoint: Float64Array, twist: Float64Array, seconds: float, sim: Batch
  ) -> None:
    facing = rotation.conjugate(setpoint[..., 3:])
    moving = np.concatenate(
      [
        rotation.rotate(facing, _capped(twist[..., :3], FEED_LIMIT[0])),
        rotation.rotate(facing, _capped(twist[..., 3:], FEED_LIMIT[1])),
      ],
      axis=-1,
    )
    # Each hand's pose in its setpoint's frame, what the actuators measure.
    here = sim.qpos[:, self._pose]
    stands = np.concatenate(
      [
        rotation.rotate(facing, here[..., :3] - setpoint[..., :3]),
        rotation.log(rotation.multiply(facing, here[..., 3:])),
      ],
      axis=-1,
    )
    # From the hand to where its path starts, cut to what the hand closes over a
    # row at CLOSING against the damping.
    gap = -seconds * moving - stands
    reach = (self._lead[[0, 3]] + seconds / 2) * CLOSING
    aim = np.concatenate(
      [_capped(gap[..., :3], reach[0]), _capped(gap[..., 3:], reach[1])], axis=-1
    )
    sim.mocap_pos[:, self._mocap] = setpoint[..., :3]
    sim.mocap_quat[:, self._mocap] = setpoint[..., 3:]
    sim.ctrl[:, self._axes] = moving
    sim.act[:, self._act] = stands + aim + self._lead * moving


def _capped(vectors: Float64Array, limit: float) -> Float64Array:
  norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
  return vectors * np.minimum(1.0, limit / np.maximum(norm, 1e-12))
