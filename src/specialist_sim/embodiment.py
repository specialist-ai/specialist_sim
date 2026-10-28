"""The embodiment that carries the hands: how they stand in the scene, how
one is put at a pose, and how they are driven along the interpolated setpoints
every physics step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import mujoco
import numpy as np

from specialist_sim import gripper, rotation
from specialist_sim.arrays import Float64Array
from specialist_sim.gripper import SLOTS, Hand

if TYPE_CHECKING:
  from specialist_sim.action import Interpolator

__all__ = ["Embodiment", "Floating", "OutOfReach"]


class OutOfReach(ValueError):
  """A pose the embodiment cannot put a hand at."""


class Embodiment(Protocol):
  slots: tuple[str, ...]
  """The hands it stands up, in the order their states are laid out."""

  def attach(self, scene: mujoco.MjSpec) -> None:
    """Stand the hands in the scene, every body of a hand prefixed ``{slot}_``."""
    ...

  def bind(self, model: mujoco.MjModel) -> tuple[Hand, ...]:
    """Take the ids the compiled model gave the hands."""
    ...

  def place(self, data: mujoco.MjData, hand: int, pose: Float64Array) -> None:
    """Write ``qpos`` so the hand's base is at a position and quaternion, or
    raise ``OutOfReach``."""
    ...

  def reset(self, env_ids: Sequence[int], qpos: Float64Array) -> None:
    """Take up the named envs where their reset put them. ``qpos`` is
    every env's, ``(num_envs, nq)``."""
    ...

  def command(
    self,
    interpolator: Interpolator,
    substep: int,
    qpos: Float64Array,
    ctrl: Float64Array,
    xfrc: Float64Array,
  ) -> None:
    """Write what every env wants for one substep of the interpolation: ``ctrl``
    is ``(num_envs, nu)`` and ``xfrc`` ``(num_envs, nbody, 6)``."""
    ...


RIG_MASS = 1e-6
RIG_INERTIA = 1e-9
"""A free rig weighs nothing: the wrench acts on the gripper alone."""


@dataclass
class Floating:
  """Grippers hanging on free joints, each pulled along the interpolated
  setpoints by a stiff impedance with the segment's velocity fed forward, and
  held up against gravity."""

  slots: tuple[str, ...] = SLOTS
  stiffness: float = 4000.0
  rotational_stiffness: float = 40.0
  damping: float = 60.0
  rotational_damping: float = 0.5
  speed: float = 1.5
  turn: float = 10.0
  """The fastest the feedforward asks a rig to go, in meters and radians a second."""
  max_force: float = 25.0
  max_torque: float = 1.5
  """What the rig may push and twist with, N and N m: a few times the
  gripper's 4 N weight, so a hand stops when the fingertips meet the table
  instead of driving through it."""

  def attach(self, scene: mujoco.MjSpec) -> None:
    for slot in self.slots:
      rig = scene.worldbody.add_body(name=f"{slot}_rig")
      rig.add_freejoint(name=f"{slot}_rig")
      rig.mass, rig.inertia = RIG_MASS, [RIG_INERTIA] * 3
      gripper.attach(scene, rig, slot)

  def bind(self, model: mujoco.MjModel) -> tuple[Hand, ...]:
    hands = tuple(gripper.bind(model, slot) for slot in self.slots)
    rigs = [model.body(f"{slot}_rig") for slot in self.slots]
    self._damping = np.array([[self.damping] * 3 + [self.rotational_damping] * 3])
    for rig in rigs:
      model.dof_damping[rig.dofadr[0] : rig.dofadr[0] + 6] = self._damping[0]
    self._rigs = np.array([rig.id for rig in rigs])
    at = np.array([model.jnt_qposadr[rig.jntadr[0]] for rig in rigs])
    self._position = at[:, None] + np.arange(3)
    self._quaternion = at[:, None] + np.arange(3, 7)
    self._carried = np.flatnonzero(np.isin(model.body_rootid, self._rigs))
    self._weight = -np.outer(model.body_mass[self._carried], model.opt.gravity)
    self._servos = np.array([hand.servos for hand in hands])
    return hands

  def place(self, data: mujoco.MjData, hand: int, pose: Float64Array) -> None:
    at = self._position[hand, 0]
    data.qpos[at : at + 7] = pose

  def reset(self, env_ids: Sequence[int], qpos: Float64Array) -> None:
    """A wrench carries no state between steps."""

  def command(
    self,
    interpolator: Interpolator,
    substep: int,
    qpos: Float64Array,
    ctrl: Float64Array,
    xfrc: Float64Array,
  ) -> None:
    here, facing = qpos[:, self._position], qpos[:, self._quaternion]
    twist = np.concatenate(
      [
        _capped(interpolator.twist[:, substep, :, :3], self.speed),
        _capped(interpolator.twist[:, substep, :, 3:], self.turn),
      ],
      axis=-1,
    )
    apart = rotation.log(
      rotation.multiply(interpolator.quaternion[:, substep], rotation.conjugate(facing))
    )
    pull = (
      self.stiffness * (interpolator.position[:, substep] - here)
      + self._damping[:, :3] * twist[..., :3]
    )
    turn = self.rotational_stiffness * apart + self._damping[:, 3:] * twist[..., 3:]
    xfrc[:, self._carried, :3] = self._weight
    xfrc[:, self._rigs, :3] += _capped(pull, self.max_force)
    xfrc[:, self._rigs, 3:] = _capped(turn, self.max_torque)
    gripper.grip(ctrl, self._servos, interpolator.opening[:, substep])


def _capped(vectors: Float64Array, limit: float) -> Float64Array:
  """Each vector shortened to ``limit`` where it is longer."""
  norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
  out: Float64Array = vectors * np.minimum(1.0, limit / np.maximum(norm, 1e-12))
  return out
