"""A commanded trajectory per hand, resampled to the physics rate, and the
impedance that chases it on a free rig.

An action is a trajectory of absolute poses in the room frame,
``[position(3), quaternion(4), opening(1)]`` per hand, at ``1 / ACTION_HZ``
intervals from now. An opening is the coupled finger angle in radians, signed as
the left finger turns, which is the number a dataset records. A control step
tracks the leading poses and drops the rest, so a policy commits to further
ahead than it executes.

The ramp resamples the trajectory at the physics rate, whatever carries the
hands. A tracker's natural frequency sits above the control rate and below the
trajectory's, so a setpoint that jumped once per control step would ring at it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.control import CTRL, XFRC, FingerServos
from specialist_sim.embodiment import FreeHand, Hand, carried
from specialist_sim.spaces import ACTION_HZ, HORIZON, PER_HAND

if TYPE_CHECKING:
  from specialist_sim.scene import Scene

__all__ = [
  "ActionCfg",
  "Impedance",
  "Ramp",
  "hold",
  "slerp",
  "turn",
]

REST = np.array([1.0, 0.0, 0.0, 0.0])
"""The quaternion that names no turn."""


@dataclass(frozen=True)
class ActionCfg:
  """The impedance on a free rig, and how fast a ramp may ask a hand to move."""

  stiffness: float = 4000.0
  rotational_stiffness: float = 40.0
  max_speed: float = 1.5
  max_turn: float = 10.0

  def build(self, scene: Scene, ramp: Ramp) -> Impedance:
    return Impedance(self, scene.hands, scene.model)


def turn(quaternion: Float64Array) -> Float64Array:
  """The unit quaternion four numbers name, one row or many.

  Four zeros name no turn, so a policy that has not learnt to rotate still
  commands a pose rather than nothing.
  """
  rows = np.atleast_2d(np.asarray(quaternion, np.float64))
  length = np.linalg.norm(rows, axis=1, keepdims=True)
  unit = np.where(length > 1e-8, rows / np.where(length > 0.0, length, 1.0), REST)
  out: Float64Array = unit.reshape(np.shape(quaternion))
  return out


def hold(state: Float64Array) -> Float64Array:
  """A trajectory that asks for one pose across the whole horizon."""
  return np.tile(np.asarray(state, np.float64), (HORIZON, 1))


class Ramp:
  """The commanded trajectory, resampled onto the substeps of one control step.

  A new trajectory departs from the last substep of the one before it.
  """

  def __init__(
    self,
    hands: int,
    substeps: int,
    control_dt: float,
    max_speed: float,
    max_turn: float,
    worlds: int,
  ):
    self.hands, self.substeps, self.worlds = hands, substeps, worlds
    self.max_speed, self.max_turn = max_speed, max_turn
    tracked = control_dt * ACTION_HZ
    if not np.isclose(tracked, round(tracked)):
      raise ValueError("a control step must span a whole number of poses")
    self.tracked = round(tracked)
    """How many of a trajectory's poses one control step executes."""
    if not 0 < self.tracked <= HORIZON:
      raise ValueError(
        f"a control step spans {self.tracked} poses, which a trajectory of "
        f"{HORIZON} does not cover"
      )
    if substeps % self.tracked:
      raise ValueError("a pose must span a whole number of physics steps")
    self.span = substeps // self.tracked
    """How many physics steps one pose spans."""
    self._blend = np.arange(1, self.span + 1) / self.span
    self.position = np.zeros((worlds, substeps, hands, 3))
    self.quaternion = np.zeros((worlds, substeps, hands, 4))
    self.opening = np.zeros((worlds, substeps, hands))
    self.twist = np.zeros((worlds, substeps, hands, 6))
    """Each substep's feedforward: linear velocity, then angular velocity
    already carried into the world frame of its setpoint."""

  def hold(self, worlds: Sequence[int], states: Float64Array) -> None:
    """Hold the named worlds' hands exactly at their states, across the whole
    step; ``states`` is ``(len(worlds), hands, PER_HAND)``."""
    at = list(worlds)
    self.position[at] = states[:, None, :, :3]
    self.quaternion[at] = states[:, None, :, 3:7]
    self.opening[at] = states[:, None, :, 7]
    self.twist[at] = 0.0

  def process(self, action: Float64Array, driven: Sequence[int]) -> None:
    """Resample the leading poses onto the substeps that track them.

    The action names the driven hands; every other hand is asked for its last
    setpoint, so it holds where it was sent. A ramp is laid once per control
    step, so it is built world by world; the controller that reads it runs
    every substep and is not.
    """
    trajectory = np.broadcast_to(
      np.concatenate(
        [self.position[:, -1], self.quaternion[:, -1], self.opening[:, -1, :, None]],
        axis=-1,
      )[:, None],
      (self.worlds, HORIZON, self.hands, PER_HAND),
    ).copy()
    trajectory[:, :, list(driven)] = self._checked(action, len(driven))
    for world in range(self.worlds):
      for hand in range(self.hands):
        legs = trajectory[world, : self.tracked, hand]
        position = np.concatenate([self.position[world, -1, hand][None], legs[:, :3]])
        quaternion = np.concatenate(
          [self.quaternion[world, -1, hand][None], turn(legs[:, 3:7])]
        )
        opening = np.concatenate([self.opening[world, -1, hand][None], legs[:, 7]])
        for leg in range(self.tracked):
          across = slice(leg * self.span, (leg + 1) * self.span)
          step, blend = position[leg + 1] - position[leg], self._blend
          spin = np.empty(3)
          mujoco.mju_subQuat(spin, quaternion[leg + 1], quaternion[leg])
          self.position[world, across, hand] = position[leg] + np.outer(blend, step)
          self.quaternion[world, across, hand] = slerp(quaternion[leg], spin, blend)
          self.opening[world, across, hand] = opening[leg] + blend * (
            opening[leg + 1] - opening[leg]
          )
          twist = self._twist(step, spin)
          self.twist[world, across, hand, :3] = twist[:3]
          self.twist[world, across, hand, 3:] = _rotated(
            self.quaternion[world, across, hand], twist[None, 3:]
          )

  def _checked(self, action: Float64Array, hands: int) -> Float64Array:
    """The driven hands' trajectory, at the shape and finiteness it must have."""
    plan = (HORIZON, PER_HAND * hands)
    wanted = (self.worlds, *plan)
    trajectory = np.asarray(action, np.float64)
    if self.worlds == 1 and trajectory.shape == plan:
      trajectory = trajectory[None]
    if trajectory.shape != wanted:
      raise ValueError(f"an action has shape {wanted}, not {trajectory.shape}")
    if not np.isfinite(trajectory).all():
      raise ValueError("an action must be finite")
    return trajectory.reshape(self.worlds, HORIZON, hands, PER_HAND)

  def _twist(self, step: Float64Array, spin: Float64Array) -> Float64Array:
    """The velocity one leg of the trajectory asks for, capped.

    A cap bounds how much of a carrier's damping a feedforward cancels without
    touching where the setpoint is.
    """
    return np.concatenate(
      [
        _capped(step * ACTION_HZ, self.max_speed),
        _capped(spin * ACTION_HZ, self.max_turn),
      ]
    )


class Impedance:
  """A stiff impedance on each free-flying rig, chasing the ramp with a wrench.

  A rig is a free body, so the pose the impedance answers to is its joint
  coordinates: no forward pass stands between the state and the wrench.
  """

  drives = CTRL | XFRC

  def __init__(self, cfg: ActionCfg, hands: Sequence[Hand], model: mujoco.MjModel):
    self.cfg = cfg
    free = [carried(hand, FreeHand) for hand in hands]
    self._rigs = np.asarray([hand.rig for hand in free], np.int64)
    qadr = np.asarray([hand.qpos for hand in free], np.int64)
    self._at_position = qadr[:, None] + np.arange(3)
    self._at_quaternion = qadr[:, None] + np.arange(3, 7)
    self._damping = np.stack(
      [model.dof_damping[at : at + 6] for at in model.body_dofadr[self._rigs]]
    )
    """What the model slows each rig by, which the feedforward cancels."""
    self._at_fingers = np.asarray([hand.fingers for hand in hands], np.int64)
    self._held = _carried(model, free)
    self.servos = FingerServos(hands)

  def reset(self, datas: Sequence[mujoco.MjData], worlds: Sequence[int]) -> None:
    """A wrench carries no state between steps."""

  def command(
    self,
    ramp: Ramp,
    substep: int,
    qpos: Float64Array,
    qvel: Float64Array,
    ctrl: Float64Array,
    xfrc: Float64Array,
  ) -> None:
    """Write the servo angles and the wrench every world wants for one substep."""
    cfg = self.cfg
    here = qpos[:, self._at_position]
    now = qpos[:, self._at_quaternion]
    want = ramp.position[:, substep]
    facing = ramp.quaternion[:, substep]
    twist = ramp.twist[:, substep]
    force = cfg.stiffness * (want - here) + self._damping[:, :3] * twist[..., :3]
    torque = cfg.rotational_stiffness * _apart(facing, now) + (
      self._damping[:, 3:] * twist[..., 3:]
    )
    lever = _rotated(now, self._mass_at(qpos))
    xfrc[:, self._rigs, :3] = force + self._held.force
    xfrc[:, self._rigs, 3:] = torque + _cross(lever, self._held.force)
    self.servos.command(ctrl, ramp.opening[:, substep])

  def _mass_at(self, qpos: Float64Array) -> Float64Array:
    """Where each gripper's mass sits in its rig's frame, ``(worlds, hands, 3)``."""
    held = self._held
    turned = qpos[:, self._at_fingers, None] - held.rest
    swung = held.cosine * np.cos(turned) + held.sine * np.sin(turned) + held.seated
    out: Float64Array = held.fixed + (held.share * swung).sum(axis=-2)
    return out


@dataclass(frozen=True)
class _Held:
  """What holding a gripper up costs, and where its mass sits.

  A finger's mass swings about its own joint, so its center is ``seated`` plus
  a turn of ``cosine`` and ``sine``; the rest of the gripper is ``fixed``.
  """

  force: Float64Array
  fixed: Float64Array
  share: Float64Array
  cosine: Float64Array
  sine: Float64Array
  seated: Float64Array
  rest: Float64Array


def _carried(model: mujoco.MjModel, hands: Sequence[FreeHand]) -> _Held:
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  force, fixed, share, turned, rest = [], [], [], [], []
  for hand in hands:
    frame = data.xmat[hand.rig].reshape(3, 3)
    origin = data.xpos[hand.rig]
    mass = float(model.body_subtreemass[hand.rig])
    force.append(-mass * np.asarray(model.opt.gravity))
    parts, shares, angles = [], [], []
    for at in hand.fingers:
      joint = int(np.flatnonzero(model.jnt_qposadr == at)[0])
      body = int(model.jnt_bodyid[joint])
      axis = frame.T @ data.xaxis[joint]
      axis = axis / np.linalg.norm(axis)
      anchor = frame.T @ (data.xanchor[joint] - origin)
      arm = frame.T @ (data.subtree_com[body] - origin) - anchor
      along = axis * (axis @ arm)
      parts.append((arm - along, np.cross(axis, arm), along + anchor))
      shares.append(float(model.body_subtreemass[body]) / mass)
      angles.append(float(data.qpos[at]))
    turned.append(parts)
    share.append(shares)
    rest.append(angles)
    fixed.append(
      frame.T @ (data.subtree_com[hand.rig] - origin)
      - sum(s * (p[0] + p[2]) for s, p in zip(shares, parts, strict=True))
    )
  spun = np.asarray(turned, np.float64)
  return _Held(
    force=np.asarray(force, np.float64),
    fixed=np.asarray(fixed, np.float64),
    share=np.asarray(share, np.float64)[..., None],
    cosine=spun[:, :, 0],
    sine=spun[:, :, 1],
    seated=spun[:, :, 2],
    rest=np.asarray(rest, np.float64)[..., None],
  )


def _capped(vector: Float64Array, limit: float) -> Float64Array:
  norm = float(np.linalg.norm(vector))
  return vector * (limit / norm) if norm > limit else vector


def slerp(start: Float64Array, spin: Float64Array, blend: Float64Array) -> Float64Array:
  """Quaternions along the geodesic that ``spin`` names out of ``start``."""
  half = np.outer(blend, spin) * 0.5
  angle = np.linalg.norm(half, axis=1, keepdims=True)
  axis = np.where(angle > 1e-12, half / np.where(angle > 0.0, angle, 1.0), half)
  ahead = np.concatenate([np.cos(angle), np.sin(angle) * axis], axis=1)
  return _joined(start[None, :], ahead)


def _cross(a: Float64Array, b: Float64Array) -> Float64Array:
  """``a × b`` over any matching leading axes, without numpy's axis bookkeeping."""
  out = np.empty(np.broadcast_shapes(a.shape, b.shape))
  out[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
  out[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
  out[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
  return out


def _joined(first: Float64Array, then: Float64Array) -> Float64Array:
  """The quaternion product ``first ⊗ then``, over any matching leading axes."""
  w1, v1 = first[..., :1], first[..., 1:]
  w2, v2 = then[..., :1], then[..., 1:]
  w = w1 * w2 - np.sum(v1 * v2, axis=-1, keepdims=True)
  v = w1 * v2 + w2 * v1 + _cross(v1, v2)
  return np.concatenate([w, v], axis=-1)


def _apart(want: Float64Array, here: Float64Array) -> Float64Array:
  """The rotation vector carrying each ``here`` onto its ``want``, stable
  everywhere a rotation is: the angle comes from ``atan2``, so a half turn is
  as sound as no turn at all."""
  conj = np.concatenate([here[..., :1], -here[..., 1:]], axis=-1)
  rel = _joined(want, conj)
  rel = np.where(rel[..., :1] < 0.0, -rel, rel)
  spin = rel[..., 1:]
  swing = np.linalg.norm(spin, axis=-1, keepdims=True)
  angle = 2.0 * np.arctan2(swing, rel[..., :1])
  # As the swing vanishes the ratio tends to two, whatever the axis was.
  return np.where(
    swing > 1e-12, angle * spin / np.where(swing > 0.0, swing, 1.0), 2.0 * spin
  )


def _rotated(quaternion: Float64Array, vector: Float64Array) -> Float64Array:
  """Each ``vector`` carried through its ``quaternion``."""
  w, axis = quaternion[..., :1], quaternion[..., 1:]
  lift = _cross(axis, vector)
  out: Float64Array = vector + 2.0 * (w * lift + _cross(axis, lift))
  return out
