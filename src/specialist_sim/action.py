"""A commanded trajectory per gripper, and the wrench that chases it.

An action is a trajectory of absolute poses in the room frame,
``[position(3), quaternion(4), opening(1)]`` per hand, at ``1 / ACTION_HZ``
intervals from now. An opening is the coupled finger angle in radians, signed as
the left finger turns, which is the number a corpus records. A control step tracks the leading poses and drops the rest,
so a policy commits to further ahead than it executes.

The tracker resamples the trajectory at the physics rate. The impedance's
natural frequency sits above the control rate and below the trajectory's, so a
setpoint that jumped once per control step would ring at it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.spaces import (
  ACTION_HZ,
  GRIPPERS,
  HORIZON,
  PER_HAND,
  action_bounds,
)

if TYPE_CHECKING:
  from specialist_sim.scene import Gripper

__all__ = [
  "ActionCfg",
  "Hands",
  "hold",
  "slerp",
  "turn",
]

REST = np.array([1.0, 0.0, 0.0, 0.0])
"""The quaternion that names no turn."""


@dataclass(frozen=True)
class ActionCfg:
  stiffness: float = 4000.0
  damping: float = 60.0
  rotational_stiffness: float = 40.0
  rotational_damping: float = 0.5
  max_speed: float = 1.5
  max_turn: float = 10.0


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


class Hands:
  """The action term: decode a trajectory, resample it, and drive the model."""

  def __init__(
    self,
    grippers: Sequence[Gripper],
    substeps: int,
    control_dt: float,
    cfg: ActionCfg,
    worlds: int = 1,
  ):
    self.cfg, self.substeps, self.worlds = cfg, substeps, worlds
    self._rig = [grip.rig for grip in grippers]
    self._servos = [list(grip.servos) for grip in grippers]
    self._finger = [grip.fingers[0] for grip in grippers]
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
    hands = len(GRIPPERS)
    self._position = np.zeros((worlds, hands, 3))
    self._quaternion = np.zeros((worlds, hands, 4))
    self._opening = np.zeros((worlds, hands))
    self._ramp_position = np.zeros((worlds, substeps, hands, 3))
    self._ramp_quaternion = np.zeros((worlds, substeps, hands, 4))
    self._ramp_opening = np.zeros((worlds, substeps, hands))
    self._ramp_twist = np.zeros((worlds, substeps, hands, 6))
    """Each substep's feedforward: linear velocity, then angular velocity
    already carried into the world frame of its setpoint."""
    self._rigs = np.asarray(self._rig, np.int64)
    qadr = np.asarray([grip.qpos for grip in grippers], np.int64)
    self._at_position = qadr[:, None] + np.arange(3)
    self._at_quaternion = qadr[:, None] + np.arange(3, 7)
    self._servo_left = np.asarray([servos[0] for servos in self._servos], np.int64)
    self._servo_right = np.asarray([servos[1] for servos in self._servos], np.int64)
    self.clipped = np.zeros(worlds, np.int64)

  def reset(
    self, datas: Sequence[mujoco.MjData], worlds: Sequence[int] | None = None
  ) -> None:
    """Hold the named worlds' hands exactly where their scenes put them."""
    for world in range(len(datas)) if worlds is None else worlds:
      data = datas[world]
      for hand in range(len(GRIPPERS)):
        self._position[world, hand] = data.qpos[self._at_position[hand]]
        self._quaternion[world, hand] = data.qpos[self._at_quaternion[hand]]
        self._opening[world, hand] = float(data.qpos[self._finger[hand]])
        self._ramp_position[world, :, hand] = self._position[world, hand]
        self._ramp_quaternion[world, :, hand] = self._quaternion[world, hand]
        self._ramp_opening[world, :, hand] = self._opening[world, hand]
      self._ramp_twist[world] = 0.0

  def process(self, action: Float64Array, driven: Sequence[int]) -> None:
    """Resample the leading poses onto the substeps that track them.

    The action names the driven hands; every other hand is asked for its last
    setpoint, so it holds where it was sent. A ramp is laid once per control
    step, so it is built world by world; the wrench that reads it runs every
    substep and is not.
    """
    trajectory = np.broadcast_to(
      np.concatenate(
        [self._position, self._quaternion, self._opening[..., None]], axis=-1
      )[:, None],
      (self.worlds, HORIZON, len(GRIPPERS), PER_HAND),
    ).copy()
    trajectory[:, :, list(driven)] = self._checked(action, len(driven))
    for world in range(self.worlds):
      for hand in range(len(GRIPPERS)):
        legs = trajectory[world, : self.tracked, hand]
        held = self._position[world, hand][None]
        position = np.concatenate([held, legs[:, :3]])
        quaternion = np.concatenate(
          [self._quaternion[world, hand][None], turn(legs[:, 3:7])]
        )
        opening = np.concatenate([self._opening[world, hand][None], legs[:, 7]])
        for leg in range(self.tracked):
          across = slice(leg * self.span, (leg + 1) * self.span)
          step, blend = position[leg + 1] - position[leg], self._blend
          spin = np.empty(3)
          mujoco.mju_subQuat(spin, quaternion[leg + 1], quaternion[leg])
          self._ramp_position[world, across, hand] = position[leg] + np.outer(
            blend, step
          )
          self._ramp_quaternion[world, across, hand] = slerp(
            quaternion[leg], spin, blend
          )
          self._ramp_opening[world, across, hand] = opening[leg] + blend * (
            opening[leg + 1] - opening[leg]
          )
          twist = self._twist(step, spin)
          self._ramp_twist[world, across, hand, :3] = twist[:3]
          self._ramp_twist[world, across, hand, 3:] = _rotated(
            self._ramp_quaternion[world, across, hand], twist[None, 3:]
          )
        self._position[world, hand] = position[-1]
        self._quaternion[world, hand] = quaternion[-1]
        self._opening[world, hand] = opening[-1]

  def error(self, qpos: Float64Array, driven: Sequence[int]) -> Float64Array:
    """Each world's largest gap between a driven hand and its setpoint, in metres."""
    gaps = np.linalg.norm(
      qpos[:, self._at_position[list(driven)]] - self._position[:, list(driven)],
      axis=-1,
    )
    out: Float64Array = gaps.max(axis=1)
    return out

  def command(
    self, qpos: Float64Array, substep: int, ctrl: Float64Array, xfrc: Float64Array
  ) -> None:
    """Write the servo angles and the wrench every world wants for one substep.

    A rig is a free body, so the pose the impedance answers to is its joint
    coordinates: no forward pass stands between the state and the wrench. Only
    the rig and servo entries are ever written, so the rest of the buffers stay
    the zeros they were built with.
    """
    cfg = self.cfg
    here = qpos[:, self._at_position]
    now = qpos[:, self._at_quaternion]
    want = self._ramp_position[:, substep]
    facing = self._ramp_quaternion[:, substep]
    twist = self._ramp_twist[:, substep]
    force = cfg.stiffness * (want - here) + cfg.damping * twist[..., :3]
    torque = cfg.rotational_stiffness * _apart(facing, now) + (
      cfg.rotational_damping * twist[..., 3:]
    )
    xfrc[:, self._rigs, :3] = force
    xfrc[:, self._rigs, 3:] = torque
    angle = self._ramp_opening[:, substep]
    ctrl[:, self._servo_left] = angle
    ctrl[:, self._servo_right] = -angle

  def _checked(self, action: Float64Array, hands: int) -> Float64Array:
    """The driven hands' trajectory, clipped to the box; ``clipped`` counts
    how many entries of each world's action the box cut."""
    low, high = action_bounds(hands)
    wanted = (self.worlds, *low.shape)
    trajectory = np.asarray(action, np.float64)
    if self.worlds == 1 and trajectory.shape == low.shape:
      trajectory = trajectory[None]
    if trajectory.shape != wanted:
      raise ValueError(f"an action has shape {wanted}, not {trajectory.shape}")
    if not np.isfinite(trajectory).all():
      raise ValueError("an action must be finite")
    clipped = np.clip(trajectory, low, high)
    self.clipped = (clipped != trajectory).reshape(self.worlds, -1).sum(axis=1)
    return clipped.reshape(self.worlds, HORIZON, hands, PER_HAND)

  def _twist(self, step: Float64Array, spin: Float64Array) -> Float64Array:
    """The velocity one leg of the trajectory asks for, capped.

    It exists to cancel the free body's damping, so a cap bounds how much of it
    is cancelled without touching where the setpoint is.
    """
    return np.concatenate(
      [
        _capped(step * ACTION_HZ, self.cfg.max_speed),
        _capped(spin * ACTION_HZ, self.cfg.max_turn),
      ]
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
