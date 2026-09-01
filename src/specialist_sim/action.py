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

from specialist_sim import pose as se3
from specialist_sim.arrays import Float64Array
from specialist_sim.spaces import (
  ACTION_HIGH,
  ACTION_HZ,
  ACTION_LOW,
  GRIPPERS,
  HORIZON,
  PER_HAND,
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
    self._within = np.arange(1, self.span + 1) / self.span
    hands = len(GRIPPERS)
    self._position = np.zeros((worlds, hands, 3))
    self._quaternion = np.zeros((worlds, hands, 4))
    self._opening = np.zeros((worlds, hands))
    self._ramp_position = np.zeros((worlds, substeps, hands, 3))
    self._ramp_rotation = np.zeros((worlds, substeps, hands, 3, 3))
    self._ramp_opening = np.zeros((worlds, substeps, hands))
    self._ramp_twist = np.zeros((worlds, substeps, hands, 6))
    self._rigs = np.asarray(self._rig, np.int64)
    self._qadr = np.asarray([grip.qpos for grip in grippers], np.int64)

  def reset(
    self, datas: Sequence[mujoco.MjData], worlds: Sequence[int] | None = None
  ) -> None:
    """Hold the named worlds' hands exactly where their scenes put them."""
    for world in range(len(datas)) if worlds is None else worlds:
      data = datas[world]
      for hand, body in enumerate(self._rig):
        self._position[world, hand] = data.xpos[body]
        self._quaternion[world, hand] = se3.wxyz(data.xmat[body].reshape(3, 3))
        self._opening[world, hand] = float(data.qpos[self._finger[hand]])
        self._ramp_position[world, :, hand] = self._position[world, hand]
        self._ramp_rotation[world, :, hand] = data.xmat[body].reshape(3, 3)
        self._ramp_opening[world, :, hand] = self._opening[world, hand]
      self._ramp_twist[world] = 0.0

  def process(self, action: Float64Array) -> None:
    """Resample the leading poses onto the substeps that track them.

    A ramp is laid once per control step, so it is built world by world; the
    wrench that reads it runs every substep and is not.
    """
    trajectory = self._checked(action)
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
          step, blend = position[leg + 1] - position[leg], self._within
          self._ramp_position[world, across, hand] = position[leg] + np.outer(
            blend, step
          )
          self._ramp_rotation[world, across, hand] = slerp(
            quaternion[leg], quaternion[leg + 1], blend
          )
          self._ramp_opening[world, across, hand] = opening[leg] + blend * (
            opening[leg + 1] - opening[leg]
          )
          self._ramp_twist[world, across, hand] = self._twist(
            step, quaternion[leg], quaternion[leg + 1]
          )
        self._position[world, hand] = position[-1]
        self._quaternion[world, hand] = quaternion[-1]
        self._opening[world, hand] = opening[-1]

  def command(
    self, qpos: Float64Array, substep: int, nbody: int, nu: int
  ) -> tuple[Float64Array, Float64Array]:
    """The servo angles and the wrench every world wants for one substep.

    A rig is a free body, so the pose the impedance answers to is its joint
    coordinates: no forward pass stands between the state and the wrench.
    """
    cfg = self.cfg
    at = self._qadr
    here = qpos[:, at[:, None] + np.arange(3)]
    now = qpos[:, at[:, None] + np.arange(3, 7)]
    want = self._ramp_position[:, substep]
    rotation = self._ramp_rotation[:, substep]
    twist = self._ramp_twist[:, substep]
    turned = _turned(rotation, _matrices(now))
    force = cfg.stiffness * (want - here) + cfg.damping * twist[..., :3]
    torque = cfg.rotational_stiffness * turned + cfg.rotational_damping * np.einsum(
      "whij,whj->whi", rotation, twist[..., 3:]
    )
    xfrc = np.zeros((len(qpos), nbody, 6))
    xfrc[:, self._rigs, :3] = force
    xfrc[:, self._rigs, 3:] = torque
    angle = self._ramp_opening[:, substep]
    ctrl = np.zeros((len(qpos), nu))
    for hand, servos in enumerate(self._servos):
      ctrl[:, servos[0]] = angle[:, hand]
      ctrl[:, servos[1]] = -angle[:, hand]
    return ctrl, xfrc.reshape(len(qpos), nbody * 6)

  def _checked(self, action: Float64Array) -> Float64Array:
    wanted = (self.worlds, *ACTION_LOW.shape)
    trajectory = np.asarray(action, np.float64)
    if self.worlds == 1 and trajectory.shape == ACTION_LOW.shape:
      trajectory = trajectory[None]
    if trajectory.shape != wanted:
      raise ValueError(f"an action has shape {wanted}, not {trajectory.shape}")
    if not np.isfinite(trajectory).all():
      raise ValueError("an action must be finite")
    clipped = np.clip(trajectory, ACTION_LOW, ACTION_HIGH)
    return clipped.reshape(self.worlds, HORIZON, len(GRIPPERS), PER_HAND)

  def _twist(
    self, step: Float64Array, turn_from: Float64Array, turn_to: Float64Array
  ) -> Float64Array:
    """The velocity one leg of the trajectory asks for, capped.

    It exists to cancel the free body's damping, so a cap bounds how much of it
    is cancelled without touching where the setpoint is.
    """
    turn = np.empty(3)
    mujoco.mju_subQuat(turn, turn_to, turn_from)
    rate = ACTION_HZ
    return np.concatenate(
      [
        _capped(step * rate, self.cfg.max_speed),
        _capped(turn * rate, self.cfg.max_turn),
      ]
    )


def _matrices(quaternion: Float64Array) -> Float64Array:
  """The rotations a stack of ``wxyz`` quaternions name."""
  w, x, y, z = (quaternion[..., i] for i in range(4))
  out = np.empty(quaternion.shape[:-1] + (3, 3))
  out[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
  out[..., 0, 1] = 2.0 * (x * y - z * w)
  out[..., 0, 2] = 2.0 * (x * z + y * w)
  out[..., 1, 0] = 2.0 * (x * y + z * w)
  out[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
  out[..., 1, 2] = 2.0 * (y * z - x * w)
  out[..., 2, 0] = 2.0 * (x * z - y * w)
  out[..., 2, 1] = 2.0 * (y * z + x * w)
  out[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
  return out


def _turned(want: Float64Array, here: Float64Array) -> Float64Array:
  """The rotation vector carrying each ``here`` onto its ``want``."""
  spin = want @ np.swapaxes(here, -1, -2)
  trace = spin[..., 0, 0] + spin[..., 1, 1] + spin[..., 2, 2]
  angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
  axis = np.stack(
    [
      spin[..., 2, 1] - spin[..., 1, 2],
      spin[..., 0, 2] - spin[..., 2, 0],
      spin[..., 1, 0] - spin[..., 0, 1],
    ],
    axis=-1,
  )
  # Near zero the axis vanishes with the angle and the ratio tends to a half.
  small = angle < 1e-8
  scale = np.where(small, 0.5, angle / np.where(small, 1.0, 2.0 * np.sin(angle)))
  return axis * scale[..., None]


def _capped(vector: Float64Array, limit: float) -> Float64Array:
  norm = float(np.linalg.norm(vector))
  return vector * (limit / norm) if norm > limit else vector


def slerp(start: Float64Array, end: Float64Array, blend: Float64Array) -> Float64Array:
  """Rotation matrices along the geodesic from ``start`` to ``end``."""
  turn = np.empty(3)
  mujoco.mju_subQuat(turn, end, start)
  out = np.empty((len(blend), 3, 3))
  matrix = np.empty(9)
  for index, fraction in enumerate(blend):
    step = start.copy()
    mujoco.mju_quatIntegrate(step, turn, float(fraction))
    mujoco.mju_quat2Mat(matrix, step)
    out[index] = matrix.reshape(3, 3)
  return out
