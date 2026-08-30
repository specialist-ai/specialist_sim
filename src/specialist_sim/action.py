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

FINGER_SHUT = np.radians(-9.1175)
"""The angle at which the pads meet, which a gripper command of zero asks for."""
FINGER_OPEN = np.radians(40.0)
"""The joint's open limit, which a gripper command of one asks for."""


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


def gripper_from_opening(opening: float) -> float:
  """The joint angle an opening asks for, saturating at shut and open."""
  span = float(np.clip(opening, 0.0, 1.0))
  return float(FINGER_SHUT + span * (FINGER_OPEN - FINGER_SHUT))


def opening_from_gripper(angle: float) -> float:
  """The opening an angle stands for, saturating at shut and open."""
  return float(np.clip((angle - FINGER_SHUT) / (FINGER_OPEN - FINGER_SHUT), 0.0, 1.0))


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
  ):
    self.cfg, self.substeps = cfg, substeps
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
    self._position = np.zeros((hands, 3))
    self._quaternion = np.zeros((hands, 4))
    self._opening = np.zeros(hands)
    self._ramp_position = np.zeros((substeps, hands, 3))
    self._ramp_rotation = np.zeros((substeps, hands, 3, 3))
    self._ramp_opening = np.zeros((substeps, hands))
    self._ramp_twist = np.zeros((substeps, hands, 6))

  def reset(self, data: mujoco.MjData) -> None:
    """Hold every hand exactly where the scene put it."""
    for hand, body in enumerate(self._rig):
      self._position[hand] = data.xpos[body]
      self._quaternion[hand] = se3.wxyz(data.xmat[body].reshape(3, 3))
      self._opening[hand] = float(data.qpos[self._finger[hand]])
      self._ramp_position[:, hand] = self._position[hand]
      self._ramp_rotation[:, hand] = data.xmat[body].reshape(3, 3)
      self._ramp_opening[:, hand] = self._opening[hand]
    self._ramp_twist[:] = 0.0
    for hand in range(len(GRIPPERS)):
      self._command_fingers(data, hand, self._opening[hand])

  def process(self, action: Float64Array, data: mujoco.MjData) -> None:
    """Resample the leading poses onto the substeps that track them."""
    trajectory = self._checked(action)
    for hand in range(len(GRIPPERS)):
      legs = trajectory[: self.tracked, hand]
      position = np.concatenate([self._position[hand][None], legs[:, :3]])
      quaternion = np.concatenate([self._quaternion[hand][None], turn(legs[:, 3:7])])
      opening = np.concatenate([self._opening[hand][None], legs[:, 7]])
      for leg in range(self.tracked):
        across = slice(leg * self.span, (leg + 1) * self.span)
        step, blend = position[leg + 1] - position[leg], self._within
        self._ramp_position[across, hand] = position[leg] + np.outer(blend, step)
        self._ramp_rotation[across, hand] = slerp(
          quaternion[leg], quaternion[leg + 1], blend
        )
        self._ramp_opening[across, hand] = opening[leg] + blend * (
          opening[leg + 1] - opening[leg]
        )
        self._ramp_twist[across, hand] = self._twist(
          step, quaternion[leg], quaternion[leg + 1]
        )
      self._position[hand] = position[-1]
      self._quaternion[hand] = quaternion[-1]
      self._opening[hand] = opening[-1]

  def apply(self, data: mujoco.MjData, substep: int) -> None:
    for hand, body in enumerate(self._rig):
      self._command_fingers(data, hand, float(self._ramp_opening[substep, hand]))
      self._pull(
        data,
        body,
        self._ramp_position[substep, hand],
        self._ramp_rotation[substep, hand],
        self._ramp_twist[substep, hand],
      )

  def _checked(self, action: Float64Array) -> Float64Array:
    trajectory = np.asarray(action, np.float64)
    if trajectory.shape != ACTION_LOW.shape:
      raise ValueError(
        f"an action has shape {ACTION_LOW.shape}, not {trajectory.shape}"
      )
    if not np.isfinite(trajectory).all():
      raise ValueError("an action must be finite")
    clipped = np.clip(trajectory, ACTION_LOW, ACTION_HIGH)
    return clipped.reshape(HORIZON, len(GRIPPERS), PER_HAND)

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

  def _command_fingers(self, data: mujoco.MjData, hand: int, angle: float) -> None:
    data.ctrl[self._servos[hand]] = (angle, -angle)

  def _pull(
    self,
    data: mujoco.MjData,
    body: int,
    position: Float64Array,
    rotation: Float64Array,
    twist: Float64Array,
  ) -> None:
    cfg = self.cfg
    turn = se3.log(rotation @ data.xmat[body].reshape(3, 3).T)
    data.xfrc_applied[body, :3] = cfg.stiffness * (position - data.xpos[body]) + (
      cfg.damping * twist[:3]
    )
    data.xfrc_applied[body, 3:] = cfg.rotational_stiffness * turn + (
      cfg.rotational_damping * (rotation @ twist[3:])
    )


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
