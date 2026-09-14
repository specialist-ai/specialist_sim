"""The action: a trajectory of poses per hand, ``[position(3),
quaternion(4), opening(1)]`` in the room frame, one pose period apart from now.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array

__all__ = ["HORIZON", "Ramp", "hold"]

HORIZON = 16
"""How many poses a policy names."""


def hold(state: Float64Array, poses: int = HORIZON) -> Float64Array:
  """A trajectory that asks for one state, ``(..., hands, 8)``, at every pose."""
  out: Float64Array = np.repeat(np.expand_dims(state, -3), poses, axis=-3)
  return out


class Ramp:
  """The tracked poses laid onto the substeps of one control step: each
  substep's setpoint, ``(worlds, substeps, hands, ...)``, and its twist,
  linear then angular, in the room frame."""

  def __init__(self, worlds: int, hands: int, tracked: int, span: int, pose_dt: float):
    self.tracked, self.span, self.pose_dt = tracked, span, pose_dt
    """How many poses a control step tracks, how many substeps each spans,
    and how long a pose period is."""
    substeps = tracked * span
    self.position = np.zeros((worlds, substeps, hands, 3))
    self.quaternion = np.zeros((worlds, substeps, hands, 4))
    self.quaternion[..., 0] = 1.0
    self.opening = np.zeros((worlds, substeps, hands))
    self.twist = np.zeros((worlds, substeps, hands, 6))
    self._blend = (np.arange(1, span + 1) / span)[None, None, :, None, None]

  @property
  def last(self) -> Float64Array:
    """Where the ramp ends, ``(worlds, hands, 8)``."""
    out: Float64Array = np.concatenate(
      [self.position[:, -1], self.quaternion[:, -1], self.opening[:, -1, :, None]],
      axis=-1,
    )
    return out

  def hold(self, worlds: Sequence[int], states: Float64Array) -> None:
    """Hold the named worlds' hands at their states, ``(len(worlds), hands,
    8)``, across the whole step."""
    at = list(worlds)
    self.position[at] = states[:, None, :, :3]
    self.quaternion[at] = states[:, None, :, 3:7]
    self.opening[at] = states[:, None, :, 7]
    self.twist[at] = 0.0

  def process(self, action: Float64Array) -> None:
    """Lay the leading poses of ``action``, ``(worlds, poses, hands, 8)``, onto
    the substeps."""
    worlds, _, hands, _ = action.shape
    legs = np.concatenate([self.last[:, None], action[:, : self.tracked]], axis=1)
    position, opening = legs[..., :3], legs[..., 7]
    quaternion = rotation.normalize(legs[..., 3:7])
    step = position[:, 1:] - position[:, :-1]
    spin = rotation.log(
      rotation.multiply(rotation.conjugate(quaternion[:, :-1]), quaternion[:, 1:])
    )
    blend = self._blend
    shape = (worlds, self.tracked, self.span, hands)
    self.position = (position[:, :-1, None] + blend * step[:, :, None]).reshape(
      worlds, -1, hands, 3
    )
    self.quaternion = rotation.multiply(
      quaternion[:, :-1, None], rotation.exp(blend * spin[:, :, None])
    ).reshape(worlds, -1, hands, 4)
    self.opening = (
      opening[:, :-1, None]
      + blend[..., 0] * (opening[:, 1:] - opening[:, :-1])[:, :, None]
    ).reshape(worlds, -1, hands)
    linear = np.broadcast_to(step[:, :, None] / self.pose_dt, (*shape, 3))
    angular = rotation.rotate(
      self.quaternion,
      np.broadcast_to(spin[:, :, None] / self.pose_dt, (*shape, 3)).reshape(
        worlds, -1, hands, 3
      ),
    )
    self.twist = np.concatenate([linear.reshape(worlds, -1, hands, 3), angular], -1)
