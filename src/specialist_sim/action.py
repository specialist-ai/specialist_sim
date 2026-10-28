"""The action: an action chunk per hand, each action ``[position(3),
quaternion(4), opening(1)]`` in the room frame, one action period apart from now.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array

__all__ = ["CHUNK_SIZE", "Interpolator", "hold"]

CHUNK_SIZE = 16
"""How many actions a policy predicts in one chunk."""


def hold(state: Float64Array, chunk_size: int = CHUNK_SIZE) -> Float64Array:
  """An action chunk that asks for one state, ``(..., hands, 8)``, at every action."""
  out: Float64Array = np.repeat(np.expand_dims(state, -3), chunk_size, axis=-3)
  return out


class Interpolator:
  """The executed actions interpolated onto the substeps of one control step:
  each substep's setpoint, ``(num_envs, substeps, hands, ...)``, and its twist,
  linear then angular, in the room frame."""

  def __init__(
    self, num_envs: int, hands: int, n_action_steps: int, span: int, action_dt: float
  ):
    self.n_action_steps, self.span, self.action_dt = n_action_steps, span, action_dt
    """How many actions a control step executes, how many substeps each spans,
    and how long an action period is."""
    substeps = n_action_steps * span
    self.position = np.zeros((num_envs, substeps, hands, 3))
    self.quaternion = np.zeros((num_envs, substeps, hands, 4))
    self.quaternion[..., 0] = 1.0
    self.opening = np.zeros((num_envs, substeps, hands))
    self.twist = np.zeros((num_envs, substeps, hands, 6))
    self._blend = (np.arange(1, span + 1) / span)[None, None, :, None, None]

  @property
  def last(self) -> Float64Array:
    """Where the interpolation ends, ``(num_envs, hands, 8)``."""
    out: Float64Array = np.concatenate(
      [self.position[:, -1], self.quaternion[:, -1], self.opening[:, -1, :, None]],
      axis=-1,
    )
    return out

  def hold(self, env_ids: Sequence[int], states: Float64Array) -> None:
    """Hold the named envs' hands at their states, ``(len(env_ids), hands,
    8)``, across the whole step."""
    at = list(env_ids)
    self.position[at] = states[:, None, :, :3]
    self.quaternion[at] = states[:, None, :, 3:7]
    self.opening[at] = states[:, None, :, 7]
    self.twist[at] = 0.0

  def process(self, action: Float64Array) -> None:
    """Interpolate the leading actions of ``action``, ``(num_envs, chunk_size,
    hands, 8)``, onto the substeps."""
    num_envs, _, hands, _ = action.shape
    knots = np.concatenate(
      [self.last[:, None], action[:, : self.n_action_steps]], axis=1
    )
    position, opening = knots[..., :3], knots[..., 7]
    quaternion = rotation.normalize(knots[..., 3:7])
    step = position[:, 1:] - position[:, :-1]
    spin = rotation.log(
      rotation.multiply(rotation.conjugate(quaternion[:, :-1]), quaternion[:, 1:])
    )
    blend = self._blend
    shape = (num_envs, self.n_action_steps, self.span, hands)
    self.position = (position[:, :-1, None] + blend * step[:, :, None]).reshape(
      num_envs, -1, hands, 3
    )
    self.quaternion = rotation.multiply(
      quaternion[:, :-1, None], rotation.exp(blend * spin[:, :, None])
    ).reshape(num_envs, -1, hands, 4)
    self.opening = (
      opening[:, :-1, None]
      + blend[..., 0] * (opening[:, 1:] - opening[:, :-1])[:, :, None]
    ).reshape(num_envs, -1, hands)
    linear = np.broadcast_to(step[:, :, None] / self.action_dt, (*shape, 3))
    angular = rotation.rotate(
      self.quaternion,
      np.broadcast_to(spin[:, :, None] / self.action_dt, (*shape, 3)).reshape(
        num_envs, -1, hands, 3
      ),
    )
    self.twist = np.concatenate([linear.reshape(num_envs, -1, hands, 3), angular], -1)
