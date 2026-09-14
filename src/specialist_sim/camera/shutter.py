"""When each lens exposes, and which of its frames reaches a policy. A lens
fires on the physics lattice at ``CAMERA_HZ`` from a phase each episode sets.
Time counts physics steps from the episode's opening; a frame is filed in rows.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from specialist_sim.arrays import Float64Array, Int32Array, Int64Array, UInt8Array

CAMERA_HZ = 30000.0 / 1001.0


@dataclass(frozen=True)
class _Frame:
  """One exposure: the instant its shutter fell, in rows, and what it saw."""

  at: float
  image: UInt8Array
  body: Int32Array


class Shutter:
  """Every lens of ``num_envs`` envs, ``count`` each: its exposures to come,
  and the frames it holds until each reaches a policy ``latency`` after its shutter."""

  def __init__(
    self,
    num_envs: int,
    count: int,
    physics_dt: float,
    per_row: int,
    per_step: int,
    latency: float,
  ):
    # Physics steps from one exposure to the next.
    self._period = 1.0 / (CAMERA_HZ * physics_dt)
    self._per_row = per_row
    self._span = per_row * per_step
    # The latency in physics steps and in rows.
    self._steps = latency / physics_dt
    self._rows = latency / (physics_dt * per_row)
    self._phase = np.zeros((num_envs, count))
    # Each lens's next exposure, counted from its first, and its physics step.
    self._shot = np.zeros((num_envs, count), np.int64)
    self._next = np.zeros((num_envs, count), np.int64)
    # What each env's lenses hold, oldest first.
    self._frames: list[list[list[_Frame]]] = [
      [[] for _ in range(count)] for _ in range(num_envs)
    ]

  def open(
    self,
    env_ids: Sequence[int],
    phases: Float64Array,
    image: UInt8Array,
    body: Int32Array,
  ) -> None:
    """Open the named envs' lenses at ``phases``, ``(k, lenses)`` in ``[0, 1)``,
    as if already running: each holds frames back past the latency, all
    ``image[k, lens]``."""
    for k, env in enumerate(env_ids):
      self._phase[env] = phases[k]
      for lens in range(self._phase.shape[1]):
        last = math.floor(0.5 / self._period - self._phase[env, lens])
        first = last - math.ceil(self._rows / (self._period / self._per_row)) - 1
        self._shot[env, lens] = last + 1
        self._next[env, lens] = self._shutter(last + 1, env, lens)
        self._frames[env][lens] = [
          _Frame(at / self._per_row, image[k, lens], body[k, lens])
          for at in (self._shutter(shot, env, lens) for shot in range(first, last + 1))
        ]

  def shots(self, elapsed: Int64Array) -> list[tuple[int, list[tuple[int, list[int]]]]]:
    """The steps 1 to ``per_row`` into the row starting at ``elapsed``,
    ``(num_envs,)``, that expose a frame to trace, each with its lenses and
    envs; every lens winds past the row."""
    end = elapsed + self._per_row
    fired: dict[int, dict[int, list[int]]] = {}
    for env, lens in np.argwhere(self._next <= end[:, None]).tolist():
      while self._next[env, lens] <= end[env]:
        at = int(self._next[env, lens])
        if self._wanted(env, lens, at):
          offset = at - int(elapsed[env])
          fired.setdefault(offset, {}).setdefault(lens, []).append(env)
        self._shot[env, lens] += 1
        self._next[env, lens] = self._shutter(int(self._shot[env, lens]), env, lens)
    return [
      (offset, sorted(lenses.items())) for offset, lenses in sorted(fired.items())
    ]

  def keep(
    self,
    lens: int,
    env_ids: Sequence[int],
    at: Int64Array,
    image: UInt8Array,
    body: Int32Array,
  ) -> None:
    """File one lens's frames in the named envs, taken ``at`` physics steps in."""
    for k, env in enumerate(env_ids):
      self._frames[env][lens].append(
        _Frame(int(at[k]) / self._per_row, image[k], body[k])
      )

  def deliver(self, row: Int64Array) -> tuple[UInt8Array, Int32Array, Int64Array]:
    """Each lens's newest frame arrived by each env's ``row``, older ones dropped:
    images ``(num_envs, lenses, S, S, 3)``, bodies, and the nearest row of each."""
    delivered = []
    for env, held in enumerate(self._frames):
      cutoff = row[env] - self._rows
      for lens, frames in enumerate(held):
        newest = max(index for index, frame in enumerate(frames) if frame.at <= cutoff)
        held[lens] = frames[newest:]
      delivered.append([frames[0] for frames in held])
    return (
      np.array([[frame.image for frame in env] for env in delivered], np.uint8),
      np.array([[frame.body for frame in env] for env in delivered], np.int32),
      np.array([[round(frame.at) for frame in env] for env in delivered], np.int64),
    )

  def _shutter(self, shot: int, env: int, lens: int) -> int:
    """The physics step exposure ``shot`` falls on."""
    return int(round((shot + self._phase[env, lens]) * self._period))

  def _wanted(self, env: int, lens: int, at: int) -> bool:
    """Whether the exposure falling ``at`` is the newest arrived at the first
    observation it reaches, and so is traced."""
    cutoff = math.ceil((at + self._steps) / self._span) * self._span - self._steps
    return self._shutter(int(self._shot[env, lens]) + 1, env, lens) > cutoff
