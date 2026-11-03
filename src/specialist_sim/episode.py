"""A recorded episode on disk, ``<folder>/<name>.npz``, every row of it.

Row ``k`` is the env at ``timestamp[k]``, seconds from the episode's first row:

* ``state``: the hands as they stood, ``(rows, hands, 8)``. A dataset reads its
  state and, in the UMI formulation, its actions from this.
* ``commanded``: the setpoint each hand was chasing, ``(rows, hands, 8)``: the
  commanded formulation's actions, kept for checks.
* ``qpos``, ``qvel``: the whole env. No image is stored: a dataset build traces
  the lenses from ``qpos``, so a change to a lens needs no new recording.
* ``meta``: JSON with the source, the scene it was recorded in, the knobs it
  was stood with, the clock it ran on, and how the run scored. ``env_from``
  stands the same env again from it.

A hand is ``[position(3), wxyz(4), opening(1)]`` in the room frame, in
``gripper.SLOTS`` order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from specialist_sim.arrays import Float64Array

__all__ = ["Episode", "Recorder", "listing", "load", "save"]


@dataclass(frozen=True)
class Episode:
  name: str
  timestamp: Float64Array
  state: Float64Array
  commanded: Float64Array
  qpos: Float64Array
  qvel: Float64Array
  meta: dict[str, Any]


def save(folder: Path, episode: Episode) -> Path:
  folder.mkdir(parents=True, exist_ok=True)
  path = folder / f"{episode.name}.npz"
  np.savez_compressed(
    path,
    timestamp=episode.timestamp,
    state=episode.state,
    commanded=episode.commanded,
    qpos=episode.qpos,
    qvel=episode.qvel,
    meta=np.array(json.dumps(episode.meta)),
  )
  return path


def load(path: Path) -> Episode:
  with np.load(path) as stored:
    return Episode(
      name=Path(path).stem,
      timestamp=stored["timestamp"],
      state=stored["state"],
      commanded=stored["commanded"],
      qpos=stored["qpos"],
      qvel=stored["qvel"],
      meta=json.loads(str(stored["meta"])),
    )


def listing(folder: Path) -> list[Path]:
  """Every episode in ``folder``, in name order."""
  return sorted(Path(folder).glob("*.npz"))


class Recorder:
  """One episode at a time: ``start``, ``extend`` with the rows of every step,
  then ``finish`` to take it or ``discard`` to drop it."""

  def __init__(self) -> None:
    self.name: str | None = None
    """The episode being recorded, ``None`` between episodes."""
    self._meta: dict[str, Any] = {}
    self._rows: list[tuple[Float64Array, ...]] = []

  def start(self, name: str, meta: dict[str, Any]) -> None:
    self.name, self._meta, self._rows = name, dict(meta), []

  def extend(
    self,
    timestamp: Float64Array,
    state: Float64Array,
    commanded: Float64Array,
    qpos: Float64Array,
    qvel: Float64Array,
  ) -> None:
    """Add rows, each argument with a leading axis over them. Between episodes
    nothing is kept."""
    if self.name is not None:
      self._rows.append(
        tuple(
          np.array(rows, np.float64)
          for rows in (timestamp, state, commanded, qpos, qvel)
        )
      )

  def __len__(self) -> int:
    return sum(len(rows[0]) for rows in self._rows)

  def finish(self, meta: dict[str, Any] | None = None) -> Episode:
    """The episode, ``meta`` added to what ``start`` was given, and the
    recorder emptied."""
    if self.name is None or not self._rows:
      raise ValueError("no episode is being recorded, or it holds no row")
    timestamp, state, commanded, qpos, qvel = (
      np.concatenate(column) for column in zip(*self._rows, strict=True)
    )
    out = Episode(
      self.name,
      timestamp - timestamp[0],
      state,
      commanded,
      qpos,
      qvel,
      {**self._meta, **(meta or {})},
    )
    self.discard()
    return out

  def discard(self) -> None:
    self.name, self._meta, self._rows = None, {}, []
