"""A recorded episode on disk, ``<folder>/<name>.npz``, every row of it.

Row ``k`` is the env at ``timestamp[k]``, seconds from the episode's first row:

* ``state``: the hands as they stood, ``(rows, hands, 8)``.
* ``setpoint``: the setpoint each hand was chasing, ``(rows, hands, 8)``.
* ``qpos``: the whole env, which images are traced from.
* ``meta``: JSON, ``Env.describe()`` beside the recorder's own keys.
  ``env.make`` stands the same env again from it.

A hand is ``[position(3), wxyz(4), opening(1)]`` in the room frame, in
``gripper.HANDS`` order.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from specialist_sim.arrays import Float64Array


@dataclass(frozen=True)
class Rows:
  timestamp: Float64Array
  state: Float64Array
  setpoint: Float64Array
  qpos: Float64Array

  def __getitem__(self, rows: slice) -> Rows:
    return Rows(*(getattr(self, column.name)[rows] for column in fields(self)))


@dataclass(frozen=True)
class Episode:
  name: str
  rows: Rows
  meta: dict[str, Any]


def joined(rows: Sequence[Rows]) -> Rows:
  """Rows end to end, timed from the first."""
  timestamp, *rest = (
    np.concatenate([getattr(part, column.name) for part in rows])
    for column in fields(Rows)
  )
  return Rows(timestamp - timestamp[0], *rest)


def save(folder: Path, episode: Episode) -> Path:
  folder.mkdir(parents=True, exist_ok=True)
  path = folder / f"{episode.name}.npz"
  np.savez_compressed(
    path,
    **{column.name: getattr(episode.rows, column.name) for column in fields(Rows)},
    meta=np.array(json.dumps(episode.meta)),
  )
  return path


def load(path: Path) -> Episode:
  with np.load(path) as stored:
    return Episode(
      name=path.stem,
      rows=Rows(*(stored[column.name] for column in fields(Rows))),
      meta=json.loads(str(stored["meta"])),
    )
