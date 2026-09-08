"""The patch of table a dataset picks its cubes from, read from the cube's rows.

A grasp stands where the cube stood when the fingers shut on it, so the region
a policy was shown is the ring around every closing in the dataset it learned
from, with the hole around the tape those closings keep clear of. Derived at
evaluation time — dataset in, ``Region`` out — and deterministic: rows are
ordered by session and step before anything is reduced, so any process
derives the same region from the same dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.randomise import Region

__all__ = [
  "CUBE",
  "ENGAGED",
  "HAND",
  "OPENING",
  "closings",
  "derive",
  "opened",
  "region",
  "ring",
]

ENGAGED = 0.15
"""The opening below which the fingers have closed on the cube: pads meet a
30 mm cube near +0.02 rad and each other at -0.16, so a crossing here is a
grasp under way rather than a hand travelling open."""

STEP = "step"
"""The dataset's step lattice, which orders a session's rows."""

CUBE = "/state/cube/pose:Transform3D:translation"
OPENING = "/state/{hand}/opening:Scalars:scalars"
"""The catalog columns a region is read from, as the state layer writes them."""

HAND = "right"
"""The hand whose closings define where a pick begins."""


def region(
  catalog_url: str, dataset: str, trim: float = 99.0, nearest: float = 5.0
) -> Region:
  """The region `dataset`'s picks begin in, read from its catalog.

  The dataset is named, never defaulted: the region belongs to the dataset a
  policy was trained on, and evaluating against any other is a decision. So
  are the two percentiles. `trim` drops the closings farthest from the median
  before the hull, so one stray grasp does not stretch the ring; `nearest` is
  the percentile of the closings' distances from the tape that becomes the
  hole's radius — the 5th matches the dataset's density, the 1st admits its
  rarest picks.
  """
  cube, opening, segment = _rows(catalog_url, dataset)
  return derive(closings(cube, opening, segment), opened(cube, segment), trim, nearest)


def derive(
  stood: Float64Array, tape: Float64Array, trim: float, nearest: float
) -> Region:
  """The ring around the closings and the hole around the tape, from the
  closings alone. The ring holds at least 98 of every 100 closings."""
  if len(stood) < 4:
    raise ValueError(f"a region needs closings to stand in, and there are {len(stood)}")
  stray = np.linalg.norm(stood - np.median(stood, axis=0), axis=1)
  kept = stood[stray <= np.percentile(stray, trim)]
  away = np.linalg.norm(stood - tape, axis=1)
  return Region(
    vertices=tuple(map(tuple, ring(kept))),
    hole=(float(tape[0]), float(tape[1]), float(np.percentile(away, nearest))),
  )


def closings(
  cube: Float64Array, opening: Float64Array, segment: np.ndarray
) -> Float64Array:
  """Where the cube stood at every closing, one row each.

  A closing is the row where the fingers cross from open to shut, and a row is
  only that if the row before it belongs to the same session.
  """
  shut = opening < ENGAGED
  crossing = np.zeros(len(opening), bool)
  crossing[1:] = shut[1:] & ~shut[:-1] & (segment[1:] == segment[:-1])
  crossing &= np.isfinite(cube).all(axis=1)
  return cube[crossing][:, :2]


def opened(cube: Float64Array, segment: np.ndarray) -> Float64Array:
  """Where the cube stands as sessions open: the tape, as the median of every
  session's first row."""
  first = np.ones(len(segment), bool)
  first[1:] = segment[1:] != segment[:-1]
  first &= np.isfinite(cube).all(axis=1)
  out: Float64Array = np.median(cube[first][:, :2], axis=0)
  return out


def ring(points: Float64Array) -> Float64Array:
  """The counter-clockwise convex hull around `points`, by monotone chain."""
  order = points[np.lexsort((points[:, 1], points[:, 0]))]

  def half(seq: Float64Array) -> list[Float64Array]:
    out: list[Float64Array] = []
    for point in seq:
      while len(out) > 1:
        a, b = out[-2], out[-1]
        if (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) > 0:
          break
        out.pop()
      out.append(point)
    return out

  return np.array(half(order)[:-1] + half(order[::-1])[:-1])


def _rows(
  catalog_url: str, dataset: str
) -> tuple[Float64Array, Float64Array, np.ndarray]:
  """Every row's cube position, right opening and session, ordered by session
  and step. The catalog returns rows in whatever order it scanned them."""
  from datafusion import col
  from rerun.catalog import CatalogClient

  # The client and the dataset must outlive the frame, whose context they hold.
  client = CatalogClient(catalog_url)
  entry = client.get_dataset(dataset)
  frame = entry.reader(index=STEP)
  names = frame.schema().names
  place = next(name for name in names if "cube" in name and "translation" in name)
  grip = next(name for name in names if "right" in name and "opening" in name)
  table = frame.select(
    col("rerun_segment_id"), col(STEP), col(f'"{place}"'), col(f'"{grip}"')
  ).to_arrow_table()
  segment = table.column(0).to_numpy(zero_copy_only=False)
  step = table.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
  order = np.lexsort((step, segment))
  return (
    _numbers(table.column(2), 3)[order],
    _numbers(table.column(3), 1)[order, 0],
    segment[order],
  )


def _numbers(column: Any, width: int) -> Float64Array:
  """``(rows, width)`` from a column of one-entry lists, read through the
  buffers: a row with no entry reads as not a number."""
  import pyarrow as pa

  outer = column.combine_chunks()
  offsets = np.asarray(outer.offsets, np.int64)
  inner = outer.flatten()
  if pa.types.is_fixed_size_list(inner.type):
    inner = inner.flatten()
  values = np.asarray(inner.to_numpy(zero_copy_only=False), np.float64).reshape(
    -1, width
  )
  out = np.full((len(outer), width), np.nan)
  present = (offsets[1:] - offsets[:-1]) > 0
  out[present] = values[offsets[:-1][present]]
  return out
