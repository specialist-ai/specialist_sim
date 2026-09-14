"""The patch of table a dataset picks its cubes from: the ring around every
closing, less the hole around the tape they keep clear of.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
from rerun.catalog import CatalogClient
from specialist_corpus import schema

from specialist_sim.arrays import Float64Array
from specialist_sim.dataset.layer import CUBE_POSE
from specialist_sim.pickplace.task import Region

__all__ = ["ENGAGED", "closings", "derive", "opened", "region", "ring"]

ENGAGED = 0.15
"""The opening below which the fingers have closed on the cube."""

SEGMENT = "rerun_segment_id"
"""The catalog's column naming the session a row belongs to."""


def region(
  catalog_url: str, dataset: str, trim: float = 99.0, nearest: float = 5.0
) -> Region:
  """The region ``dataset``'s picks begin in, read from its catalog. ``trim``
  is the percentile of closings kept for the ring, ``nearest`` the percentile
  of their distances from the tape that is the hole's radius."""
  cube, opening, segment = _rows(catalog_url, dataset)
  return derive(closings(cube, opening, segment), opened(cube, segment), trim, nearest)


def derive(
  stood: Float64Array, tape: Float64Array, trim: float, nearest: float
) -> Region:
  """The ring around the closings and the hole around the tape."""
  stray = np.linalg.norm(stood - np.median(stood, axis=0), axis=1)
  kept = stood[stray <= np.percentile(stray, trim)]
  away = np.linalg.norm(stood - tape, axis=1)
  return Region(
    vertices=tuple((float(x), float(y)) for x, y in ring(kept)),
    hole=(float(tape[0]), float(tape[1]), float(np.percentile(away, nearest))),
  )


def closings(
  cube: Float64Array, opening: Float64Array, segment: np.ndarray
) -> Float64Array:
  """Where the cube stood at every closing: the rows where the fingers cross
  from open to shut, the row before belonging to the same session."""
  shut = opening < ENGAGED
  crossing = np.zeros(len(opening), bool)
  crossing[1:] = shut[1:] & ~shut[:-1] & (segment[1:] == segment[:-1])
  crossing &= np.isfinite(cube).all(axis=1)
  return cube[crossing][:, :2]


def opened(cube: Float64Array, segment: np.ndarray) -> Float64Array:
  """Where the cube stands as sessions open: the median of every session's
  first row."""
  first = np.ones(len(segment), bool)
  first[1:] = segment[1:] != segment[:-1]
  first &= np.isfinite(cube).all(axis=1)
  out: Float64Array = np.median(cube[first][:, :2], axis=0)
  return out


def ring(points: Float64Array) -> Float64Array:
  """The counter-clockwise convex hull around ``points``, by monotone chain."""
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
  and row."""
  opening = schema.state_opening("right")
  # The client and the dataset hold the reader's context until the table is read.
  client = CatalogClient(catalog_url)
  entry = client.get_dataset(dataset)
  view = entry.filter_contents([f"/{CUBE_POSE}", f"/{opening}"])
  table = (
    view.reader(index=schema.ROW_TIMELINE)
    .select(
      SEGMENT,
      schema.ROW_TIMELINE,
      schema.selector(CUBE_POSE, schema.POSE_TRANSLATION),
      schema.selector(opening, schema.SCALARS),
    )
    .to_arrow_table()
  )
  segment = table.column(SEGMENT).to_numpy(zero_copy_only=False)
  row = (
    table.column(schema.ROW_TIMELINE).to_numpy(zero_copy_only=False).astype(np.int64)
  )
  order = np.lexsort((row, segment))
  return (
    _numbers(table.column(schema.selector(CUBE_POSE, schema.POSE_TRANSLATION)), 3)[
      order
    ],
    _numbers(table.column(schema.selector(opening, schema.SCALARS)), 1)[order, 0],
    segment[order],
  )


def _numbers(column: pa.ChunkedArray, width: int) -> Float64Array:
  """``(rows, width)`` from a column of one-entry lists; a row with no entry
  reads as not a number."""
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
