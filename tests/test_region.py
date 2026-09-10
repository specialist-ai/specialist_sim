"""The start region a corpus's closings define, from the rows alone."""

import numpy as np
import pytest

from specialist_sim.corpus.region import closings, derive, opened
from specialist_sim.randomise import Region


def test_the_ring_around_a_corpus_holds_the_closings_it_was_derived_from() -> None:
  """One stray grasp is trimmed before the hull; everything ordinary stays."""
  rng = np.random.default_rng(0)
  sessions, rows = 200, 50
  segment = np.repeat(np.arange(sessions), rows)
  opening = np.tile(np.linspace(0.6, -0.1, rows), sessions)
  cube = np.zeros((sessions * rows, 3))
  patch = np.stack(
    [rng.uniform(0.40, 0.55, sessions), rng.uniform(-0.08, 0.08, sessions)], axis=1
  )
  patch[0] = (2.0, 2.0)
  cube[:, :2] = np.repeat(patch, rows, axis=0)
  stood = closings(cube, opening, segment)
  assert len(stood) == sessions, "one closing per session, none across a boundary"
  hull = derive(stood, np.array([0.49, 0.0]), trim=99.0, nearest=5.0)
  ring_only = Region(vertices=hull.vertices)
  inside = sum(ring_only.contains(spot) or _on_ring(hull, spot) for spot in stood)
  assert inside >= 0.98 * len(stood)
  assert not ring_only.contains(np.array([2.0, 2.0])), (
    "the planted stray grasp stays outside"
  )


def test_the_hole_keeps_every_draw_off_the_table_the_corpus_never_picks_from() -> None:
  rng = np.random.default_rng(1)
  sessions, rows, tape = 300, 4, np.array([0.49, 0.0])
  segment = np.repeat(np.arange(sessions), rows)
  opening = np.tile(np.array([0.6, 0.6, 0.3, 0.1]), sessions)
  bearing = rng.uniform(-np.pi, np.pi, sessions)
  away = rng.uniform(0.086, 0.105, sessions)
  cube = np.zeros((sessions * rows, 3))
  cube[:, :2] = np.repeat(
    tape + np.stack([away * np.cos(bearing), away * np.sin(bearing)], 1), rows, 0
  )
  cube[0::rows, :2] = tape
  assert np.allclose(opened(cube, segment), tape)
  region = derive(closings(cube, opening, segment), opened(cube, segment), 99.0, 5.0)
  assert region.hole is not None and 0.085 <= region.hole[2] <= 0.09
  for seed in range(100):
    spot = region.draw(np.random.default_rng(seed))
    assert np.hypot(*(spot - tape)) >= region.hole[2]


def test_a_closing_needs_the_row_before_it_in_the_same_session() -> None:
  opening = np.array([0.5, 0.1, 0.5, 0.1])
  cube = np.zeros((4, 3))
  assert len(closings(cube, opening, np.array([0, 0, 1, 1]))) == 2
  assert len(closings(cube, opening, np.array([0, 1, 2, 3]))) == 0


def test_the_derivation_is_the_same_whatever_order_the_rows_arrive_in() -> None:
  rng = np.random.default_rng(2)
  sessions, rows = 50, 6
  segment = np.repeat(np.arange(sessions), rows)
  opening = np.tile(np.array([0.6, 0.5, 0.4, 0.2, 0.1, 0.0]), sessions)
  cube = np.zeros((sessions * rows, 3))
  cube[:, :2] = np.repeat(rng.uniform(0.35, 0.6, (sessions, 2)), rows, 0)
  cube[0::rows, :2] = (0.49, 0.0)
  row = np.tile(np.arange(rows), sessions)
  first = derive(closings(cube, opening, segment), opened(cube, segment), 99.0, 5.0)
  shuffled = rng.permutation(len(segment))
  order = np.lexsort((row[shuffled], segment[shuffled]))
  back = shuffled[order]
  again = derive(
    closings(cube[back], opening[back], segment[back]),
    opened(cube[back], segment[back]),
    99.0,
    5.0,
  )
  assert first == again


def _on_ring(region: Region, spot: np.ndarray) -> bool:
  return any(np.allclose(spot, vertex) for vertex in region.vertices)


def test_a_region_refuses_too_few_closings() -> None:
  with pytest.raises(ValueError, match="closings"):
    derive(
      np.array([[0.4, 0.0], [0.5, 0.0], [0.45, 0.05]]), np.array([0.49, 0.0]), 99.0, 5.0
    )
