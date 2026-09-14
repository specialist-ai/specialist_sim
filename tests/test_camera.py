"""The MEI lens: projection, its inverse, and the rays a frame draws."""

from dataclasses import replace

import numpy as np
import pytest

from specialist_sim.camera import NOMINAL, Mei, project, unproject


def test_a_bearing_survives_the_projection_round_trip() -> None:
  mei = NOMINAL.mei.scaled(240 / NOMINAL.width)
  rays = NOMINAL.rays(240)
  pixels = project(rays.bearings.astype(np.float64), mei)
  assert np.abs(unproject(pixels, mei) - rays.bearings).max() < 1e-6


def test_the_lens_matches_a_pinhole_when_the_mirror_is_flat() -> None:
  flat = Mei(fx=NOMINAL.mei.fx, fy=NOMINAL.mei.fy, cx=NOMINAL.mei.cx, cy=NOMINAL.mei.cy)
  points = np.array([[0.0, 0.0, 1.0], [0.3, -0.2, 2.0], [-0.1, 0.4, 0.7]])
  wanted = np.stack(
    [
      flat.fx * points[:, 0] / points[:, 2] + flat.cx,
      flat.fy * points[:, 1] / points[:, 2] + flat.cy,
    ],
    axis=1,
  )
  assert project(points, flat) == pytest.approx(wanted, abs=1e-9)


def test_a_point_behind_the_lens_has_no_pixel() -> None:
  behind = np.array([[np.sin(a), 0.0, np.cos(a)] for a in np.radians([150.0, 180.0])])
  assert np.isnan(project(behind, NOMINAL.mei)).all()


def test_the_same_lens_at_twice_the_width_doubles_its_pixels() -> None:
  near, far = (
    NOMINAL.mei.scaled(240 / NOMINAL.width),
    NOMINAL.mei.scaled(480 / NOMINAL.width),
  )
  for coarse, fine in ((near.fx, far.fx), (near.fy, far.fy), (near.cx, far.cx)):
    assert fine == pytest.approx(2.0 * coarse)
  assert far.xi == near.xi and far.k1 == near.k1


def test_a_frame_draws_the_disc_the_mask_keeps() -> None:
  size = 64
  drawn = np.zeros((size, size), bool)
  drawn.ravel()[NOMINAL.rays(size).pixel] = True
  rows, columns = np.indices((size, size))
  inside = np.hypot(rows + 0.5 - size / 2, columns + 0.5 - size / 2) <= size / 2
  assert np.array_equal(drawn, inside)
  whole = replace(NOMINAL, mask=None).rays(size)
  assert len(whole.pixel) > len(NOMINAL.rays(size).pixel)
