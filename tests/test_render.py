from dataclasses import replace

import numpy as np
import pytest

from specialist_sim.calibration import CameraCfg
from specialist_sim.mei import mei_project, mei_unproject
from specialist_sim.rays import build_rays


def test_a_bearing_survives_the_projection_round_trip() -> None:
  params = CameraCfg().at(240)
  rays = build_rays(params, 240, 240, CameraCfg().mask_radius)
  pixels = mei_project(rays.bearings.astype(np.float64), params)
  again = mei_unproject(pixels, params)
  assert np.abs(again - rays.bearings).max() < 1e-6


def test_the_lens_matches_a_pinhole_when_the_mirror_is_flat() -> None:
  """With the mirror at zero the model is a plain pinhole, which is a closed form."""
  flat = replace(CameraCfg().params, xi=0.0, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0)
  points = np.array([[0.0, 0.0, 1.0], [0.3, -0.2, 2.0], [-0.1, 0.4, 0.7]])
  wanted = np.stack(
    [
      flat.fx * points[:, 0] / points[:, 2] + flat.cx,
      flat.fy * points[:, 1] / points[:, 2] + flat.cy,
    ],
    axis=1,
  )
  assert mei_project(points, flat) == pytest.approx(wanted, abs=1e-9)


def test_a_point_behind_the_lens_has_no_pixel() -> None:
  behind = np.array([[np.sin(a), 0.0, np.cos(a)] for a in np.radians([150.0, 180.0])])
  assert np.isnan(mei_project(behind, CameraCfg().params)).all()


def test_the_same_lens_at_twice_the_width_doubles_its_pixels() -> None:
  near, far = CameraCfg().at(240), CameraCfg().at(480)
  for coarse, fine in ((near.fx, far.fx), (near.fy, far.fy), (near.cx, far.cx)):
    assert fine == pytest.approx(2.0 * coarse)
  assert far.xi == near.xi and far.k1 == near.k1
