"""Each camera sees the room, the other hand, and not its own housing."""

from dataclasses import replace
from typing import Any

import mujoco
import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.calibration import CameraCfg
from specialist_sim.mei import mei_project, mei_unproject
from specialist_sim.rays import build_rays
from specialist_sim.spaces import GRIPPERS


@pytest.fixture(scope="module")
def observation() -> dict[str, Any]:
  return PickPlaceEnv().reset(seed=3)[0]


def test_every_camera_returns_a_lit_image(observation: dict[str, Any]) -> None:
  for name in GRIPPERS:
    image = observation[name]["image"]
    assert image.dtype == np.uint8
    assert (image.sum(axis=2) > 0).mean() > 0.2


def test_the_two_cameras_see_different_things(observation: dict[str, Any]) -> None:
  assert not np.array_equal(
    observation[GRIPPERS[0]]["image"], observation[GRIPPERS[1]]["image"]
  )


def test_nothing_is_drawn_outside_the_lens_disc(observation: dict[str, Any]) -> None:
  size = observation[GRIPPERS[0]]["image"].shape[0]
  rays = build_rays(CameraCfg().at(size), size, size, CameraCfg().mask_radius)
  assert observation[GRIPPERS[0]]["image"][~rays.valid].sum() == 0


def test_a_bearing_survives_the_projection_round_trip() -> None:
  params = CameraCfg().at(240)
  rays = build_rays(params, 240, 240, CameraCfg().mask_radius)
  pixels = mei_project(rays.bearings.astype(np.float64), params)
  again = mei_unproject(pixels, params)
  assert np.abs(again - rays.bearings).max() < 1e-6


def test_a_camera_reports_which_body_each_pixel_struck() -> None:
  """Each pixel's body id names what it struck."""
  env = PickPlaceEnv()
  env.reset(seed=3)
  model = env.scene.model
  seen = {
    slot: {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body))
      for body in np.unique(view.body)
      if body >= 0
    }
    for slot, view in env.views.items()
  }
  own, other = GRIPPERS
  assert f"{own}_base" not in seen[own]
  assert f"{own}_base" in seen[other]
  assert {"table", "cube"} <= seen[own]


def test_the_env_renders_through_the_camera_it_is_given() -> None:
  """A supplied calibration sets both where the lens sits and how it projects."""
  camera = CameraCfg(
    pos=(0.0, 0.0, 0.25), params=replace(CameraCfg().params, fx=4000.0, fy=4000.0)
  )
  env = PickPlaceEnv(PickPlaceCfg(cameras=dict.fromkeys(GRIPPERS, camera)))
  moved = env.reset(seed=3)[0]

  lens = env.scene.model.body(f"{GRIPPERS[0]}_lens")
  assert env.scene.model.body_pos[lens.id] == pytest.approx(camera.pos)
  plain = PickPlaceEnv().reset(seed=3)[0]
  assert not np.array_equal(moved[GRIPPERS[0]]["image"], plain[GRIPPERS[0]]["image"])


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


def test_a_body_lands_on_the_pixel_the_lens_projects_it_to() -> None:
  """The render and the projection agree about where the room is."""
  env = PickPlaceEnv()
  # The cube rests on the tape, which it would otherwise stand in front of.
  env.reset(seed=3, options={"cube": (0.60, -0.10)})
  scene = env.scene
  lens = scene.model.body(f"{GRIPPERS[0]}_lens").id
  rotation = scene.data.xmat[lens].reshape(3, 3)
  for name in ("table", "tape"):
    body = scene.model.body(name).id
    local = rotation.T @ (scene.data.xpos[body] - scene.data.xpos[lens])
    pixel = mei_project(local.reshape(1, 3), CameraCfg().at(env.cfg.image_size))[0]
    column, row = int(round(pixel[0])), int(round(pixel[1]))
    struck = env.views[GRIPPERS[0]].body[row, column]
    assert mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, int(struck)) == name
