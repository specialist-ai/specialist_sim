"""The far background: which way its faces look, and what a missed ray reads."""

import mujoco
import numpy as np
import pytest

from sampler import read
from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.calibration import CameraCfg
from specialist_sim.embodiment import ASSETS
from specialist_sim.render import sky
from specialist_sim.render.renderer import Renderer
from specialist_sim.spaces import GRIPPERS

EMPTY = "<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>"


def test_each_face_is_read_along_the_way_it_looks() -> None:
  """Six flat faces, sampled straight down each one's center."""
  faces = np.zeros((len(sky.ORDER), 4, 4, 3), np.float32)
  for index in range(len(sky.ORDER)):
    faces[index] = (index + 1.0) / len(sky.ORDER)
  centers = [sky.FACES[key][0] for key in sky.ORDER]
  for index, got in enumerate(read(faces, centers)):
    assert got == pytest.approx([(index + 1.0) / len(sky.ORDER)] * 3, abs=1e-6)


def test_a_face_is_read_across_its_columns_and_down_its_rows() -> None:
  """Half way to a face's edge is three quarters of the way across it.

  Each ramp reads half a texel short of its corner because the sampler lands
  between texel centers.
  """
  size = 8
  faces = np.zeros((len(sky.ORDER), size, size, 3), np.float32)
  ramp = np.arange(size, dtype=np.float32) / size
  right = sky.ORDER.index("R")
  faces[right, :, :, 0] = ramp
  faces[right, :, :, 1] = ramp[:, None]
  center, across, down = (np.array(axis, np.float64) for axis in sky.FACES["R"])

  column, row = read(faces, [center + 0.5 * across + 0.25 * down])[0][:2]
  assert column == pytest.approx(0.75 - 0.5 / size, abs=1e-6)
  assert row == pytest.approx(0.625 - 0.5 / size, abs=1e-6)


def test_a_scene_carrying_no_skybox_is_refused() -> None:
  with pytest.raises(ValueError, match="skybox"):
    sky.read(mujoco.MjModel.from_xml_string(EMPTY))


def test_the_room_background_has_no_viewpoint_locked_features() -> None:
  faces = sky.read(mujoco.MjModel.from_xml_path(str(ASSETS / "room.xml")))
  assert np.ptp(faces, axis=(0, 1, 2)).max() == pytest.approx(0.0)
  assert faces.mean() > 0.0, "a missed ray is distinguishable from the lens mask"


def test_the_lights_shade_the_geoms_and_never_the_background() -> None:
  """A missed ray carries the background's fixed exposure.

  Dimming the scene moves every lit pixel and no part of the room, which is the
  whole of what it means for the sky to arrive already exposed.
  """
  size = 64
  slot = GRIPPERS[0]
  env = PickPlaceEnv(PickPlaceCfg(image_size=size))
  env.reset(seed=3)
  model, data = env.scene.model, env.scene.data

  bright = Renderer(model, {slot: CameraCfg()}, size).render(data)[slot]
  model.light_diffuse[:] *= 0.5
  dim = Renderer(model, {slot: CameraCfg()}, size).render(data)[slot]

  room, lit = bright.body < 0, bright.body >= 0
  assert (bright.image[room].sum(axis=-1) > 0).any(), "the room is not blackness"
  assert np.array_equal(bright.image[room], dim.image[room])
  assert not np.array_equal(bright.image[lit], dim.image[lit])
