"""The tracer draws the scene the way the lens model projects it."""

from typing import Any

import mujoco
import numpy as np
import pytest

from fixtures.scene import BLIND, Bare, look, overhead
from specialist_sim.camera.lens import Lens, project
from specialist_sim.camera.render import Renderer, parts
from specialist_sim.env import Env, Start
from specialist_sim.gripper import HANDS, HOUSING

Stood = tuple[Env[Bare], mujoco.MjData]


@pytest.fixture(scope="module")
def stood() -> Stood:
  """Both hands standing over the slab, and the room as it stands."""
  env = Env(Bare(), BLIND)
  standing = np.array(
    [
      [0.47, 0.16, -0.08, 1.0, 0.0, 0.0, 0.0, 0.3],
      [0.47, -0.16, -0.08, 0.0, 0.0, 0.0, 1.0, 0.3],
    ]
  )
  env.reset([0], start=Start(standing))
  return env, env.sim.get_data(0)


def _renderer(env: Env[Bare], size: int) -> Any:
  return Renderer(
    env.model, [(hand.lens_site, Lens(), hand.housing) for hand in env.hands], size
  )


def _lenses(stood: Stood, size: int) -> Any:
  """Both lenses' views of the room as it stands."""
  env, data = stood
  return _renderer(env, size).render(
    data.geom_xpos[None],
    data.geom_xmat[None],
    data.site_xpos[None],
    data.site_xmat[None],
    range(len(HANDS)),
  )


def test_a_lens_reports_which_body_each_pixel_struck(stood: Stood) -> None:
  model = stood[0].model
  images, bodies = _lenses(stood, 96)
  assert images.dtype == np.uint8 and images.shape == (1, 2, 96, 96, 3)
  for lens in range(len(HANDS)):
    assert (images[0, lens].sum(axis=2) > 0).mean() > 0.2
  assert not np.array_equal(images[0, 0], images[0, 1])
  drawn = np.zeros(96 * 96, bool)
  drawn[Lens().rays(96).pixel] = True
  outside = ~drawn.reshape(96, 96)
  assert images[0, 0][outside].sum() == 0 and (bodies[0, 0][outside] == -1).all()
  seen = [
    {model.body(int(body)).name for body in np.unique(bodies[0, lens]) if body >= 0}
    for lens in range(len(HANDS))
  ]
  own, other = HANDS
  assert f"{own}_gripper_base" in seen[0], "a lens sees the body it is mounted on"
  assert f"{other}_gripper_base" in seen[0], "a lens sees the other hand"
  assert "world" in seen[0], "a lens sees the slab"


def test_a_lens_cannot_see_the_part_it_looks_out_of(stood: Stood) -> None:
  model = stood[0].model
  renderer = _renderer(stood[0], 32)
  names = [model.geom(part.geom).name for part in renderer.parts]
  for hand, view in zip(HANDS, renderer.views, strict=True):
    seen = view.seen.numpy()
    hidden = {name for name, drawn in zip(names, seen, strict=True) if not drawn}
    housing = tuple(f"{hand}_gripper_{part}" for part in HOUSING)
    assert hidden and hidden == {name for name in names if name.startswith(housing)}


def test_a_body_lands_on_the_pixel_the_lens_projects_it_to(stood: Stood) -> None:
  """The render and the projection agree about where the room is."""
  env, data = stood
  model = env.model
  above = overhead((0.47, 0.0, 0.4), 300.0, 240)
  _, body = look(model, data, above, 240)
  for hand in env.hands:
    local = np.diag([1.0, -1.0, -1.0]) @ (data.xpos[hand.base] - np.asarray(above.pos))
    column, row = np.rint(project(local[None], above.params.scaled(240))[0]).astype(int)
    struck = model.body(int(body[row, column])).name
    assert struck.startswith(f"{hand.name}_gripper_"), struck
  assert model.body(int(body[5, 5])).name == "world", "the slab fills the corners"


def test_a_geom_is_shaded_by_its_material_unless_it_has_a_color_of_its_own() -> None:
  model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <asset>
    <material name="ink" rgba="0.078 0.078 0.078 1"/>
    <material name="chalk" rgba="0.867 0.867 0.867 1"/>
  </asset>
  <worldbody>
    <light dir="0 0 -1" diffuse="0.5 0.5 0.5"/>
    <light castshadow="false" dir="0 0 -1" ambient="0.5 0.5 0.5" diffuse="0.2 0.2 0.2"/>
    <geom name="inked" type="box" size="0.1 0.1 0.1" group="1" material="ink"/>
    <geom name="chalked" type="box" size="0.1 0.1 0.1" group="1" material="chalk"/>
    <geom name="painted" type="box" size="0.1 0.1 0.1" group="1" material="ink"
          rgba="0.9 0.4 0.1 1"/>
    <geom name="hidden" type="box" size="0.1 0.1 0.1" group="0" material="ink"/>
  </worldbody>
</mujoco>
""")
  shaded = {model.geom(part.geom).name: part.albedo for part in parts(model)}
  assert set(shaded) == {"inked", "chalked", "painted"}
  assert shaded["inked"] == pytest.approx([0.078] * 3, abs=1e-6)
  assert shaded["chalked"] == pytest.approx([0.867] * 3, abs=1e-6)
  assert shaded["painted"] == pytest.approx([0.9, 0.4, 0.1], abs=1e-6)
