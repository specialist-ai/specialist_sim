"""The tracer draws the world the way the lens model projects it."""

from typing import Any

import mujoco
import numpy as np
import pytest

from specialist_sim import rotation
from specialist_sim.camera import NOMINAL, Camera, Mei, project
from specialist_sim.embodiment import Floating
from specialist_sim.gripper import SLOTS, Hand
from specialist_sim.render import HOUSING, Renderer, parts
from world import bare

STANDING = np.array(
  [
    [0.47, 0.16, -0.08, 1.0, 0.0, 0.0, 0.0, 0.3],
    [0.47, -0.16, -0.08, 0.0, 0.0, 0.0, 1.0, 0.3],
  ]
)
"""Both hands over the slab, facing each other."""


@pytest.fixture(scope="module")
def stood() -> tuple[mujoco.MjModel, tuple[Hand, ...], mujoco.MjData]:
  floating = Floating()
  spec = bare()
  floating.attach(spec)
  model = spec.compile()
  hands = floating.bind(model)
  data = mujoco.MjData(model)
  for index, (hand, state) in enumerate(zip(hands, STANDING, strict=True)):
    floating.place(data, index, state[:7])
    data.qpos[list(hand.fingers)] = state[7], -state[7]
  mujoco.mj_forward(model, data)
  return model, hands, data


def _lenses(stood: Any, size: int) -> Any:
  model, hands, data = stood
  cameras = {hand.slot: (hand.base, NOMINAL) for hand in hands}
  return Renderer(model, cameras, size).render(data)


def test_each_lens_returns_a_lit_frame_of_its_own(stood: Any) -> None:
  views = _lenses(stood, 64)
  for slot in SLOTS:
    assert views[slot].image.dtype == np.uint8 and views[slot].image.shape == (
      64,
      64,
      3,
    )
    assert (views[slot].image.sum(axis=2) > 0).mean() > 0.2
  assert not np.array_equal(views[SLOTS[0]].image, views[SLOTS[1]].image)
  drawn = np.zeros(64 * 64, bool)
  drawn[NOMINAL.rays(64).pixel] = True
  outside = ~drawn.reshape(64, 64)
  assert views[SLOTS[0]].image[outside].sum() == 0
  assert (views[SLOTS[0]].body[outside] == -1).all()


def test_a_lens_reports_which_body_each_pixel_struck(stood: Any) -> None:
  model = stood[0]
  views = _lenses(stood, 96)
  seen = {
    slot: {
      model.body(int(body)).name for body in np.unique(views[slot].body) if body >= 0
    }
    for slot in SLOTS
  }
  own, other = SLOTS
  assert f"{own}_base" in seen[own], "a lens sees the body it is mounted on"
  assert f"{other}_base" in seen[own], "a lens sees the other hand"
  assert "world" in seen[own], "a lens sees the slab"


def test_a_lens_cannot_see_the_part_it_looks_out_of(stood: Any) -> None:
  model, hands, _ = stood
  renderer = Renderer(model, {hand.slot: (hand.base, NOMINAL) for hand in hands}, 32)
  names = [model.geom(part.geom).name for part in renderer.parts]
  for slot in SLOTS:
    seen = renderer.lenses[slot].seen.numpy()
    hidden = {name for name, drawn in zip(names, seen, strict=True) if not drawn}
    housing = tuple(f"{slot}_{part}" for part in HOUSING)
    assert hidden and hidden == {name for name in names if name.startswith(housing)}


def test_a_body_lands_on_the_pixel_the_lens_projects_it_to(stood: Any) -> None:
  """The render and the projection agree about where the world is."""
  model, hands, data = stood
  down = rotation.from_matrix(
    np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
  )
  above = Camera(
    mei=Mei(fx=300.0, fy=300.0, cx=120.0, cy=120.0),
    width=240,
    pos=(0.47, 0.0, 0.4),
    quat=tuple(down),
    mask=None,
  )
  view = Renderer(model, {"above": (0, above)}, 240).render(data)["above"]
  lens = rotation.to_matrix(np.asarray(down))
  for hand in hands:
    local = lens.T @ (data.xpos[hand.base] - np.asarray(above.pos))
    column, row = np.rint(project(local[None], above.mei)[0]).astype(int)
    struck = model.body(int(view.body[row, column])).name
    assert struck.startswith(hand.slot), struck
  assert model.body(int(view.body[5, 5])).name == "world", "the slab fills the corners"


MATERIALS = """
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
"""


def test_a_geom_is_shaded_by_its_material_unless_it_has_a_color_of_its_own() -> None:
  model = mujoco.MjModel.from_xml_string(MATERIALS)
  shaded = {model.geom(part.geom).name: part.albedo for part in parts(model)}
  assert set(shaded) == {"inked", "chalked", "painted"}
  assert shaded["inked"] == pytest.approx([0.078] * 3, abs=1e-6)
  assert shaded["chalked"] == pytest.approx([0.867] * 3, abs=1e-6)
  assert shaded["painted"] == pytest.approx([0.9, 0.4, 0.1], abs=1e-6)


CHECKED = """
<mujoco>
  <asset>
    <texture name="check" type="2d" builtin="checker" rgb1="1 1 1" rgb2="0 0 0"
             width="64" height="64"/>
    <material name="checked" texture="check" texrepeat="1 1"/>
  </asset>
  <worldbody>
    <light dir="0 0 -1" diffuse="0.5 0.5 0.5"/>
    <light castshadow="false" dir="0 0 -1" ambient="0.5 0.5 0.5" diffuse="0.2 0.2 0.2"/>
    <geom name="board" type="box" pos="0 0 -1" size="0.5 0.5 0.05" group="1" material="checked"/>
  </worldbody>
</mujoco>
"""


def test_a_textured_box_reads_its_map() -> None:
  model = mujoco.MjModel.from_xml_string(CHECKED)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  (board,) = parts(model)
  assert board.texture is not None and board.texture.shape == (64, 64, 3)
  down = rotation.from_matrix(
    np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
  )
  above = Camera(
    mei=Mei(fx=100.0, fy=100.0, cx=64.0, cy=64.0),
    width=128,
    pos=(0.0, 0.0, 0.0),
    quat=tuple(down),
    mask=None,
  )
  view = Renderer(model, {"above": (0, above)}, 128).render(data)["above"]
  struck = view.body == 0
  assert 0.4 < struck.mean() < 0.9
  bright = view.image[struck].max(axis=1)
  black, white = bright < 10, bright > 100
  assert 0.35 < black.mean() < 0.65, "the checker is half black"
  assert (black | white).mean() > 0.9, (
    "the checker is two colours, blended at the edges"
  )


def test_a_world_without_lights_is_lit_by_nothing_but_its_ambient() -> None:
  dark = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type="box" size="1 1 0.1" pos="0 0 -1" group="1"/></worldbody></mujoco>'
  )
  data = mujoco.MjData(dark)
  mujoco.mj_forward(dark, data)
  down = rotation.from_matrix(
    np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
  )
  above = Camera(
    mei=Mei(fx=50.0, fy=50.0, cx=16.0, cy=16.0),
    width=32,
    pos=(0.0, 0.0, 0.0),
    quat=tuple(down),
    mask=None,
  )
  view = Renderer(dark, {"above": (0, above)}, 32).render(data)["above"]
  assert (view.body == 0).all() and view.image.sum() == 0
