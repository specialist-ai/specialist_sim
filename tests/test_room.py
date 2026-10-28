"""The pick-and-place room: what stands in it, and what the env reads off it."""

import mujoco
import numpy as np
import pytest

from specialist_sim import gripper, rotation
from specialist_sim.camera import Camera
from specialist_sim.embodiment import Embodiment
from specialist_sim.gripper import SLOTS
from specialist_sim.pickplace import Cfg, PickPlaceEnv
from specialist_sim.pickplace.room import SCENE_VIEW
from specialist_sim.render import Renderer


def test_the_room_stands_the_hands_at_a_table_with_a_cube(
  embodiment: Embodiment,
) -> None:
  env = PickPlaceEnv(Cfg(embodiment=embodiment, image_size=32))
  room = env.room
  assert env.rig.slots == SLOTS
  assert room.cube.rest > room.table_top
  assert len(room.pads) > 0 and gripper.SHUT < room.pinch < gripper.OPEN
  assert 0.05 < np.linalg.norm(room.grasp) < 0.15
  data = mujoco.MjData(room.model)
  room.place_cube(data, np.array([0.6, 0.1, 0.4]))
  mujoco.mj_forward(room.model, data)
  assert (
    data.xpos[room.cube.body][:2] == pytest.approx([0.6, 0.1])
    and data.xpos[room.cube.body][2] > room.table_top
  )


def test_the_scene_view_watches_the_table() -> None:
  env = PickPlaceEnv(Cfg(image_size=32))
  env.reset([0])
  view = Renderer(env.rig.model, {"room": (0, SCENE_VIEW)}, 96).render(
    env.rig.datas[0]
  )["room"]
  assert view.image.shape == (96, 96, 3) and (view.image.sum(axis=2) > 0).mean() > 0.9
  struck = np.bincount(view.body[view.body >= 0].ravel()).argmax()
  assert env.rig.model.body(int(struck)).name == "table"


def test_the_surfaces_carry_their_maps() -> None:
  env = PickPlaceEnv(Cfg(image_size=32))
  model = env.rig.model
  mapped = {
    model.geom(part.geom).name
    for part in env.rig._renderer.parts
    if part.texture is not None
  }
  assert mapped == {"floor", "wall_left", "wall_far", "table"}


def test_the_fiducials_stand_in_the_room_only_when_asked() -> None:
  bare = PickPlaceEnv(Cfg(image_size=32))
  assert mujoco.mj_name2id(bare.rig.model, mujoco.mjtObj.mjOBJ_BODY, "tag_0") < 0
  env = PickPlaceEnv(Cfg(image_size=32, fiducials=True))
  env.reset([3])
  ahead = rotation.from_matrix(
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
  )
  toward = Camera(quat=tuple(ahead), pos=(0.0, 0.0, 0.0))
  view = Renderer(env.rig.model, {"origin": (0, toward)}, 96).render(env.rig.datas[0])[
    "origin"
  ]
  seen = {
    env.rig.model.body(int(body)).name for body in np.unique(view.body) if body >= 0
  }
  assert "tag_0" in seen
