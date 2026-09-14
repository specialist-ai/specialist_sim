"""The pick-and-place room: what stands in it, and what the task reads off it."""

import mujoco
import numpy as np
import pytest

from specialist_sim import desk, rotation
from specialist_sim.camera import Camera, Mei
from specialist_sim.embodiment import Embodiment
from specialist_sim.env import Cfg, Env
from specialist_sim.gripper import SLOTS
from specialist_sim.pickplace import PickPlace
from specialist_sim.render import Renderer

DOWN = rotation.from_matrix(
  np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
)
SCENE_VIEW = Camera(
  mei=Mei(fx=520.0, fy=520.0, cx=240.0, cy=240.0),
  width=480,
  pos=(desk.CENTER[0], desk.CENTER[1], desk.TOP + 0.7),
  quat=tuple(DOWN),
  mask=None,
)
"""A pinhole 0.7 m over the desk, looking down at it, for watching an episode."""


def test_the_room_stands_the_hands_at_a_desk_with_a_cube(
  embodiment: Embodiment,
) -> None:
  env = Env(PickPlace(), Cfg(embodiment=embodiment, image_size=32))
  room = env.task.room
  assert env.rig.slots == SLOTS
  assert room.cube.rest > room.table_top
  assert len(room.pads) > 0
  assert 0.05 < np.linalg.norm(room.grasp) < 0.15
  data = mujoco.MjData(room.model)
  room.place_cube(data, np.array([0.1, 0.2, 0.4]))
  mujoco.mj_forward(room.model, data)
  assert (
    data.xpos[room.cube.body][:2] == pytest.approx([0.1, 0.2])
    and data.xpos[room.cube.body][2] > room.table_top
  )


def test_the_scene_view_watches_the_desk() -> None:
  env = Env(PickPlace(), Cfg(image_size=32))
  env.reset([0])
  view = Renderer(env.rig.model, {"room": (0, SCENE_VIEW)}, 96).render(
    env.rig.datas[0]
  )["room"]
  assert view.image.shape == (96, 96, 3) and (view.image.sum(axis=2) > 0).mean() > 0.9
  struck = np.bincount(view.body[view.body >= 0].ravel()).argmax()
  assert env.rig.model.body(int(struck)).name == "room"
