"""The hand: how an embodiment stands it up, and what it touches."""

import mujoco
import numpy as np
import pytest

from fixtures.scene import Bare
from specialist_sim import gripper, rotation
from specialist_sim.embodiment import Embodiment, Floating
from specialist_sim.env import Cfg, Env
from specialist_sim.gripper import HANDS


def _stand(embodiment: type[Embodiment]) -> Env[Bare]:
  return Env(Bare(), Cfg(embodiment=embodiment, image_hands=()))


def test_a_hand_is_read_back_where_its_embodiment_put_it(
  embodiment: type[Embodiment],
) -> None:
  """A pose is the gripper's base tag's, whatever holds it up."""
  env = _stand(embodiment)
  model = env.model
  data = mujoco.MjData(env.model)
  states = np.array(
    [
      [0.47, 0.16, -0.08, *rotation.exp(np.array([0.3, -0.2, 0.5])), 0.3],
      [0.47, -0.16, -0.08, 0.0, 0.0, 0.0, 1.0, 0.1],
    ]
  )
  for index, (hand, state) in enumerate(zip(env.hands, states, strict=True)):
    env.embodiment.place(data.qpos, index, state[:7])
    data.qpos[list(hand.fingers)] = state[7], -state[7]
  mujoco.mj_kinematics(model, data)
  read = gripper.states(env.hands, data.qpos[None], data.xpos[None], data.xquat[None])[
    0
  ]
  assert read[:, :3] == pytest.approx(states[:, :3], abs=1e-6)
  assert np.abs(read[:, 3:7]) == pytest.approx(np.abs(states[:, 3:7]), abs=1e-6)
  assert read[:, 7] == pytest.approx(states[:, 7])


def test_the_two_hands_collide_with_each_other() -> None:
  env = _stand(Floating)
  model = env.model
  data = mujoco.MjData(env.model)
  for index in range(2):
    env.embodiment.place(
      data.qpos, index, np.array([0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    )
  mujoco.mj_forward(model, data)
  rigs = {model.body(f"{hand}_gripper_rig").id for hand in HANDS}
  roots = model.body_rootid[model.geom_bodyid]
  between = [
    contact
    for contact in data.contact
    if {roots[contact.geom[0]], roots[contact.geom[1]]} == rigs
  ]
  assert between, "one gripper passes through the other"
