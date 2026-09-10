"""A task names the hands a policy drives; the rig stays whole underneath."""

from dataclasses import replace

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.action import hold
from specialist_sim.embodiment import EmbodimentCfg, FloatingCfg, FreeHand, carried
from specialist_sim.observation import flat_row
from specialist_sim.randomise import Kicked, OnTape, Region
from specialist_sim.spaces import GRIPPERS, HORIZON, PER_HAND
from specialist_sim.task import (
  BIMANUAL,
  PlacedOnTape,
  RestingIn,
  Task,
  left_task,
  right_task,
)

PATCH = Region(vertices=((0.30, -0.14), (0.62, -0.14), (0.62, 0.10), (0.30, 0.10)))
"""A patch of table, off the tape on the right hand's side and beyond."""
SMALL = PickPlaceCfg(image_size=64)


def _env(task: Task, embodiment: EmbodimentCfg) -> PickPlaceEnv:
  return PickPlaceEnv(replace(SMALL, task=task, embodiment=embodiment))


def test_a_one_hand_task_reproduces_the_two_hand_run_that_held_the_other(
  embodiment: EmbodimentCfg,
) -> None:
  """The controlled experiment: driving the right hand alone, with the env
  holding the left where it was sent, is bitwise the run that drove both and
  asked the left for its reset pose every step."""
  both = _env(
    Task(driven=GRIPPERS, start=Kicked(PATCH), goal=PlacedOnTape()), embodiment
  )
  right = _env(right_task(PATCH), embodiment)
  obs_both, _ = both.reset(seed=11)
  obs_right, _ = right.reset(seed=11)
  assert np.array_equal(
    obs_both["right_gripper"]["pose"], obs_right["right_gripper"]["pose"]
  )
  sent = both.scene.observe(both.datas[0].qpos[None])[0, 0]
  for step in range(20):
    plan = hold(flat_row(obs_right))
    plan[:, 0] += 0.002 * (step + 1)
    plan[:, 7] -= 0.01 * (step + 1)
    whole = np.concatenate([np.tile(sent, (HORIZON, 1)), plan], axis=1)
    obs_both, *_ = both.step(whole)
    obs_right, *_ = right.step(plan)
    for key in ("pose", "opening", "image"):
      assert np.array_equal(
        obs_both["right_gripper"][key], obs_right["right_gripper"][key]
      )
  assert np.array_equal(both.datas[0].qpos, right.datas[0].qpos)


def test_a_one_hand_task_shows_and_commands_that_hand_alone() -> None:
  env = _env(right_task(PATCH), FloatingCfg())
  obs, _ = env.reset(seed=0)
  assert set(obs) == {"row", "right_gripper"}
  assert env.action_space.shape == (HORIZON, PER_HAND)
  assert flat_row(obs).shape == (PER_HAND,)
  with pytest.raises(ValueError, match="shape"):
    env.step(np.zeros((HORIZON, 2 * PER_HAND)))


def test_the_idle_hand_stays_in_the_air_and_in_the_picture(
  embodiment: EmbodimentCfg,
) -> None:
  env = _env(right_task(PATCH), embodiment)
  obs, _ = env.reset(seed=3)
  left = env.scene.hands[0]
  parked = env.datas[0].xpos[left.base].copy()
  for step in range(30):
    plan = hold(flat_row(obs))
    # The nudge stops at the table's far edge, where the right lens still has
    # the left gripper in view.
    plan[:, 0] = np.minimum(plan[:, 0] + 0.003 * (step + 1), 0.85)
    obs, *_ = env.step(plan)
  moved = np.linalg.norm(env.datas[0].xpos[left.base] - parked)
  assert moved < 0.001, f"the idle hand drifted {moved * 1e3:.2f} mm"
  model = env.scene.model
  seen = set(np.unique(env.views["right_gripper"].body).tolist())
  left_bodies = {
    body
    for body in range(model.nbody)
    if model.body(body).name.startswith("left_gripper")
  }
  assert seen & left_bodies, "the right lens sees the idle left gripper"
  assert "left_gripper" not in obs


def test_the_idle_hand_returns_to_where_it_was_sent_after_a_bump() -> None:
  env = _env(right_task(PATCH), FloatingCfg())
  obs, _ = env.reset(seed=5)
  left = carried(env.scene.hands[0], FreeHand)
  sent = env.datas[0].qpos[left.qpos : left.qpos + 3].copy()
  env.datas[0].qpos[left.qpos] += 0.01
  for _ in range(30):
    obs, *_ = env.step(hold(flat_row(obs)))
  back = np.linalg.norm(env.datas[0].qpos[left.qpos : left.qpos + 3] - sent)
  assert back < 0.002, f"the bumped hand stayed {back * 1e3:.1f} mm away"


def test_a_kick_succeeds_when_the_cube_rests_in_the_picking_region() -> None:
  env = _env(left_task(PATCH), FloatingCfg())
  spot = np.array([0.55, -0.05])
  assert PATCH.contains(spot) and not PATCH.contains(np.array([0.20, 0.0]))
  obs, info = env.reset(seed=0, options={"cube": spot})
  assert not bool(info["on_target"][0])
  won = False
  for _ in range(12):
    obs, reward, terminated, _, info = env.step(hold(flat_row(obs)))
    won |= bool(terminated[0])
    if won:
      break
  assert won and bool(info["is_success"][0]) and reward[0] == 1.0
  env = _env(left_task(PATCH), FloatingCfg())
  obs, info = env.reset(seed=0)
  for _ in range(12):
    obs, _, terminated, _, info = env.step(hold(flat_row(obs)))
  assert not bool(info["is_success"][0]), "a cube left on the tape is no kick"


def test_a_task_names_hands_the_rig_has() -> None:
  with pytest.raises(ValueError, match="drives some of"):
    Task(driven=("left_hand",), start=OnTape(), goal=PlacedOnTape())
  with pytest.raises(ValueError, match="drives some of"):
    Task(driven=(), start=OnTape(), goal=PlacedOnTape())
  with pytest.raises(ValueError, match="in that order"):
    Task(driven=tuple(reversed(GRIPPERS)), start=OnTape(), goal=PlacedOnTape())
  assert BIMANUAL.driven == GRIPPERS
  assert isinstance(left_task(PATCH).goal, RestingIn)


def test_a_task_may_see_a_lens_it_does_not_drive() -> None:
  env = _env(
    Task(("right_gripper",), Kicked(PATCH), PlacedOnTape(), seen=("left_gripper",)),
    FloatingCfg(),
  )
  obs, _ = env.reset(seed=0)
  assert set(obs) == {"row", "left_gripper", "right_gripper"}
  assert set(obs["left_gripper"]) == {"image", "image_row"}
  assert set(obs["right_gripper"]) == {"pose", "opening", "image", "image_row"}
  assert flat_row(obs).shape == (PER_HAND,)
  assert env.observation_space["left_gripper"].keys() == {"image", "image_row"}
