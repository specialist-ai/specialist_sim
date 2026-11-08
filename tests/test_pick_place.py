"""The pick-and-place room and task: what stands in it and how an episode is
judged."""

from dataclasses import replace

import numpy as np
import pytest

from fixtures.scene import SMALL, look, overhead
from specialist_sim.arrays import Float64Array
from specialist_sim.env import Env, Start
from specialist_sim.tasks import desk, pick_place
from specialist_sim.tasks.pick_place import PickPlace


def _cube(env: Env[PickPlace]) -> Float64Array:
  """Every env's cube position, ``(num_envs, 3)``."""
  return env.objects().poses[:, 0, :3]


def test_a_settled_cube_on_the_tape_latches_success_after_the_dwell() -> None:
  """Success latches on the step the cube has rested on the tape for the
  dwell, and holds after it."""
  env = Env(PickPlace(), replace(SMALL, autoreset=False))
  observation, _ = env.reset([0])
  aside = np.array([[0.10, -0.10, pick_place.REST + desk.GAP, 1.0, 0.0, 0.0, 0.0]])
  env.reset([0], start=Start(observation.state[0, -1], aside))
  env.step()
  assert not (np.abs(_cube(env)[0, :2]) <= pick_place.ON_TAPE).all()
  at = env.model.joint("cube").qposadr[0]
  env.sim.qpos[0, at : at + 2] = 0.0
  dwell = round(PickPlace.dwell / env.cfg.clock.step_dt)
  successes = []
  dones = []
  for _ in range(dwell + 3):
    _, info = env.step()
    successes.append(bool(info["success"][0]))
    dones.append(bool(info["done"][0]))
  assert successes == [False] * (dwell - 1) + [True] * 4, "the dwell is not the latch"
  assert dones == successes
  assert not info["labels"]["picked"][0], (
    "a cube nudged onto the tape was never carried"
  )
  env.sim.qpos[0, at : at + 2] = 0.10, -0.10
  _, info = env.step()
  assert info["progress"][0] == 0.0 and info["success"][0] and info["done"][0], (
    "success stays latched once the cube leaves the tape"
  )


def test_a_fallen_cube_ends_the_episode_and_the_next_step_opens_seed_1() -> None:
  """Env 0 under its hands as they stand; a reset first gives env 1 seed 9."""
  env = Env(PickPlace(), SMALL, 2)
  env.reset([0, 0])
  at = env.model.joint("cube").qposadr[0]
  env.sim.qpos[:, at + 2] = desk.TOP - desk.FALLEN - 0.01
  before, info = env.step()
  assert info["done"].all() and not info["success"].any()
  env.reset([9], [1])
  after, info = env.step()
  assert info["seed"].tolist() == [1, 9] and after.row[0, -1] == env.cfg.clock.per_step
  assert np.allclose(after.state[0, 0, :, :3], before.state[0, -1, :, :3], atol=0.01)


def test_a_seed_names_one_start_in_the_region_and_off_the_tape() -> None:
  env = Env(PickPlace(), SMALL)

  def spot(seed: int) -> Float64Array:
    env.reset([seed])
    return _cube(env)[0, :2]

  spots = np.stack([spot(seed) for seed in range(16)])
  assert (np.abs(spots) <= np.array(pick_place.REGION) / 2.0 + 1e-9).all()
  assert not (np.abs(spots) <= pick_place.ON_TAPE).all(axis=1).any()
  assert np.array_equal(spots[3], spot(3)), "a seed names one start"


def test_the_room_stands_a_desk_with_a_cube_the_overview_watches() -> None:
  env = Env(PickPlace(), SMALL)
  env.reset([0])
  objects = env.objects()
  assert objects.poses.shape == (1, 1, 7), "the cube is the room's one object"
  assert objects.poses[0, 0, :3] == pytest.approx(
    env.sim.xpos[0, env.model.body("cube").id]
  )
  view = overhead((0.0, 0.0, desk.TOP + 0.7), 104.0, 96)
  image, body = look(env.model, env.sim.get_data(0), view, 96)
  assert image.shape == (96, 96, 3) and (image.sum(axis=2) > 0).mean() > 0.9
  struck = np.bincount(body[body >= 0].ravel()).argmax()
  assert env.model.body(int(struck)).name == "room"
