"""Resetting and stepping: seeding, validity, and the shape of what comes back."""

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.action import hold
from specialist_sim.spaces import (
  ACTION_HIGH,
  GRIPPERS,
  IMAGE_SHAPE,
  STATE_HIGH,
  STATE_LOW,
)


@pytest.fixture(scope="module")
def env() -> PickPlaceEnv:
  return PickPlaceEnv()


def test_a_reset_reports_every_camera_and_the_hands(env: PickPlaceEnv) -> None:
  obs, info = env.reset(seed=0)
  assert sorted(obs) == sorted([*GRIPPERS, "state"])
  for name in GRIPPERS:
    assert obs[name].shape == IMAGE_SHAPE and obs[name].dtype == np.uint8
  assert obs["state"].shape == STATE_LOW.shape and obs["state"].dtype == np.float32
  assert set(info) == set(env.step(hold(obs["state"]))[4])


def test_the_same_seed_starts_the_same_episode(env: PickPlaceEnv) -> None:
  first, _ = env.reset(seed=7)
  second, _ = env.reset(seed=7)
  assert first["state"] == pytest.approx(second["state"])
  assert env.reset(seed=8)[0]["state"] != pytest.approx(first["state"])


def test_a_reset_never_starts_inside_something(env: PickPlaceEnv) -> None:
  for seed in range(20):
    env.reset(seed=seed)
    assert not env._penetrating()
    assert env.scene.data.qvel == pytest.approx(0.0)


def test_an_episode_truncates_at_the_horizon() -> None:
  env = PickPlaceEnv(PickPlaceCfg(max_steps=3))
  obs, _ = env.reset(seed=0)
  commanded = hold(obs["state"])
  flags = [env.step(commanded)[3] for _ in range(3)]
  assert flags == [False, False, True]


def test_a_command_outside_the_box_is_clipped(env: PickPlaceEnv) -> None:
  env.reset(seed=0)
  env.step(np.full_like(ACTION_HIGH, 1e6, dtype=np.float64))
  assert np.isfinite(env.scene.data.qpos).all()


def test_reset_options_place_the_cube_and_the_hands() -> None:
  env = PickPlaceEnv()
  chosen = env.reset(seed=1)[0]["state"].astype(np.float64)
  cube = (0.52, 0.06)
  obs, _ = env.reset(seed=2, options={"state": chosen, "cube": cube})
  assert obs["state"] == pytest.approx(chosen, abs=1e-6)
  assert env.scene.data.xpos[env.scene.cube.body][:2] == pytest.approx(cube, abs=1e-6)


def test_a_render_mode_frame_watches_the_room() -> None:
  """The watching view sees the table, and moves when the scene does."""
  cfg = PickPlaceCfg(render_size=200)
  env = PickPlaceEnv(cfg, render_mode="rgb_array")
  obs, _ = env.reset(seed=0)
  before = env.render()
  assert before is not None and before.shape == (200, 200, 3)
  assert (before.sum(axis=2) > 0).mean() > 0.9
  commanded = hold(obs["state"])
  commanded[:, 2] += 0.15
  for _ in range(10):
    env.step(commanded)
  after = env.render()
  assert after is not None and not np.array_equal(before, after)
  assert PickPlaceEnv().render() is None


def test_a_command_outside_the_box_lands_on_the_edge_of_it() -> None:
  """A command past the bounds is held at them, rather than driving the plant."""
  env = PickPlaceEnv()
  obs, _ = env.reset(seed=0)
  commanded = hold(obs["state"])
  commanded[:, :3] = 1e6
  env.step(commanded)
  assert env.hands._position[0] == pytest.approx(STATE_HIGH[:3], abs=1e-6)
  for _ in range(60):
    obs, *_ = env.step(commanded)
  assert (obs["state"][:3] <= STATE_HIGH[:3] + 1e-3).all()
  assert (obs["state"][:3] >= STATE_LOW[:3] - 1e-3).all()


def test_reset_reports_what_the_cube_is_actually_doing() -> None:
  """The opening info describes the scene."""
  env = PickPlaceEnv()
  _, info = env.reset(seed=4)
  assert info["resting"] and not info["held"]
  assert not info["on_target"] and not info["dropped"] and not info["is_success"]
