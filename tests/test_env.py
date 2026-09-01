"""Resetting and stepping: seeding, validity, and the shape of what comes back."""

from typing import Any

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.action import hold
from specialist_sim.env import MAX_LAG
from specialist_sim.observation import flat_row
from specialist_sim.spaces import (
  ACTION_HIGH,
  GRIPPERS,
  IMAGE_SHAPE,
  PER_HAND,
  STATE_HIGH,
  STATE_LOW,
)


@pytest.fixture(scope="module")
def env() -> PickPlaceEnv:
  return PickPlaceEnv()


def test_a_reset_reports_every_camera_and_the_hands(env: PickPlaceEnv) -> None:
  obs, info = env.reset(seed=0)
  assert sorted(obs) == sorted([*GRIPPERS, "step"])
  assert obs["step"][0].tolist() == list(range(1 - env.rows, 1))
  for name in GRIPPERS:
    hand = obs[name]
    assert hand["image"].shape == (env.worlds, *IMAGE_SHAPE)
    assert hand["image"].dtype == np.uint8
    assert hand["pose"].shape == (env.worlds, env.rows, 7)
    assert hand["pose"].dtype == np.float32
    assert hand["opening"].shape == (env.worlds, env.rows)
    assert np.array_equal(hand["pose"][0], np.tile(hand["pose"][0, -1], (env.rows, 1)))
    assert np.array_equal(hand["opening"][0], np.full(env.rows, hand["opening"][0, -1]))
    assert hand["image_step"][0] == -env.cfg.camera_lag
  assert flat_row(obs).shape == STATE_LOW.shape
  assert set(info) == set(env.step(hold(flat_row(obs)))[4])


def test_the_same_seed_starts_the_same_episode(env: PickPlaceEnv) -> None:
  first, _ = env.reset(seed=7)
  second, _ = env.reset(seed=7)
  assert flat_row(first) == pytest.approx(flat_row(second))
  assert flat_row(env.reset(seed=8)[0]) != pytest.approx(flat_row(first))


def test_a_reset_never_starts_inside_something(env: PickPlaceEnv) -> None:
  for seed in range(20):
    env.reset(seed=seed)
    assert not env._penetrating(env.scene.data)
    assert env.scene.data.qvel == pytest.approx(0.0)


def test_an_episode_truncates_at_the_horizon() -> None:
  env = PickPlaceEnv(PickPlaceCfg(max_steps=3))
  obs, _ = env.reset(seed=0)
  commanded = hold(flat_row(obs))
  flags = [env.step(commanded)[3] for _ in range(3)]
  assert flags == [False, False, True]


def test_a_command_outside_the_box_is_clipped(env: PickPlaceEnv) -> None:
  env.reset(seed=0)
  env.step(np.full_like(ACTION_HIGH, 1e6, dtype=np.float64))
  assert np.isfinite(env.scene.data.qpos).all()


def test_reset_options_place_the_cube_and_the_hands() -> None:
  env = PickPlaceEnv()
  chosen = flat_row(env.reset(seed=1)[0]).astype(np.float64)
  cube = (0.52, 0.06)
  obs, _ = env.reset(seed=2, options={"state": chosen, "cube": cube})
  assert flat_row(obs) == pytest.approx(chosen, abs=1e-6)
  assert env.scene.data.xpos[env.scene.cube.body][:2] == pytest.approx(cube, abs=1e-6)


def test_a_render_mode_frame_watches_the_room() -> None:
  """The watching view sees the table, and moves when the scene does."""
  cfg = PickPlaceCfg(render_size=200)
  env = PickPlaceEnv(cfg, render_mode="rgb_array")
  obs, _ = env.reset(seed=0)
  before = env.render()
  assert before is not None and before.shape == (200, 200, 3)
  assert (before.sum(axis=2) > 0).mean() > 0.9
  commanded = hold(flat_row(obs))
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
  commanded = hold(flat_row(obs))
  commanded[:, :3] = 1e6
  env.step(commanded)
  assert env.hands._position[0, 0] == pytest.approx(STATE_HIGH[:3], abs=1e-6)
  for _ in range(60):
    obs, *_ = env.step(commanded)
  assert (flat_row(obs)[:3] <= STATE_HIGH[:3] + 1e-3).all()
  assert (flat_row(obs)[:3] >= STATE_LOW[:3] - 1e-3).all()


def test_reset_reports_what_the_cube_is_actually_doing() -> None:
  """The opening info describes the scene."""
  env = PickPlaceEnv()
  _, info = env.reset(seed=4)
  assert info["resting"] and not info["held"]
  assert info["on_target"] and not info["dropped"] and not info["is_success"]


def _delivered(
  lag: int, steps: int = 8
) -> tuple[dict[tuple[str, int], Any], Any, list[int], int]:
  """Every frame a run hands over, by the row it says the frame was taken at,
  beside the plant's state at each step and the rows those frames came from."""
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, max_steps=10**6, camera_lag=lag))
  obs, _ = env.reset(seed=0)
  action = hold(flat_row(obs))
  action[:, 2] += 0.06
  action[:, PER_HAND + 1] -= 0.05
  seen, qpos, taken = {}, [], []
  for _ in range(steps):
    obs, _, _, _, _ = env.step(action)
    qpos.append(env.scene.data.qpos.copy())
    taken.append(int(obs[GRIPPERS[0]]["image_step"][0]))
    for slot in GRIPPERS:
      seen[(slot, int(obs[slot]["image_step"][0]))] = obs[slot]["image"][0].copy()
  return seen, np.asarray(qpos), taken, env.rows


def test_a_lagging_camera_delivers_what_a_level_one_saw() -> None:
  """A frame is quarantined behind the row it was taken at, and is the frame a
  camera keeping up with the rest of the stack took at that row.

  Rows line up to compare only when the lag is a whole number of them; the
  plant is left alone at any lag, which the test below holds.
  """
  live, plain, _, _ = _delivered(0)
  late, lagged, _, _ = _delivered(20)
  assert np.array_equal(plain, lagged), "the camera moved the plant"
  shared = sorted({row for _, row in late} & {row for _, row in live})
  assert len(shared) >= 5, f"the two runs only share {len(shared)} rows"
  for slot in GRIPPERS:
    for row in shared:
      assert np.array_equal(late[(slot, row)], live[(slot, row)]), (
        f"{slot} was handed a different frame for row {row}"
      )


def test_a_camera_that_lags_leaves_the_plant_alone() -> None:
  """A frame taken partway through a step reads the model between physics steps
  and puts back what it read, so the plant runs as though unwatched.

  A lag that is not a whole number of rows is what takes that path: the frame is
  captured inside the step rather than at the boundary it ends on.
  """
  _, plain, _, rows = _delivered(0)
  for lag in (3, 17, 20):
    _, lagged, taken, _ = _delivered(lag)
    assert np.array_equal(plain, lagged), f"a camera {lag} rows behind moved the plant"
    assert any(row % rows for row in taken) == bool(lag % rows), (
      f"a lag of {lag} rows did not capture where it was meant to"
    )


def test_a_camera_lag_the_stack_cannot_hold_is_refused() -> None:
  for lag in (-1, MAX_LAG + 1):
    with pytest.raises(ValueError, match="rows behind"):
      PickPlaceEnv(PickPlaceCfg(camera_lag=lag))


def test_a_step_reports_the_rows_it_spanned(env: PickPlaceEnv) -> None:
  """Rows are the robot's own clock, and a step returns at the last of them."""
  obs, _ = env.reset(seed=0)
  action = hold(flat_row(obs))
  for step in range(1, 4):
    obs, _, _, _, _ = env.step(action)
    assert obs["step"][0].tolist() == list(
      range((step - 1) * env.rows + 1, step * env.rows + 1)
    )
    for slot in GRIPPERS:
      assert obs[slot]["image_step"][0] == obs["step"][0][-1] - env.cfg.camera_lag


def test_one_world_restarts_while_the_others_carry_on() -> None:
  """An evaluation runs many episodes at once and they do not end together."""
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, max_steps=10**6), worlds=3)
  obs, _ = env.reset(seeds=[0, 1, 2])
  action = np.stack([hold(flat_row(obs, world=w)) for w in range(3)])
  for _ in range(3):
    obs, *_ = env.step(action)
  assert obs["step"][:, -1].tolist() == [3 * env.rows] * 3
  carried = obs[GRIPPERS[0]]["pose"][[0, 2], -1].copy()

  obs, _ = env.reset(seeds=[7], worlds=[1])
  assert obs["step"][1, -1] == 0, "the restarted world keeps counting"
  assert obs["step"][0, -1] == 3 * env.rows, "the others were restarted with it"
  assert np.array_equal(obs[GRIPPERS[0]]["pose"][[0, 2], -1], carried)

  action = np.stack([hold(flat_row(obs, world=w)) for w in range(3)])
  obs, *_ = env.step(action)
  assert obs["step"][:, -1].tolist() == [4 * env.rows, env.rows, 4 * env.rows]


def test_a_reset_refuses_a_world_it_does_not_have() -> None:
  env = PickPlaceEnv(PickPlaceCfg(image_size=64), worlds=2)
  env.reset(seed=0)
  with pytest.raises(ValueError, match="numbered"):
    env.reset(worlds=[2])
  with pytest.raises(ValueError, match="at most once"):
    env.reset(worlds=[0, 0])


def test_a_world_in_a_batch_walks_the_path_it_walks_alone() -> None:
  """Worlds share nothing but the model, so company cannot change one."""
  cfg = PickPlaceCfg(image_size=64, max_steps=10**6)
  alone = PickPlaceEnv(cfg)
  crowd = PickPlaceEnv(cfg, worlds=3)
  obs_a, _ = alone.reset(seed=7)
  obs_c, _ = crowd.reset(seeds=[5, 7, 9])
  for _ in range(6):
    act_a = hold(flat_row(obs_a, world=0))
    act_c = np.stack([hold(flat_row(obs_c, world=w)) for w in range(3)])
    act_c[1] = act_a
    obs_a, *_ = alone.step(act_a)
    obs_c, *_ = crowd.step(act_c)
    assert np.array_equal(alone.datas[0].qpos, crowd.datas[1].qpos)
    for slot in GRIPPERS:
      assert np.array_equal(obs_a[slot]["image"][0], obs_c[slot]["image"][1])
