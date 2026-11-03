"""The LEGO sorting scene: layouts, scoring, both kinds of bin, the swapped
drawing, and the env."""

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from specialist_sim import episode
from specialist_sim.action import hold
from specialist_sim.gripper import SLOTS
from specialist_sim.sort_lego import Cfg, SortLegoEnv, assets, initial, scene, task
from specialist_sim.sort_lego.env import COLORS

BLIND = Cfg(render=False)
"""The scene without lenses: physics and scoring only."""


def _card(model: mujoco.MjModel, color: str) -> np.ndarray:
  """The texels the card on the bin of ``color`` wears."""
  material = model.geom_matid[model.geom(f"{color}_bin_card").id]
  texture = model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB]
  start = model.tex_adr[texture]
  size = (
    model.tex_width[texture] * model.tex_height[texture] * model.tex_nchannel[texture]
  )
  return np.asarray(model.tex_data[start : start + size])


@pytest.mark.parametrize("mode", initial.MODES)
def test_a_seed_lays_the_same_tray(mode: str) -> None:
  first, second = (
    initial.draw(mode, 7, COLORS, BLIND.bins),
    initial.draw(mode, 7, COLORS, BLIND.bins),
  )
  assert np.array_equal(first.poses, second.poses)
  assert np.array_equal(first.active, second.active)


@pytest.mark.parametrize("bins", scene.BINS)
@pytest.mark.parametrize("mode", initial.MODES)
def test_bricks_lie_in_the_well_and_the_rest_in_their_bins(
  mode: str, bins: str
) -> None:
  for seed in range(5):
    layout = initial.draw(mode, seed, COLORS, bins)
    on = layout.poses[layout.active]
    assert (
      np.abs(on[:, :2] - scene.TRAY_CENTER) <= np.asarray(initial.WALLS) + 0.03
    ).all()
    parked = zip(
      layout.poses[~layout.active], np.array(COLORS)[~layout.active], strict=True
    )
    for pose, color in parked:
      assert task.in_bin(pose[:3], bins) == color


def test_a_count_sets_how_many_bricks_lie_on_the_tray() -> None:
  assert initial.draw("drop", 3, COLORS, BLIND.bins, bricks=5).active.sum() == 5
  assert initial.draw("pile", 3, COLORS, BLIND.bins, bricks=5).active.sum() == 5
  assert initial.draw("single", 0, COLORS, BLIND.bins).active.sum() == 1


def test_a_drop_lays_one_to_eight_bricks_none_inside_another() -> None:
  env = SortLegoEnv(BLIND)
  model, data = env.rig.model, env.rig.datas[0]
  bricks = env._brick_geoms
  counts = set()
  for seed in range(60):
    layout = initial.draw("drop", seed, COLORS, BLIND.bins)
    counts.add(int(layout.active.sum()))
    env.place(0, layout.poses, initial.starts(SLOTS))
    mujoco.mj_forward(model, data)
    for contact in data.contact[: data.ncon]:
      one, other = contact.geom
      assert not (bricks[one] and bricks[other] and contact.dist < 0)
  assert counts == set(range(1, 9))


def test_a_reset_settles_the_bricks_on_the_tray_and_scores_none_sorted() -> None:
  env = SortLegoEnv(replace(BLIND, bricks=3), num_envs=2)
  observation, report = env.reset([3, 4])
  for index in range(2):
    layout = env.layouts[index]
    assert layout is not None and layout.seed == 3 + index
    on = np.flatnonzero(layout.active)
    assert all(task.on_tray(env.bricks(index)[brick, :3]) for brick in on)
  assert report.started.tolist() == [3, 3] and report.sorted.tolist() == [0, 0]
  assert not report.success.any() and not report.done.any()
  assert observation.state.shape == (2, env.cfg.clock.per_step, 2, 8)
  assert observation.state[0, -1, 1] == pytest.approx(initial.home(SLOTS[1]), abs=1e-3)


@pytest.mark.parametrize("bins", scene.BINS)
def test_a_brick_in_its_own_bin_is_sorted_and_the_episode_succeeds(bins: str) -> None:
  env = SortLegoEnv(replace(BLIND, bricks=1, bins=bins))
  observation, _ = env.reset([5])
  layout = env.layouts[0]
  assert layout is not None
  brick = int(np.flatnonzero(layout.active)[0])
  color = env.cfg.colors[brick]
  at = env._brick_q[brick]
  qpos = env.rig.datas[0].qpos
  qpos[at : at + 3] = scene.BINS[bins].floor_middle(color) + [0.0, 0.0, 0.03]
  qpos[at + 3 : at + 7] = [1.0, 0.0, 0.0, 0.0]
  _, report = env.step(hold(observation.state[:, -1]))
  assert report.sorted[0] == 1 and report.success[0] and report.done[0]
  assert report.progress[0] == 1.0 and not report.labels["missorted"][0]


def test_hands_standing_where_a_brick_would_start_pass_over_that_seed() -> None:
  env = SortLegoEnv(replace(BLIND, mode="single"))
  layout = initial.draw("single", 11, COLORS, BLIND.bins)
  brick = layout.poses[np.flatnonzero(layout.active)[0]]
  hands = initial.starts(SLOTS)
  hands[1, :3] = brick[:3] + np.array([0.0, 0.0, 0.01])
  env.reset([11], hands=hands)
  assert env.layouts[0] is not None and env.layouts[0].seed > 11


def test_the_swapped_scene_draws_bricks_and_cards_in_the_other_color() -> None:
  plain = mujoco.MjSpec.compile(assets.spec(COLORS, "clear"))
  swapped = mujoco.MjSpec.compile(assets.spec(COLORS, "clear", swapped=True))
  for index, color in enumerate(COLORS):
    body = plain.geom_bodyid == plain.body(f"brick_{index}").id
    other = "blue" if color == "red" else "red"
    assert swapped.geom_rgba[body, :3] == pytest.approx(
      np.tile(assets.COLOR_RGB[other], (int(body.sum()), 1))
    )
  for color, other in (("red", "blue"), ("blue", "red")):
    assert np.array_equal(_card(swapped, color), _card(plain, other))
  assert np.array_equal(swapped.body_pos, plain.body_pos)
  assert np.array_equal(swapped.geom_size, plain.geom_size)


def test_swapped_akrobins_are_drawn_in_the_other_color() -> None:
  plain = mujoco.MjSpec.compile(assets.spec(COLORS, "akrobins"))
  swapped = mujoco.MjSpec.compile(assets.spec(COLORS, "akrobins", swapped=True))
  for color, other in (("red", "blue"), ("blue", "red")):
    shell = plain.geom(f"{color}_bin_shell").id
    assert plain.geom_rgba[shell, :3] == pytest.approx(assets.BIN_RGB[color])
    assert swapped.geom_rgba[shell, :3] == pytest.approx(assets.BIN_RGB[other])
  assert np.array_equal(swapped.geom_pos, plain.geom_pos)


def test_the_akrobin_shell_matches_its_measured_size() -> None:
  model = mujoco.MjSpec.compile(assets.spec(COLORS, "akrobins"))
  start, count = (
    model.mesh_vertadr[model.mesh("akrobin").id],
    model.mesh_vertnum[model.mesh("akrobin").id],
  )
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  shell = model.geom("red_bin_shell").id
  vertices = (
    model.mesh_vert[start : start + count] @ data.geom_xmat[shell].reshape(3, 3).T
  )
  vertices += data.geom_xpos[shell]
  size = vertices.max(axis=0) - vertices.min(axis=0)
  assert size == pytest.approx(scene.BINS["akrobins"].size, abs=5e-4)
  middle = scene.BINS["akrobins"].middle("red")
  assert (vertices.max(axis=0) + vertices.min(axis=0))[:2] / 2 == pytest.approx(
    middle, abs=5e-4
  )
  assert vertices[:, 2].min() == pytest.approx(scene.DESK_TOP, abs=1e-4)


@pytest.mark.parametrize("x", [-0.03, 0.0, 0.045, 0.055])
def test_a_brick_dropped_into_an_akrobin_comes_to_rest_sorted(x: float) -> None:
  """From over the floor, or onto the front ramp, which a brick slides down. One
  brick of each color, so the bin is empty but for it."""
  env = SortLegoEnv(replace(BLIND, bricks=1, mode="single", colors=("red", "blue")))
  observation, _ = env.reset([5])
  layout = env.layouts[0]
  assert layout is not None
  brick = int(np.flatnonzero(layout.active)[0])
  color = env.cfg.colors[brick]
  at = env._brick_q[brick]
  qpos = env.rig.datas[0].qpos
  middle = scene.BINS["akrobins"].middle(color)
  qpos[at : at + 3] = [middle[0] + x, middle[1], scene.DESK_TOP + 0.09]
  qpos[at + 3 : at + 7] = np.array([0.96, 0.1, 0.25, 0.0]) / np.linalg.norm(
    [0.96, 0.1, 0.25, 0.0]
  )
  for _ in range(8):
    observation, report = env.step(hold(observation.state[:, -1]))
  pose = env.bricks(0)[brick]
  assert report.sorted[0] == 1, pose
  assert pose[2] < scene.DESK_TOP + scene.BINS["akrobins"].floor[2] + 0.01, pose
  assert pose[0] - middle[0] < scene.BINS["akrobins"].floor[1] + 0.01, pose


def test_an_episode_records_every_row_and_round_trips(tmp_path: Path) -> None:
  env = SortLegoEnv(replace(BLIND, bricks=1))
  observation, _ = env.reset([2])
  recorder = episode.Recorder()
  recorder.start("one", env.describe(0))
  recorder.extend(*(rows[-1:] for rows in env.rig.rows(0)))
  for _ in range(2):
    observation, report = env.step(hold(observation.state[:, -1]))
    recorder.extend(*env.rig.rows(0))
  path = episode.save(tmp_path, recorder.finish({"score": env.score(0).as_dict()}))
  loaded = episode.load(path)
  rows = 2 * env.cfg.clock.per_step + 1
  assert loaded.timestamp == pytest.approx(np.arange(rows) * env.cfg.clock.row_dt)
  assert loaded.state.shape == (rows, 2, 8) and loaded.commanded.shape == (rows, 2, 8)
  assert loaded.qpos.shape == (rows, env.rig.model.nq)
  assert loaded.meta["layout"] == {"mode": "drop", "seed": 2, "bricks": 1}
  assert loaded.meta["score"]["started"] == 1 and recorder.name is None


def test_the_lens_fires_on_every_fifth_row_and_sees_the_tray() -> None:
  env = SortLegoEnv(Cfg(image_size=64, driven=(SLOTS[1],), frames=2, bricks=2))
  observation, _ = env.reset([1])
  for _ in range(3):
    observation, _ = env.step(hold(observation.state[:, -1]))
    rows = observation.image_row[0, 0]
    assert (rows % 5 == 0).all() and rows[1] - rows[0] == 5
  assert observation.image.shape == (1, 1, 2, 64, 64, 3)
  assert (observation.image[0, 0, -1].sum(axis=-1) > 0).mean() > 0.2
