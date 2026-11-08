"""The LEGO sorting room: layouts, the AkroBins, how a tray is judged, and the
task."""

import json
from dataclasses import fields, replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from fixtures.lego import brick_geoms, brick_q
from fixtures.scene import BLIND
from specialist_sim import episode
from specialist_sim.arrays import Float64Array
from specialist_sim.camera.lens import Lens
from specialist_sim.env import Cfg, Env, Objects, Start, make
from specialist_sim.episode import Episode, Rows
from specialist_sim.gripper import HANDS
from specialist_sim.tasks import desk
from specialist_sim.tasks.sort_lego import SortLego, initial, measures, scene
from specialist_sim.tasks.sort_lego.measures import BRICK_COLORS, COLORS, in_bin


def _drawn(mode: str, seed: int, bricks: int | None = None) -> Float64Array:
  return initial.draw(mode, np.random.default_rng(seed), bricks)


def _floor(color: str) -> Float64Array:
  """The middle of a bin's flat floor, on it, in the room."""
  x, y = measures.bin_middle(color)
  back, front, top = measures.BIN_FLOOR
  return np.array([x - (back + front) / 2, y, desk.TOP + top])


def _active(env: Env[SortLego]) -> int:
  """The first brick env 0's tray started with."""
  return int(np.flatnonzero(in_bin(env.objects().laid[0, :, :3]) < 0)[0])


@pytest.mark.parametrize("mode", initial.MODES)
def test_a_seed_lays_its_tray_in_the_well_and_parks_the_rest(mode: str) -> None:
  own = np.array([COLORS.index(color) for color in BRICK_COLORS])  # each brick's bin
  parked = [
    measures.parked(color, BRICK_COLORS[:index].count(color))
    for index, color in enumerate(BRICK_COLORS)
  ]
  for seed in range(200):
    poses = _drawn(mode, seed)
    landed = in_bin(poses[:, :3])
    on = landed < 0
    assert np.array_equal(on, (poses[:, :3] != parked).any(axis=1))
    assert (np.abs(poses[on, :2] - measures.TRAY_CENTER) <= initial.WALLS + 0.03).all()
    assert (landed[~on] == own[~on]).all()
  assert np.array_equal(poses, _drawn(mode, seed))
  assert (in_bin(_drawn(mode, 3, bricks=5)[:, :3]) < 0).sum() == 5


def test_a_drop_lays_one_to_eight_bricks_none_inside_another() -> None:
  env = Env(SortLego(), BLIND)
  sim = env.sim
  bricks = brick_geoms(env)
  at = np.array([brick_q(env, i) for i in range(len(BRICK_COLORS))])[:, None]
  counts = set()
  for seed in range(60):
    poses = _drawn("drop", seed)
    counts.add(int((in_bin(poses[:, :3]) < 0).sum()))
    sim.qpos[0, at + np.arange(7)] = poses
    sim.forward([0], stage=mujoco.mjtStage.mjSTAGE_POS)
    for contact in sim.get_data(0).contact:
      one, other = contact.geom
      assert not (bricks[one] and bricks[other] and contact.dist < 0)
  assert counts == set(range(1, 9))


def test_a_reset_settles_the_bricks_on_the_tray_and_scores_none_sorted() -> None:
  env = Env(SortLego(bricks=3), BLIND, 2)
  observation, info = env.reset([3, 4])
  for key, count in {"started": 3, "on_tray": 3, "sorted": 0, "missorted": 0}.items():
    assert info["counts"][key].tolist() == [count] * 2, key
  assert info["counts"]["lost"].tolist() == [0, 0]
  assert not info["success"].any() and not info["done"].any()
  at = brick_q(env, _active(env))
  env.sim.qpos[0, at : at + 3] = 0.0, 0.4, desk.TOP + 0.01
  _, info = env.step()
  assert info["done"].tolist() == [True, False], "a brick touched the desk"
  assert observation.state[0, -1, 1] == pytest.approx(initial.home(HANDS[1]), abs=1e-3)


def test_a_brick_counts_once_by_where_it_landed() -> None:
  """A brick that started on the tray counts once, as sorted, missorted, lost
  or on the tray, and a lost brick under the tray is not on it; a brick that
  started in its bin counts for nothing."""
  red, red_too, blue, blue_too, red_off = 0, 1, 6, 7, 2
  tray = [*measures.TRAY_CENTER, measures.TRAY_FLOOR + 0.01]
  laid = np.zeros((1, len(BRICK_COLORS), 7))
  laid[0, :, :3] = [
    measures.parked(color, BRICK_COLORS[:index].count(color))
    for index, color in enumerate(BRICK_COLORS)
  ]
  laid[0, [red, red_too, blue, blue_too, red_off], :3] = tray
  poses = laid.copy()
  poses[0, red, :3] = _floor("red") + [0.0, 0.0, 0.01]
  poses[0, red_too, :3] = _floor("blue") + [0.0, 0.0, 0.01]
  poses[0, blue, :3] = [*measures.TRAY_CENTER, desk.TOP - desk.FALLEN - 0.01]
  poses[0, red_off, :3] = [0.0, 0.4, desk.TOP + 0.01]
  n = len(BRICK_COLORS)
  still, apart, ok = np.zeros((1, n, 6)), np.zeros((1, n), bool), np.zeros(1, bool)
  verdict = SortLego().judge(Objects(poses, still, apart, laid, ok))
  counted = {key: int(value[0]) for key, value in verdict.counts.items()}
  assert counted == {"started": 5, "sorted": 1, "missorted": 1, "on_tray": 1, "lost": 1}
  assert verdict.failed[0], "a brick is missorted"
  assert verdict.progress[0] == 0.2
  held, moving = apart.copy(), still.copy()
  held[0, red], moving[0, red, 3] = True, 0.1
  for velocity, grip in ((still, held), (moving, apart)):
    judged = SortLego().judge(Objects(poses, velocity, grip, laid, ok))
    assert judged.counts["sorted"][0] == 0, "a held or moving brick is not sorted"


def test_hands_standing_where_a_brick_would_start_pass_over_that_seed() -> None:
  """Each env's seed is decided on its own hands."""
  env = Env(SortLego(mode="pile", bricks=1), BLIND, 2)
  poses = _drawn("pile", 11, bricks=1)
  brick = poses[in_bin(poses[:, :3]) < 0][0]
  hands = np.stack([initial.starts(HANDS)] * 2)
  hands[1, 1, :3] = brick[:3] + np.array([0.0, 0.0, 0.01])
  _, info = env.reset([11, 11], start=Start(hands))
  assert info["seed"][0] == 11 and info["seed"][1] > 11


def test_the_akrobin_shell_matches_its_measured_size() -> None:
  model = mujoco.MjSpec.compile(scene.spec())
  start = model.mesh_vertadr[model.mesh("akrobin").id]
  count = model.mesh_vertnum[model.mesh("akrobin").id]
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  shell = model.geom("red_bin_shell").id
  vertices = (
    model.mesh_vert[start : start + count] @ data.geom_xmat[shell].reshape(3, 3).T
  )
  vertices += data.geom_xpos[shell]
  size = vertices.max(axis=0) - vertices.min(axis=0)
  assert size == pytest.approx(measures.BIN_SIZE, abs=5e-4)
  assert (vertices.max(axis=0) + vertices.min(axis=0))[:2] / 2 == pytest.approx(
    measures.bin_middle("red"), abs=5e-4
  )
  assert vertices[:, 2].min() == pytest.approx(desk.TOP, abs=1e-4)


@pytest.mark.parametrize("x", [-0.03, 0.0, 0.045, 0.055])
def test_a_brick_dropped_into_an_akrobin_comes_to_rest_sorted(x: float) -> None:
  """From over the floor, or onto the front ramp, which a brick slides down, ``x``
  along the bin's own x. Every other brick waits on the floor, away from the
  desk, so the bin is empty but for it."""
  env = Env(SortLego(bricks=1, mode="pile"), replace(BLIND, autoreset=False))
  env.reset([5])
  brick = _active(env)
  for other in range(len(BRICK_COLORS)):
    if other != brick:
      q = brick_q(env, other)
      env.sim.qpos[0, q : q + 7] = [2.0 + 0.05 * other, 0.0, 0.01, 1.0, 0, 0, 0]
  at = brick_q(env, brick)
  middle = measures.bin_middle(BRICK_COLORS[brick])
  env.sim.qpos[0, at : at + 3] = [middle[0] - x, middle[1], desk.TOP + 0.09]
  tipped = np.array([0.96, 0.1, 0.25, 0.0])
  env.sim.qpos[0, at + 3 : at + 7] = tipped / np.linalg.norm(tipped)
  for _ in range(8):
    _, info = env.step()
  assert info["counts"]["sorted"][0] == 1 and info["success"][0] and info["done"][0], (
    env.objects().poses[0, brick]
  )
  dof = env.model.joint(f"brick_{brick}_joint").dofadr[0]
  speed = float(np.linalg.norm(env.sim.qvel[0, dof : dof + 6]))
  assert speed < 0.02, f"the brick was still going at {speed:.3f} after 0.8 s"


def test_an_episode_records_every_row_and_round_trips(tmp_path: Path) -> None:
  env = Env(SortLego(bricks=1), BLIND)
  env.reset([2])
  taken = [env.rows(0)[-1:]]
  for _ in range(2):
    env.step()
    taken.append(env.rows(0))
  meta = env.describe()
  joined = episode.joined(taken)
  loaded = episode.load(episode.save(tmp_path, Episode("one", joined, meta)))
  rows = 2 * env.cfg.clock.per_step + 1
  assert loaded.rows.timestamp == pytest.approx(np.arange(rows) * env.cfg.clock.row_dt)
  for column in fields(Rows):
    kept = getattr(loaded.rows, column.name)
    assert np.array_equal(kept, getattr(joined, column.name)), column.name
    assert kept.dtype == np.float64, column.name
  again = make(loaded.meta)
  assert again.task == env.task and again.cfg == env.cfg


def test_an_env_is_stood_again_from_what_it_describes() -> None:
  lens = replace(Lens(), pos=(0.0, 0.0, 0.25))
  cfg = Cfg(lenses={HANDS[0]: lens}, driven=(HANDS[1],), image_hands=(), image_size=48)
  env = Env(SortLego(mode="pile", bricks=2), cfg)
  meta = json.loads(json.dumps(env.describe()))
  again = make(meta)
  assert again.task == env.task and again.cfg == env.cfg
  seen = make(meta, image_hands=HANDS)
  assert seen.cfg.image_hands == HANDS and seen.cfg.lenses == cfg.lenses


def test_the_lens_sees_the_lit_tray() -> None:
  one_hand = (HANDS[1],)
  cfg = Cfg(image_size=64, driven=one_hand, image_hands=one_hand, camera_latency=0.0)
  env = Env(SortLego(bricks=2), cfg)
  observation, _ = env.reset([1])
  assert (observation.image[0, 0].sum(axis=-1) > 0).mean() > 0.2
