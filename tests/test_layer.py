"""The layer a demonstration is written as: what it holds of the scene, and in
whose order it holds it."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("specialist_corpus", reason="the dataset extra is not installed")

from specialist_corpus import schema  # noqa: E402
from specialist_corpus.read import Recording  # noqa: E402

from specialist_sim.camera import NOMINAL  # noqa: E402
from specialist_sim.dataset.layer import (  # noqa: E402
  CUBE_POSE,
  Episode,
  cube_chunk,
  decoded,
  record,
  write,
)
from specialist_sim.expert import ExpertCfg  # noqa: E402
from specialist_sim.pickplace import Cfg, PickPlace  # noqa: E402
from specialist_sim.pickplace.script import expert  # noqa: E402

ROOT_HALF = np.sqrt(0.5)
"""A quarter turn about the vertical, as either convention spells its halves."""


def _episode(steps: int) -> tuple[PickPlace, Episode]:
  env = PickPlace(Cfg(image_size=32))
  demonstrator = expert(env, np.random.default_rng(0), ExpertCfg(length=(0.0, 0.0)))
  return env, record(env, demonstrator, seed=0, max_steps=steps)[0]


def test_the_layer_holds_the_scene_at_every_row_of_every_step() -> None:
  env, episode = _episode(6)
  rows = env.cfg.clock.per_step
  assert len(episode.row_times) == 6 * rows
  assert np.allclose(np.diff(episode.row_times), env.cfg.clock.row_dt)
  cube = np.asarray(episode.cube)
  assert cube.shape == (6 * rows, 7)
  at = env.room.cube.qpos
  assert np.allclose(cube[-1], env.rig.datas[0].qpos[at : at + 7])
  assert np.allclose(
    episode.poses["left"][-1], env.rig.states(env.rig.datas[0].qpos[None])[0, 0, :7]
  )
  span = 1.0 / (NOMINAL.hz * env.cfg.clock.row_dt)
  for hand in schema.HANDS:
    times = np.asarray(episode.frame_times[hand]) / env.cfg.clock.row_dt
    assert float(np.mean(np.diff(times))) == pytest.approx(span, abs=0.05)
    assert len(times) == pytest.approx(6 * rows / span, abs=1)
    assert np.abs(times[1:4] - np.rint(times[1:4])).min() > 1e-9, (
      "a shutter falls between rows, as a capture's does"
    )
  apart = np.asarray(episode.frame_times["right"][:4]) - np.asarray(
    episode.frame_times["left"][:4]
  )
  assert np.allclose(apart, apart[0], atol=1e-9), "a pair of lenses holds its offset"
  assert decoded(episode.frames["right"][0]).shape == (32, 32, 3)


def test_the_cube_pose_is_written_with_the_quaternion_the_dataset_reads() -> None:
  """MuJoCo holds a quarter turn about the vertical as ``wxyz``; the dataset
  reads ``xyzw``, and the layer is what the dataset reads."""
  episode = Episode(
    row_hz=100.0,
    row_times=[0.0],
    cube=[np.array([0.1, 0.2, 0.3, ROOT_HALF, 0.0, 0.0, ROOT_HALF])],
  )
  batch = cube_chunk(episode).to_record_batch()
  assert cube_chunk(episode).entity_path == f"/{CUBE_POSE}"
  assert np.allclose(
    batch.column("Transform3D:quaternion").to_pylist()[0][0],
    [0.0, 0.0, ROOT_HALF, ROOT_HALF],
  )
  assert np.allclose(
    batch.column("Transform3D:translation").to_pylist()[0][0], [0.1, 0.2, 0.3]
  )


def test_a_written_layer_reads_back_as_what_was_recorded(tmp_path: Path) -> None:
  """A session reads back exactly as a capture does: hands and cube on the row
  lattice, images on the camera's grid, each labeled by the row its shutter
  fell on, both hands under one label."""
  env, episode = _episode(4)
  path = schema.layer_rrd(tmp_path, schema.state_layer(90, 100.0), "sim0000")
  write(path, "sim0000", episode, 32, 0, "test-sha", NOMINAL.hz)
  states = set(
    np.rint(np.asarray(episode.row_times) * episode.row_hz).astype(int).tolist()
  )
  labels: list[np.ndarray] = []
  shutters_seen: list[np.ndarray] = []
  with Recording(path) as seen:
    for hand in schema.HANDS:
      times, xyz = seen.values(schema.state_pose(hand), schema.POSE_TRANSLATION, 3)
      _, quat = seen.values(schema.state_pose(hand), schema.POSE_QUATERNION, 4)
      kept = np.asarray(episode.poses[hand])
      assert np.allclose(xyz, kept[:, :3], atol=1e-6)
      assert np.allclose(quat, kept[:, [4, 5, 6, 3]], atol=1e-6)
      assert np.allclose(
        np.rint(times * episode.row_hz) / episode.row_hz, episode.row_times
      )
      opening = seen.values(schema.state_opening(hand), schema.SCALARS, 1)[1][:, 0]
      assert np.allclose(opening, episode.openings[hand], atol=1e-6)
      entity = schema.state_image(hand)
      image = seen.series(f"/{entity}")
      rows = image.column(schema.ROW_TIMELINE).to_numpy()
      taken = np.concatenate(
        [
          np.asarray(one).reshape(-1)
          for one in image.column(
            schema.selector(entity, schema.CAPTURE_ROW)
          ).to_pylist()
        ]
      )
      assert set(np.diff(rows).tolist()) == {3, 4}, "images sit on the shutter's rows"
      shot = np.asarray(episode.frame_times[hand]) * episode.row_hz
      assert np.abs(shot - taken).max() <= 0.5, (
        "a hand's capture row is the one its own shutter fell nearest"
      )
      shutters_seen.append(taken)
      assert set(rows.tolist()) <= states
      labels.append(rows)
    assert np.array_equal(labels[0], labels[1]), "the hands share one row lattice"
    assert not np.array_equal(shutters_seen[0], shutters_seen[1]), (
      "and each keeps the row its own shutter fell on, as a capture does"
    )
    _, cube = seen.values(CUBE_POSE, schema.POSE_TRANSLATION, 3)
    assert len(cube) == len(episode.row_times)
    account = seen.property(schema.STATE)
    assert str(np.asarray(account["git_sha"]).reshape(-1)[0]) == "test-sha"
    assert int(np.asarray(account["seed"]).reshape(-1)[0]) == 0
    assert float(np.asarray(account["f_row"]).reshape(-1)[0]) == 100.0
    assert float(np.asarray(account["f_image"]).reshape(-1)[0]) == NOMINAL.hz
    assert int(np.asarray(account["scale"]).reshape(-1)[0]) == 90
