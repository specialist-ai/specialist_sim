"""The layer a demonstration is written as: what it holds of the scene, and in
whose order it holds it."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("specialist_corpus", reason="the dataset extra is not installed")

from specialist_corpus import schema  # noqa: E402
from specialist_corpus.read import Recording  # noqa: E402

from specialist_sim import PickPlaceCfg, PickPlaceEnv  # noqa: E402
from specialist_sim.action import hold  # noqa: E402
from specialist_sim.dataset.state import (  # noqa: E402
  CUBE_POSE,
  IMAGE_STRIDE,
  Episode,
  cube_chunk,
  layer_scale,
  rollout,
  write_layer,
)
from specialist_sim.expert import Expert, ExpertCfg  # noqa: E402
from specialist_sim.observation import flat_row  # noqa: E402
from specialist_sim.spaces import GRIPPERS  # noqa: E402

ROOT_HALF = np.sqrt(0.5)
"""A quarter turn about the vertical, as either convention spells its halves."""


def test_the_layer_holds_the_cube_at_every_step_the_hands_are_held_at() -> None:
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, max_steps=10**6))
  expert = Expert(env, np.random.default_rng(0), ExpertCfg(length=(0.0, 0.0)))
  episode = rollout(env, expert, seed=0, max_steps=40)
  cube = np.asarray(episode.cube, np.float64)
  assert cube.shape == (len(episode.step_times), 7)
  at = env.scene.cube.qpos
  assert np.allclose(cube[-1], env.scene.data.qpos[at : at + 7])


def test_the_cube_pose_is_written_with_the_quaternion_the_dataset_reads() -> None:
  """MuJoCo holds a quarter turn about the vertical as ``wxyz``; the dataset
  reads ``xyzw``, and the layer is what the dataset reads."""
  episode = Episode(
    step_times=[0.0],
    cube=[np.array([0.1, 0.2, 0.3, ROOT_HALF, 0.0, 0.0, ROOT_HALF])],
  )
  chunk = cube_chunk(episode)
  batch = chunk.to_record_batch()
  assert chunk.entity_path == f"/{CUBE_POSE}"
  assert np.allclose(
    batch.column("Transform3D:quaternion").to_pylist()[0][0],
    [0.0, 0.0, ROOT_HALF, ROOT_HALF],
  )
  assert np.allclose(
    batch.column("Transform3D:translation").to_pylist()[0][0], [0.1, 0.2, 0.3]
  )


def test_a_written_layer_reads_back_as_what_was_recorded(tmp_path: Path) -> None:
  """The one hard constraint: a session must read back exactly as a capture
  does, hands and cube on the 100 Hz lattice, quaternions in the dataset's
  own order."""
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, max_steps=10**6))
  expert = Expert(env, np.random.default_rng(0), ExpertCfg(length=(0.0, 0.0)))
  episode = rollout(env, expert, seed=0, max_steps=30)
  scale = layer_scale(env.cameras[GRIPPERS[0]].width, 64)
  layer = schema.state_layer(scale, IMAGE_STRIDE)
  path = schema.layer_rrd(tmp_path, layer, "sim0000")
  write_layer(path, "sim0000", episode, 64, 0, scale, 0, "test-sha")

  with Recording(path) as seen:
    for hand in schema.HANDS:
      times, xyz = seen.values(schema.state_pose(hand), schema.POSE_TRANSLATION, 3)
      _, quat = seen.values(schema.state_pose(hand), schema.POSE_QUATERNION, 4)
      kept = np.asarray(episode.poses[hand], np.float64)
      assert np.allclose(xyz, kept[:, :3], atol=1e-6)
      assert np.allclose(quat, kept[:, [4, 5, 6, 3]], atol=1e-6), (
        "the wxyz the plant holds must land as the xyzw the dataset reads"
      )
      rows = np.rint(np.asarray(times) * 100.0)
      assert np.allclose(rows / 100.0, episode.step_times, atol=1e-9), (
        "state times must sit exactly on the 100 Hz lattice"
      )
      opening = seen.values(schema.state_opening(hand), schema.SCALARS, 1)[1][:, 0]
      assert np.allclose(opening, episode.openings[hand], atol=1e-6)
    _, cube = seen.values(CUBE_POSE, schema.POSE_TRANSLATION, 3)
    assert len(cube) == len(episode.step_times)
    account = seen.row(schema.STATE)
    assert str(np.asarray(account["git_sha"]).reshape(-1)[0]) == "test-sha"
    assert int(np.asarray(account["seed"]).reshape(-1)[0]) == 0


def test_a_watcher_sees_every_substep_and_changes_nothing() -> None:
  cfg = PickPlaceCfg(image_size=64, max_steps=10**6)
  env = PickPlaceEnv(cfg, worlds=2)
  obs, _ = env.reset(seed=0)
  action = np.stack([hold(flat_row(obs, world=w)) for w in range(2)])
  seen: list[tuple[int, np.ndarray, np.ndarray]] = []

  def watch(substep: int, qpos: np.ndarray, qvel: np.ndarray) -> None:
    seen.append((substep, qpos.copy(), qvel.copy()))

  watched, *_ = env.step(action, watch=watch)
  assert [s for s, _, _ in seen] == list(range(env.ramp.substeps))
  model = env.scene.model
  assert seen[-1][1].shape == (2, model.nq) and seen[-1][2].shape == (2, model.nv)
  for world, data in enumerate(env.datas):
    assert np.array_equal(seen[-1][1][world], data.qpos)
    assert np.array_equal(seen[-1][2][world], data.qvel)

  quiet = PickPlaceEnv(cfg, worlds=2)
  obs, _ = quiet.reset(seed=0)
  unwatched, *_ = quiet.step(action)
  for slot in GRIPPERS:
    assert np.array_equal(watched[slot]["pose"], unwatched[slot]["pose"])
    assert np.array_equal(watched[slot]["image"], unwatched[slot]["image"])
