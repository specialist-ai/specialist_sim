"""The two plants: that they agree, and that the CPU one is reproducible."""

from typing import Any

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.observation import flat_rows
from specialist_sim.spaces import HORIZON

STEPS = 5


def _held(observation: dict[str, Any]) -> np.ndarray:
  """A trajectory asking every world to stay where the last row left it."""
  return np.repeat(flat_rows(observation)[:, None], HORIZON, axis=1)


def _played(env: PickPlaceEnv, seeds: list[int]) -> np.ndarray:
  """Every world's joint coordinates after a held pose is tracked for a while."""
  observation, _ = env.reset(seeds=seeds)
  action = _held(observation)
  for _ in range(STEPS):
    observation, _, _, _, _ = env.step(action)
  return np.stack([data.qpos.copy() for data in env.datas])


def _on_gpu() -> bool:
  try:
    import mujoco_warp  # noqa: F401
    import warp as wp

    wp.get_device("cuda:0")
  except Exception:
    return False
  return True


needs_gpu = pytest.mark.skipif(not _on_gpu(), reason="mujoco_warp needs a CUDA device")


def test_a_run_repeats_itself_exactly_on_the_cpu() -> None:
  seeds = list(range(4))
  first = _played(PickPlaceEnv(worlds=4), seeds)
  again = _played(PickPlaceEnv(worlds=4), seeds)
  assert np.array_equal(first, again)


def test_a_world_does_not_notice_how_many_it_is_batched_with() -> None:
  """What a final evaluation is bought with: a score cannot turn on batch size."""
  alone = _played(PickPlaceEnv(worlds=1), [0])
  batched = _played(PickPlaceEnv(worlds=4), list(range(4)))
  assert np.array_equal(alone[0], batched[0])


@needs_gpu
def test_the_two_plants_track_the_same_pose_to_the_precision_they_carry() -> None:
  seeds = list(range(4))
  on_cpu = _played(PickPlaceEnv(worlds=4), seeds)
  on_gpu = _played(PickPlaceEnv(worlds=4, physics="gpu"), seeds)
  # The device carries float32 against the host's float64, so they part in the
  # last bits and contact makes the gap grow; a held pose keeps it this small.
  assert np.abs(on_cpu - on_gpu).max() < 1e-4


@needs_gpu
def test_the_gpu_reports_what_every_world_carries() -> None:
  env = PickPlaceEnv(worlds=3, physics="gpu")
  observation, _ = env.reset(seeds=[0, 1, 2])
  observation, reward, terminated, truncated, report = env.step(_held(observation))
  assert observation["left_gripper"]["image"].shape == (3, *env.image_shape)
  assert observation["left_gripper"]["pose"].shape == (3, env.rows, 7)
  assert reward.shape == terminated.shape == truncated.shape == (3,)
  assert report["cube_pos"].shape == (3, 3)


@needs_gpu
def test_the_gpu_refuses_a_lag_it_cannot_read_the_lenses_on() -> None:
  with pytest.raises(ValueError, match="whole number of rows"):
    PickPlaceEnv(PickPlaceCfg(camera_lag=5), physics="gpu")


def test_an_unknown_backend_is_refused() -> None:
  with pytest.raises(ValueError, match="not one of"):
    PickPlaceEnv(physics="tpu")
