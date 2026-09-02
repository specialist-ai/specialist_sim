"""The protocol's promise, enforced: a fixed seed opens and runs the same."""

import hashlib

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.action import hold
from specialist_sim.eval import PROTOCOL
from specialist_sim.observation import flat_row
from specialist_sim.randomise import Region
from specialist_sim.task import BIMANUAL, Task, left_task, right_task

PATCH = Region(
  vertices=((0.30, -0.14), (0.62, -0.14), (0.62, 0.10), (0.30, 0.10)),
  hole=(0.49, 0.0, 0.08),
)
GOLDEN = {
  3: {
    "bimanual": ("1d08b1f9e19ea55d", "46f86ac71b8dc782"),
    "right": ("1bdb6d32f6852148", "840b40a40e3f8e79"),
    "left": ("1d08b1f9e19ea55d", "46f86ac71b8dc782"),
  }
}
"""Per protocol, per named task: the plant as seed 0 opens it, and after five
held steps rounded to a tenth of a millimetre. A hash that moves without a
protocol bump is an outcome that moved for a fixed seed."""


def _digest(qpos: np.ndarray, decimals: int | None = None) -> str:
  arr = np.asarray(qpos, np.float64)
  if decimals is not None:
    arr = np.round(arr, decimals)
  return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


@pytest.mark.parametrize(
  ("name", "task"),
  [("bimanual", BIMANUAL), ("right", right_task(PATCH)), ("left", left_task(PATCH))],
)
def test_seed_zero_opens_and_runs_as_the_protocol_says(name: str, task: Task) -> None:
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, task=task))
  obs, _ = env.reset(seed=0)
  opened = _digest(env.datas[0].qpos)
  for _ in range(5):
    obs, *_ = env.step(hold(flat_row(obs)))
  held = _digest(env.datas[0].qpos, 4)
  assert (opened, held) == GOLDEN[PROTOCOL][name], (
    f"{name} moved for a fixed seed: bump PROTOCOL and pin the new hashes"
  )
