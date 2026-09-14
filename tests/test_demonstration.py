"""The pick-and-place demonstration: does it finish the task, and does it
answer to the scene rather than to a clock."""

import numpy as np
import pytest

from specialist_sim.expert import Expert, ExpertCfg
from specialist_sim.pickplace import Cfg, PickPlace, Report
from specialist_sim.pickplace.script import STAGES, expert
from specialist_sim.trajectory import HORIZON

ONCE = ExpertCfg(length=(0.0, 0.0))
"""One cycle, which is all a test needs to watch."""


def _run(seed: int, steps: int = 400) -> tuple[PickPlace, Expert, Report]:
  env = PickPlace(Cfg(image_size=32))
  observation, report = env.reset([seed])
  demonstrator = expert(env, np.random.default_rng(seed), ONCE)
  for _ in range(steps):
    action = demonstrator.act(observation.state[0, -1], report.cube[0])
    observation, report = env.step(action[None])
    if demonstrator.done:
      break
  return env, demonstrator, report


@pytest.mark.parametrize("seed", [0, 3, 4, 7])
def test_the_demonstration_puts_the_cube_on_the_tape(seed: int) -> None:
  env, demonstrator, report = _run(seed)
  assert demonstrator.done, f"the demonstration stalled at stage {demonstrator.stage}"
  off = np.hypot(*(report.cube[0, :2] - env.room.tape)) * 1000
  assert report.success[0], f"the cube ended {off:.1f} mm from the tape"
  assert demonstrator.cycles == 1, (
    f"the cube took {demonstrator.cycles} cycles to place"
  )
  assert report.picked[0], "the cube was carried, not nudged"


def test_the_kick_sweeps_the_cube_off_the_tape() -> None:
  """The perturbation is the point: the pick is given something to solve."""
  env = PickPlace(Cfg(image_size=32))
  observation, report = env.reset([1])
  demonstrator = expert(env, np.random.default_rng(1), ONCE)
  start = report.cube[0].copy()
  scattered = None
  for _ in range(400):
    if demonstrator.done:
      break
    if scattered is None and STAGES[demonstrator.stage].name == "reach":
      scattered = report.cube[0].copy()
    action = demonstrator.act(observation.state[0, -1], report.cube[0])
    observation, report = env.step(action[None])
  assert scattered is not None, "the demonstration never reached the pick"
  began = float(np.hypot(*(start[:2] - env.room.tape)))
  swept = float(np.hypot(*(scattered[:2] - env.room.tape)))
  assert began < 0.01, f"the cube began {began * 1000:.1f} mm off the tape"
  assert swept > 0.03, f"the kick left the cube {swept * 1000:.1f} mm off the tape"


def test_a_demonstration_commands_a_continuous_trajectory() -> None:
  """A trajectory without jumps, turning through unit quaternions, walking no
  faster than the speed it is given."""
  env = PickPlace(Cfg(image_size=32))
  observation, report = env.reset([2])
  demonstrator = expert(env, np.random.default_rng(2), ONCE)
  last = None
  for _ in range(40):
    action = demonstrator.act(observation.state[0, -1], report.cube[0])
    assert action.shape == (HORIZON, 2, 8)
    assert np.allclose(np.linalg.norm(action[:, :, 3:7], axis=-1), 1.0, atol=1e-9)
    stride = np.linalg.norm(np.diff(action[:, :, :3], axis=0), axis=-1)
    assert (stride <= ONCE.speed * demonstrator.period + 1e-9).all()
    if last is not None:
      step = action[0, :, :3] - last[demonstrator.tracked - 1, :, :3]
      assert np.linalg.norm(step, axis=-1).max() < 0.05, "the trajectory jumped"
    last = action.copy()
    observation, report = env.step(action[None])


def test_an_episode_repeats_the_cycle_until_its_span_is_up() -> None:
  env = PickPlace(Cfg(image_size=32))
  observation, report = env.reset([5])
  demonstrator = expert(env, np.random.default_rng(5), ExpertCfg(length=(12.0, 12.0)))
  steps = 0
  while not demonstrator.done and steps < 400:
    action = demonstrator.act(observation.state[0, -1], report.cube[0])
    observation, report = env.step(action[None])
    steps += 1
  assert demonstrator.done, (
    f"the episode never finished, reaching {demonstrator.cycles} cycles"
  )
  assert demonstrator.cycles >= 2, f"the episode held only {demonstrator.cycles} cycles"
  ran = steps * env.cfg.clock.step_dt
  assert 12.0 <= ran <= 24.0, f"the episode ran {ran:.1f} s"
  assert report.success[0]
