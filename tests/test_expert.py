"""The rule-based demonstration: does it finish the task, and does it answer to
the scene rather than to a clock."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.expert import _PEAK, _STAGES, PICKING, Expert, ExpertCfg, _Move
from specialist_sim.expert_cfg import PARKED, ROVING
from specialist_sim.observation import flat_row
from specialist_sim.spaces import HORIZON, PER_HAND

ONCE = ExpertCfg(length=(0.0, 0.0))
"""One cycle, which is all a test needs to watch."""


def _run(seed: int, steps: int = 600) -> tuple[PickPlaceEnv, Expert, dict[str, Any]]:
  env = PickPlaceEnv(PickPlaceCfg(image_size=64))
  observation, info = env.reset(seed=seed)
  expert = Expert(env, np.random.default_rng(seed), ONCE)
  for _ in range(steps):
    action = expert.act(flat_row(observation), info["cube_pos"][0])
    observation, _, _, _, info = env.step(action)
    if expert.done:
      break
  return env, expert, info


@pytest.mark.parametrize("seed", [0, 3])
def test_the_demonstration_puts_the_cube_on_the_tape(seed: int) -> None:
  env, expert, info = _run(seed)
  assert expert.done, f"the demonstration stalled at stage {expert.stage}"
  assert info["is_success"][0], (
    f"the cube ended {np.hypot(*(info['cube_pos'][0][:2] - env.scene.tape)) * 1000:.1f} mm "
    "from the tape"
  )


def test_the_kick_sweeps_the_cube_off_the_tape() -> None:
  """The perturbation is the point: the pick is given something to solve."""
  env = PickPlaceEnv(PickPlaceCfg(image_size=64))
  observation, info = env.reset(seed=1)
  expert = Expert(env, np.random.default_rng(1), ONCE)
  start = np.asarray(info["cube_pos"][0], np.float64).copy()
  scattered = None
  for _ in range(600):
    if expert.done:
      break
    if scattered is None and _STAGES[expert.stage].name == PICKING:
      scattered = np.asarray(info["cube_pos"][0], np.float64).copy()
    action = expert.act(flat_row(observation), info["cube_pos"][0])
    observation, _, _, _, info = env.step(action)
  assert scattered is not None, "the demonstration never reached the pick"
  began = float(np.hypot(*(start[:2] - env.scene.tape)))
  swept = float(np.hypot(*(scattered[:2] - env.scene.tape)))
  assert began < 0.01, f"the cube began {began * 1000:.1f} mm off the tape"
  assert swept > 0.03, f"the kick left the cube {swept * 1000:.1f} mm off the tape"


def test_a_demonstration_commands_a_continuous_trajectory() -> None:
  """Bounds the interface clips to say nothing; what a demonstration owes is
  a trajectory without jumps, turning through unit quaternions."""
  env = PickPlaceEnv(PickPlaceCfg(image_size=64))
  observation, info = env.reset(seed=2)
  expert = Expert(env, np.random.default_rng(2), ONCE)
  last = None
  for _ in range(40):
    action = expert.act(flat_row(observation), info["cube_pos"][0])
    assert action.shape == (HORIZON, PER_HAND * 2)
    for hand in range(2):
      quat = action[:, hand * PER_HAND + 3 : hand * PER_HAND + 7]
      assert np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-9)
    if last is not None:
      step = action[0, :3] - last[expert.tracked - 1, :3]
      assert np.linalg.norm(step) < 0.05, "the trajectory jumped between calls"
    last = action.copy()
    observation, _, _, _, info = env.step(action)


def test_the_demonstration_walks_rather_than_jumps() -> None:
  """A step of the commanded trajectory is bounded by the speed it is given."""
  env = PickPlaceEnv(PickPlaceCfg(image_size=64))
  observation, info = env.reset(seed=4)
  cfg = ONCE
  expert = Expert(env, np.random.default_rng(4), cfg)
  for _ in range(30):
    action = expert.act(flat_row(observation), info["cube_pos"][0])
    for hand in (0, 1):
      at = hand * PER_HAND
      stride = np.linalg.norm(np.diff(action[:, at : at + 3], axis=0), axis=1)
      assert (stride <= cfg.speed / 50.0 + 1e-9).all()
    observation, _, _, _, info = env.step(action)


def test_a_move_starts_and_stops_without_a_jolt() -> None:
  """A stage's curve ends at rest with no acceleration left over, which is what
  lets the next stage begin without a corner."""
  start = np.array([0.0, 0.0, 0.0])
  goal = np.array([0.2, -0.1, 0.05])
  span = 0.8
  move = _Move.plan(start, np.zeros(3), goal, span)
  assert np.allclose(move.at(0.0), start)
  assert np.allclose(move.at(span), goal)
  assert np.allclose(move.rate_at(0.0), 0.0)
  assert np.allclose(move.rate_at(span), 0.0)
  nudge = 1e-4
  for edge in (nudge, span - nudge):
    rise = move.rate_at(edge + nudge) - move.rate_at(edge - nudge)
    assert np.abs(rise / (2.0 * nudge)).max() < 1e-2, "the curve jolts at an end"
  fastest = max(
    float(np.linalg.norm(move.rate_at(span * part / 100.0))) for part in range(101)
  )
  reach = float(np.linalg.norm(goal - start))
  assert fastest <= _PEAK * reach / span + 1e-9


def test_an_episode_repeats_the_cycle_until_its_span_is_up() -> None:
  """A recording holds many cycles, not one."""
  env = PickPlaceEnv(PickPlaceCfg(image_size=64))
  observation, info = env.reset(seed=5)
  expert = Expert(env, np.random.default_rng(5), ExpertCfg(length=(20.0, 20.0)))
  steps = 0
  while not expert.done and steps < 600:
    action = expert.act(flat_row(observation), info["cube_pos"][0])
    observation, _, _, _, info = env.step(action)
    steps += 1
  assert expert.done, f"the episode never finished, reaching {expert.cycles} cycles"
  assert expert.cycles >= 3, f"the episode held only {expert.cycles} cycles"
  ran = steps / env.cfg.control_hz
  assert 20.0 <= ran <= 32.0, f"the episode ran {ran:.1f} s"


def test_a_finished_demonstration_still_names_an_angle() -> None:
  """A demonstration that has run out holds the pose it ended on, and names its
  opening the way every other trajectory does.

  The bounds alone cannot catch this: the fraction the rules reason in lands
  inside the angle's own travel, so it has to be read against what came before.
  """
  env = PickPlaceEnv(PickPlaceCfg(image_size=64))
  observation, info = env.reset(seed=3)
  expert = Expert(env, np.random.default_rng(3), ONCE)
  before = None
  for _ in range(600):
    action = expert.act(flat_row(observation), info["cube_pos"][0])
    if expert.done:
      break
    before = action
    observation, _, _, _, info = env.step(action)
  assert before is not None, "the demonstration finished before it began"
  fingers = slice(PER_HAND - 1, None, PER_HAND)
  assert np.allclose(action[:, fingers], before[expert.tracked - 1, fingers])


def _held(cfg: ExpertCfg, seed: int, steps: int = 150) -> float:
  """The share of commanded steps on which the kicking hand does not move.

  It is the hand that is idle for most of a cycle, so it is where the two
  tunings part: one leaves it where the strike put it, the other keeps it
  traveling.
  """
  env = PickPlaceEnv(PickPlaceCfg(image_size=64))
  observation, info = env.reset(seed=seed)
  expert = Expert(env, np.random.default_rng(seed), replace(cfg, length=(0.0, 0.0)))
  still, seen, last = 0, 0, None
  for _ in range(steps):
    action = expert.act(flat_row(observation), info["cube_pos"][0])
    at = action[expert.tracked - 1, :3]
    if last is not None:
      seen += 1
      still += float(np.linalg.norm(at - last)) < 1e-6
    last = at.copy()
    observation, _, _, _, info = env.step(action)
    if expert.done:
      break
  return still / seen


@pytest.mark.parametrize("seed", [6, 7, 8])
def test_a_parked_hand_stands_still_where_a_roving_one_never_does(seed: int) -> None:
  """The two tunings differ in what a hand does when no stage is driving it,
  and that difference is most of what a policy reads off the dataset: a hand
  told to wait repeats one setpoint exactly, which teaches an instruction not
  to move. Parked holds it over half the time, which is what the first dataset
  measured; roving all but never does.
  """
  assert PARKED.loiter is None and ROVING.loiter is not None
  assert _held(ROVING, seed) < 0.05, "a roving hand came to a stop"
  assert _held(PARKED, seed) > 0.40, "a parked hand kept moving"


@pytest.mark.parametrize("tuning", [ROVING, PARKED])
def test_either_tuning_puts_the_cube_on_the_tape(tuning: ExpertCfg) -> None:
  """A tuning that cannot finish the task is not an option, whatever else it
  gets right about how a hand moves."""
  env = PickPlaceEnv(PickPlaceCfg(image_size=64))
  observation, info = env.reset(seed=1)
  expert = Expert(env, np.random.default_rng(1), replace(tuning, length=(0.0, 0.0)))
  for _ in range(600):
    action = expert.act(flat_row(observation), info["cube_pos"][0])
    observation, _, _, _, info = env.step(action)
    if expert.done:
      break
  assert expert.done, f"the demonstration stalled at stage {expert.stage}"
  assert info["is_success"][0], (
    f"the cube ended {np.hypot(*(info['cube_pos'][0][:2] - env.scene.tape)) * 1000:.1f} mm "
    "from the tape"
  )
