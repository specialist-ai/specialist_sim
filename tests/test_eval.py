"""The shared benchmark: the protocol it holds agents to, and what it scores."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.action import hold
from specialist_sim.eval import LIFT, Outcome, Region, Scatter, summary, sweep
from specialist_sim.observation import flat_rows
from specialist_sim.randomise import scattered


class _Idle:
  """An agent that stands still and remembers every reset it was told about."""

  def __init__(self) -> None:
    self.resets: list[list[int]] = []

  def reset(self, worlds: Sequence[int]) -> None:
    self.resets.append(list(worlds))

  def act(self, observation: dict[str, Any]) -> np.ndarray:
    return np.stack([hold(row) for row in flat_rows(observation)])


def test_a_sweep_scores_every_seed_and_restarts_worlds_alone() -> None:
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, max_steps=10**6), worlds=2)
  agent = _Idle()
  outcomes = sweep(env, agent, seeds=[3, 4, 5], budget=4, starts=Scatter())
  assert [outcome.seed for outcome in outcomes] == [3, 4, 5]
  assert all(outcome.steps == 4 for outcome in outcomes)
  assert all(outcome.bucket == "neither" for outcome in outcomes)
  assert agent.resets[0] == [0, 1], "the opening reset names every world"
  assert len(agent.resets) == 2 and len(agent.resets[1]) == 1, (
    "the one restart mid-batch names exactly the world that restarted"
  )
  report = summary(outcomes, env, budget=4)
  assert report["episodes"] == 3 and report["success_rate"] == 0.0
  assert (report["budget"], report["camera_lag"], report["worlds"]) == (4, 0, 2)
  assert report["buckets"]["neither"] == 3


def test_a_sweep_refuses_a_budget_the_environment_would_cut_short() -> None:
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, max_steps=10))
  with pytest.raises(ValueError, match="truncates"):
    sweep(env, _Idle(), seeds=[0], budget=10)


def test_a_scattered_start_opens_the_episode_at_the_pick() -> None:
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, max_steps=10**6))
  starts = Scatter()
  tape = np.asarray(env.scene.tape)
  for seed in (0, 1, 2):
    spot = scattered(env.scene, np.random.default_rng(seed), starts)
    away = float(np.hypot(*(spot - tape)))
    assert starts.distance[0] <= away <= starts.distance[1]
    again = scattered(env.scene, np.random.default_rng(seed), starts)
    assert np.array_equal(spot, again), "a seed names one start"
  _, info = env.reset(seed=0, options={"cube": spot})
  assert not bool(info["on_target"][0]), "the cube opens off the tape"


def test_an_outcome_names_its_bucket() -> None:
  base = dict(seed=0, dropped=False, steps=10, held_steps=5, lift=LIFT * 2)
  assert Outcome(success=True, **base).bucket == "clean"
  assert Outcome(success=False, **base).bucket == "lost"
  nudged = dict(seed=0, dropped=False, steps=10, held_steps=0, lift=0.0)
  assert Outcome(success=True, **nudged).bucket == "slid"
  assert Outcome(success=False, **nudged).bucket == "neither"


def test_a_region_draws_only_inside_its_own_ring() -> None:
  region = Region(vertices=((0.40, -0.10), (0.55, -0.05), (0.52, 0.08), (0.42, 0.06)))
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, max_steps=10**6))
  ring = np.asarray(region.vertices)
  edges = np.roll(ring, -1, axis=0) - ring
  for seed in range(20):
    spot = region.draw(env.scene, np.random.default_rng(seed))
    toward = spot - ring
    assert (edges[:, 0] * toward[:, 1] - edges[:, 1] * toward[:, 0] >= 0.0).all()
    again = region.draw(env.scene, np.random.default_rng(seed))
    assert np.array_equal(spot, again), "a seed names one start"


def test_a_region_refuses_a_ring_that_is_not_one() -> None:
  with pytest.raises(ValueError, match="at least 3"):
    Region(vertices=((0.0, 0.0), (1.0, 0.0)))
  with pytest.raises(ValueError, match="counter-clockwise"):
    Region(vertices=((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)))


def test_a_region_never_draws_inside_its_hole() -> None:
  ring = ((0.30, -0.15), (0.65, -0.15), (0.65, 0.15), (0.30, 0.15))
  region = Region(vertices=ring, hole=(0.49, 0.0, 0.08))
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, max_steps=10**6))
  for seed in range(200):
    spot = region.draw(env.scene, np.random.default_rng(seed))
    assert np.hypot(spot[0] - 0.49, spot[1]) >= 0.08
  with pytest.raises(ValueError, match="not negative"):
    Region(vertices=ring, hole=(0.49, 0.0, -0.01))
