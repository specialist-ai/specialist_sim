"""The shared benchmark: the protocol it holds agents to, and what it scores."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.action import hold
from specialist_sim.eval import LIFT, LIFTS, PROTOCOL, Outcome, against, summary, sweep
from specialist_sim.observation import flat_rows
from specialist_sim.randomise import Kicked, Region
from specialist_sim.spaces import GRIPPERS
from specialist_sim.task import BIMANUAL, PlacedOnTape, Task, right_task

PATCH = Region(vertices=((0.30, -0.14), (0.62, -0.14), (0.62, 0.10), (0.30, 0.10)))


class _Idle:
  """An agent that stands still and remembers every reset it was told about."""

  def __init__(self) -> None:
    self.resets: list[list[tuple[int, int]]] = []

  def reset(self, worlds: Sequence[int], seeds: Sequence[int]) -> None:
    self.resets.append(list(zip(worlds, seeds, strict=True)))

  def act(self, observation: dict[str, Any]) -> np.ndarray:
    return np.stack([hold(row) for row in flat_rows(observation)])


def test_a_sweep_scores_every_seed_and_restarts_worlds_alone() -> None:
  kicked = Task(driven=GRIPPERS, start=Kicked(PATCH), goal=PlacedOnTape())
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, task=kicked), worlds=2)
  agent = _Idle()
  outcomes = sweep(env, agent, seeds=[3, 4, 5], budget=4)
  assert [outcome.seed for outcome in outcomes] == [3, 4, 5]
  assert all(outcome.steps == 4 for outcome in outcomes)
  assert all(outcome.bucket == "neither" for outcome in outcomes)
  assert agent.resets[0] == [(0, 3), (1, 4)], (
    "the opening reset names every world and seed"
  )
  assert agent.resets[1] == [(0, 5)], (
    "the restart names exactly the world that restarted, and its seed"
  )
  report = summary(outcomes, env, budget=4)
  assert report["episodes"] == 3 and report["success_rate"] == 0.0
  assert report["protocol"] == PROTOCOL
  assert (
    set(report["clean_at_lift"]) == set(LIFTS) and report["clean_at_lift"][LIFT] == 0.0
  )
  assert (report["budget"], report["camera_lag"], report["worlds"]) == (4, 0, 2)
  assert (
    report["driven"] == GRIPPERS
    and "Kicked" in report["start"]
    and len(report["revision"]) == 12
  )
  assert report["tracking"] < 0.01
  assert report["buckets"]["neither"] == 3


def test_a_sweep_refuses_a_budget_the_environment_would_cut_short() -> None:
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, max_steps=10))
  with pytest.raises(ValueError, match="truncates"):
    sweep(env, _Idle(), seeds=[0], budget=10)


def test_a_kicked_start_opens_the_episode_at_the_pick() -> None:
  task = Task(driven=GRIPPERS, start=Kicked(PATCH), goal=PlacedOnTape())
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, task=task))
  _, info = env.reset(seed=0)
  assert not bool(info["on_target"][0]), "the cube opens off the tape"
  assert PATCH.contains(info["cube_pos"][0, :2])
  _, again = env.reset(seed=0)
  assert np.array_equal(info["cube_pos"], again["cube_pos"]), "a seed names one start"


def test_an_outcome_names_its_bucket() -> None:
  carried: dict[str, Any] = dict(
    seed=0,
    dropped=False,
    steps=10,
    held_steps=5,
    lift=LIFT * 2,
    broke=False,
    tracking=0.0,
  )
  assert Outcome(success=True, **carried).bucket == "clean"
  assert Outcome(success=False, **carried).bucket == "lost"
  nudged: dict[str, Any] = dict(
    seed=0,
    dropped=False,
    steps=10,
    held_steps=0,
    lift=0.0,
    broke=False,
    tracking=0.0,
  )
  assert Outcome(success=True, **nudged).bucket == "slid"
  assert Outcome(success=False, **nudged).bucket == "neither"
  assert Outcome(success=False, **{**carried, "broke": True}).bucket == "broke"
  assert Outcome(success=True, **{**carried, "broke": True}).bucket == "broke"


def test_a_region_draws_only_inside_its_own_ring() -> None:
  region = Region(vertices=((0.40, -0.10), (0.55, -0.05), (0.52, 0.08), (0.42, 0.06)))
  ring = np.asarray(region.vertices)
  edges = np.roll(ring, -1, axis=0) - ring
  for seed in range(20):
    spot = region.draw(np.random.default_rng(seed))
    toward = spot - ring
    assert (edges[:, 0] * toward[:, 1] - edges[:, 1] * toward[:, 0] >= 0.0).all()
    again = region.draw(np.random.default_rng(seed))
    assert np.array_equal(spot, again), "a seed names one start"


def test_a_region_refuses_a_ring_that_is_not_one() -> None:
  with pytest.raises(ValueError, match="at least 3"):
    Region(vertices=((0.0, 0.0), (1.0, 0.0)))
  with pytest.raises(ValueError, match="counter-clockwise"):
    Region(vertices=((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)))


def test_a_region_never_draws_inside_its_hole() -> None:
  ring = ((0.30, -0.15), (0.65, -0.15), (0.65, 0.15), (0.30, 0.15))
  region = Region(vertices=ring, hole=(0.49, 0.0, 0.08))
  for seed in range(200):
    spot = region.draw(np.random.default_rng(seed))
    assert np.hypot(spot[0] - 0.49, spot[1]) >= 0.08
  with pytest.raises(ValueError, match="not negative"):
    Region(vertices=ring, hole=(0.49, 0.0, -0.01))


def test_a_bimanual_reset_is_untouched_by_the_task_machinery() -> None:
  """The default task opens as every recorded episode does: on the tape."""
  env = PickPlaceEnv(PickPlaceCfg(image_size=64, task=BIMANUAL))
  _, info = env.reset(seed=0)
  assert bool(info["on_target"][0])
  assert right_task(
    Region(((0.3, -0.1), (0.6, -0.1), (0.6, 0.1), (0.3, 0.1)))
  ).driven == ("right_gripper",)


def test_the_bench_has_no_start_of_its_own() -> None:
  env = PickPlaceEnv(PickPlaceCfg(image_size=64))
  with pytest.raises(TypeError):
    sweep(env, _Idle(), seeds=[0], budget=1, starts=PATCH)  # type: ignore[call-arg]


def test_an_agent_is_measured_against_the_expert_where_the_expert_drives() -> None:
  env = PickPlaceEnv(PickPlaceCfg(image_size=64))
  gaps = against(env, _Idle(), driven=GRIPPERS, seeds=[0, 1], budget=6)
  assert len(gaps) == 24 and all(np.isfinite(gaps))
  assert max(gaps) > 0.0, "a hand that holds still is not what the expert commands"
