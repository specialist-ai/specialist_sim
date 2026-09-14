"""The bench holds any scene to one protocol: a budget, restarts one world at a
time, and an outcome per seed."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from specialist_sim.bench import summary, sweep
from specialist_sim.rig import Observation
from specialist_sim.trajectory import hold


@dataclass
class _Report:
  success: np.ndarray
  done: np.ndarray
  broke: np.ndarray
  tracking: np.ndarray
  labels: dict[str, np.ndarray]


class _Scene:
  """Two worlds that each succeed after as many steps as their seed, or never."""

  cfg = "stub"
  assets = ()

  def __init__(self) -> None:
    self.worlds = 2
    self.seeds = np.zeros(2, np.int64)
    self.steps = np.zeros(2, np.int64)

  def _observe(self) -> Observation:
    return Observation(
      row=np.zeros((2, 1), np.int64),
      state=np.zeros((2, 1, 1, 8)),
      image=np.zeros((2, 1, 2, 2, 3), np.uint8),
      body=np.zeros((2, 1, 2, 2), np.int32),
      image_row=np.zeros(2, np.int64),
    )

  def _report(self) -> _Report:
    won = (self.steps >= self.seeds) & (self.seeds > 0)
    return _Report(
      success=won,
      done=won,
      broke=np.zeros(2, bool),
      labels={"carried": self.steps > 1},
      tracking=np.full(2, 0.001),
    )

  def reset(self, seeds: Sequence[int], worlds: Sequence[int] | None = None) -> Any:
    chosen = list(range(2)) if worlds is None else list(worlds)
    self.seeds[chosen] = seeds
    self.steps[chosen] = 0
    return self._observe(), self._report()

  def step(self, action: np.ndarray) -> Any:
    self.steps += 1
    return self._observe(), self._report()


class _Idle:
  def __init__(self) -> None:
    self.resets: list[list[tuple[int, int]]] = []

  def reset(self, worlds: Sequence[int], seeds: Sequence[int]) -> None:
    self.resets.append(list(zip(worlds, seeds, strict=True)))

  def act(self, observation: Observation) -> np.ndarray:
    return hold(observation.state[:, -1])


def test_a_sweep_scores_every_seed_and_restarts_worlds_alone() -> None:
  scene, agent = _Scene(), _Idle()
  outcomes = sweep(scene, agent, seeds=[3, 1, 0], budget=4)
  assert [(o.seed, o.success, o.steps, dict(o.labels)) for o in outcomes] == [
    (0, False, 4, {"carried": True}),
    (1, True, 1, {"carried": False}),
    (3, True, 3, {"carried": True}),
  ]
  assert agent.resets == [[(0, 3), (1, 1)], [(1, 0)]]
  report = summary(outcomes, scene, budget=4)
  assert report["episodes"] == 3 and report["budget"] == 4 and report["worlds"] == 2
  assert report["success_rate"] == 2 / 3 and report["carried_rate"] == 2 / 3
  assert report["steps_to_success"] == 2.0 and report["tracking"] == 0.001
  assert len(report["revision"]) == 12 and report["cfg"] == "'stub'"


def test_a_scene_names_what_the_bench_counts() -> None:
  """The bench rates whatever a scene labels and knows nothing of what it
  means; only the scene reads its own names back."""
  scene, agent = _Scene(), _Idle()
  outcomes = sweep(scene, agent, seeds=[3, 1, 0], budget=4)
  assert all(set(outcome.labels) == {"carried"} for outcome in outcomes)
  assert bucket_of(outcomes) == ["lost", "slid", "clean"]


def bucket_of(outcomes: Any) -> list[str]:
  """What a scene would make of them, as ``pickplace`` does."""
  return [
    "broke"
    if outcome.broke
    else ("clean" if outcome.labels["carried"] else "slid")
    if outcome.success
    else ("lost" if outcome.labels["carried"] else "neither")
    for outcome in outcomes
  ]
