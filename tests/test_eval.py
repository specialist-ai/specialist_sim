"""Evaluation holds any env to one protocol: a time limit, restarts one env at a
time, and an outcome per seed."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from specialist_sim.action import hold
from specialist_sim.eval import eval_policy, summary
from specialist_sim.rig import Observation


@dataclass
class _Report:
  success: np.ndarray
  done: np.ndarray
  unstable: np.ndarray
  tracking_error: np.ndarray
  progress: np.ndarray
  labels: dict[str, np.ndarray]


class _Env:
  """Two envs that each succeed after as many steps as their seed, or never."""

  cfg = "stub"
  assets = ()

  def __init__(self) -> None:
    self.num_envs = 2
    self.seeds = np.zeros(2, np.int64)
    self.steps = np.zeros(2, np.int64)

  def _observe(self) -> Observation:
    return Observation(
      row=np.zeros((2, 1), np.int64),
      state=np.zeros((2, 1, 1, 8)),
      image=np.zeros((2, 1, 1, 2, 2, 3), np.uint8),
      body=np.zeros((2, 1, 1, 2, 2), np.int32),
      image_row=np.zeros((2, 1, 1), np.int64),
    )

  def _report(self) -> _Report:
    won = (self.steps >= self.seeds) & (self.seeds > 0)
    return _Report(
      success=won,
      done=won,
      unstable=np.zeros(2, bool),
      labels={"carried": self.steps > 1},
      tracking_error=np.full(2, 0.001),
      progress=np.where(won, 1.0, 0.5),
    )

  def reset(self, seeds: Sequence[int], env_ids: Sequence[int] | None = None) -> Any:
    chosen = list(range(2)) if env_ids is None else list(env_ids)
    self.seeds[chosen] = seeds
    self.steps[chosen] = 0
    return self._observe(), self._report()

  def step(self, action: np.ndarray) -> Any:
    self.steps += 1
    return self._observe(), self._report()


class _Idle:
  def __init__(self) -> None:
    self.resets: list[list[tuple[int, int]]] = []

  def reset(self, env_ids: Sequence[int], seeds: Sequence[int]) -> None:
    self.resets.append(list(zip(env_ids, seeds, strict=True)))

  def predict_action_chunk(self, observation: Observation) -> np.ndarray:
    return hold(observation.state[:, -1])


def test_eval_policy_scores_every_seed_and_restarts_envs_alone() -> None:
  env, policy = _Env(), _Idle()
  outcomes = eval_policy(env, policy, seeds=[3, 1, 0], max_episode_steps=4)
  assert [(o.seed, o.success, o.steps, dict(o.labels)) for o in outcomes] == [
    (0, False, 4, {"carried": True}),
    (1, True, 1, {"carried": False}),
    (3, True, 3, {"carried": True}),
  ]
  assert policy.resets == [[(0, 3), (1, 1)], [(1, 0)]]
  report = summary(outcomes, env, max_episode_steps=4)
  assert (
    report["n_episodes"] == 3
    and report["max_episode_steps"] == 4
    and report["num_envs"] == 2
  )
  assert report["success_rate"] == 2 / 3 and report["carried_rate"] == 2 / 3
  assert report["progress"] == (1.0 + 1.0 + 0.5) / 3
  assert report["steps_to_success"] == 2.0 and report["tracking_error"] == 0.001
  assert len(report["revision"]) == 12 and report["cfg"] == "'stub'"


def test_an_env_names_what_evaluation_counts() -> None:
  """Evaluation rates whatever an env labels and knows nothing of what it
  means; only the env reads its own names back."""
  env, policy = _Env(), _Idle()
  outcomes = eval_policy(env, policy, seeds=[3, 1, 0], max_episode_steps=4)
  assert all(set(outcome.labels) == {"carried"} for outcome in outcomes)
  assert bucket_of(outcomes) == ["lost", "slid", "clean"]


def bucket_of(outcomes: Any) -> list[str]:
  """What an env would make of them, as ``pickplace`` does."""
  return [
    "unstable"
    if outcome.unstable
    else ("clean" if outcome.labels["carried"] else "slid")
    if outcome.success
    else ("lost" if outcome.labels["carried"] else "neither")
    for outcome in outcomes
  ]
