"""Evaluation drives every task the same way: a time limit, restarts one env at a
time, and an outcome per seed."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import mujoco
import numpy as np
import pytest

from scene import HOME, bare
from specialist_sim.action import hold
from specialist_sim.arrays import Float64Array
from specialist_sim.env import Cfg, Env, Task, Verdict
from specialist_sim.eval import Outcome, eval_policy, mcnemar, summary
from specialist_sim.rig import Observation, Rig


@dataclass
class _Count(Task):
  """A scene with nothing in it and a goal of standing in it: an env is won
  once it has taken as many steps as its seed, and a seed of zero never is."""

  name: ClassVar[str] = "count"

  def spec(self) -> mujoco.MjSpec:
    return bare()

  @property
  def focus(self) -> Float64Array:
    return np.zeros(3)

  def bind(self, rig: Rig) -> None:
    super().bind(rig)
    self.seeds = np.zeros(rig.num_envs, np.int64)

  def open(self, env_idx: int, seed: int, start: Any) -> Float64Array:
    self.rig.place(env_idx, HOME)
    self.seeds[env_idx] = seed
    return np.zeros(len(self.rig.lenses))

  def judge(self) -> Verdict:
    steps = self.rig.row // self.rig.cfg.clock.per_step
    won = (steps >= self.seeds) & (self.seeds > 0)
    return Verdict(
      progress=np.where(won, 1.0, 0.5),
      solved=won,
      failed=np.zeros(self.rig.num_envs, np.bool_),
      labels={"carried": steps > 1},
    )


def _env() -> Env[_Count]:
  return Env(_Count(), Cfg(render=False), num_envs=2)


class _Idle:
  def __init__(self) -> None:
    self.resets: list[list[tuple[int, int]]] = []

  def reset(self, env_ids: Sequence[int], seeds: Sequence[int]) -> None:
    self.resets.append(list(zip(env_ids, seeds, strict=True)))

  def predict_action_chunk(self, observation: Observation) -> np.ndarray:
    return hold(observation.state[:, -1])


def test_eval_policy_scores_every_seed_and_restarts_envs_alone() -> None:
  env, policy = _env(), _Idle()
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
  assert report["steps_to_success"] == 2.0 and report["tracking_error"] < 0.01
  assert len(report["revision"]) == 12 and "render=False" in report["cfg"]


def test_a_task_names_what_evaluation_counts() -> None:
  """Evaluation rates whatever a task labels and knows nothing of what it
  means; only the task reads its own names back."""
  env, policy = _env(), _Idle()
  outcomes = eval_policy(env, policy, seeds=[3, 1, 0], max_episode_steps=4)
  assert all(set(outcome.labels) == {"carried"} for outcome in outcomes)
  assert bucket_of(outcomes) == ["lost", "slid", "clean"]


def bucket_of(outcomes: Any) -> list[str]:
  """What a task would make of them, from the names it counts under."""
  return [
    "unstable"
    if outcome.unstable
    else ("clean" if outcome.labels["carried"] else "slid")
    if outcome.success
    else ("lost" if outcome.labels["carried"] else "neither")
    for outcome in outcomes
  ]


def test_two_evaluations_are_compared_seed_by_seed() -> None:
  def outcomes(wins: list[bool]) -> list[Outcome]:
    return [
      Outcome(seed, won, float(won), 1, False, 0.0, {}) for seed, won in enumerate(wins)
    ]

  first = outcomes([True] * 12 + [False] * 3)
  second = outcomes([False] * 11 + [True] * 4)
  only_first, only_second, p = mcnemar(first, second)
  assert (only_first, only_second) == (11, 3) and 0.05 < p < 0.06
  assert mcnemar(first, first) == (0, 0, 1.0)
  with pytest.raises(ValueError):
    mcnemar(first, outcomes([True]))
