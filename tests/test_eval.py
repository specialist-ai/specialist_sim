"""Evaluation drives every task the same way: a time limit, restarts one env at a
time, and an outcome per seed."""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from fixtures.scene import HOME, Idle
from specialist_sim import eval as evaluation
from specialist_sim.arrays import Float64Array
from specialist_sim.env import Cfg, Env, Observation, Task
from specialist_sim.eval import eval_policy, summary


class _Count:
  """Two envs where nothing moves: an env is won once it has taken as many
  steps as its seed, and a seed of zero never is."""

  num_envs = 2
  cfg = Cfg()

  def __init__(self) -> None:
    self.seeds = np.zeros(self.num_envs, np.int64)
    self.steps = np.zeros(self.num_envs, np.int64)

  def reset(
    self, seed: Sequence[int], env_ids: Sequence[int] | None = None
  ) -> tuple[Observation, dict[str, Any]]:
    chosen = list(range(self.num_envs)) if env_ids is None else list(env_ids)
    self.seeds[chosen] = seed
    self.steps[chosen] = 0
    return self._observe(), self._info()

  def step(self, action: Float64Array) -> tuple[Observation, dict[str, Any]]:
    self.steps += 1
    return self._observe(), self._info()

  def describe(self) -> dict[str, Any]:
    return {}

  def _observe(self) -> Observation:
    return Observation(
      row=self.steps[:, None],
      state=np.broadcast_to(HOME, (self.num_envs, 1, 2, 8)),
      image=np.zeros((self.num_envs, 0, 1, 1, 3), np.uint8),
      body=np.zeros((self.num_envs, 0, 1, 1), np.int32),
      capture_row=np.zeros((self.num_envs, 0), np.int64),
    )

  def _info(self) -> dict[str, Any]:
    won = (self.steps >= self.seeds) & (self.seeds > 0)
    return {
      "done": won,
      "success": won,
      "progress": np.where(won, 1.0, 0.5),
      "unstable": np.zeros(self.num_envs, np.bool_),
      "tracking_error": np.zeros(self.num_envs),
      "labels": {"carried": self.steps > 1},
    }


def _env() -> Env[Task]:
  return cast(Env[Task], _Count())


def test_eval_policy_scores_every_seed_and_restarts_envs_alone(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Every seed is an outcome, and the summary rates whatever a task labels."""
  env = _env()
  policy = Idle()
  step_dt = env.cfg.clock.step_dt
  outcomes = eval_policy(env, policy, seeds=[3, 1, 0], time_limit=4 * step_dt)
  assert [(o.seed, o.success, dict(o.labels)) for o in outcomes] == [
    (0, False, {"carried": True}),
    (1, True, {"carried": False}),
    (3, True, {"carried": True}),
  ]
  assert [o.time for o in outcomes] == pytest.approx(
    [4 * step_dt, step_dt, 3 * step_dt]
  )
  assert policy.resets == [[(0, 3), (1, 1)], [(1, 0)]]
  assert [outcome.progress for outcome in outcomes] == [0.5, 1.0, 1.0]
  monkeypatch.setattr(evaluation, "revision", lambda env: "")
  report = summary(outcomes, env, 4 * step_dt)
  assert report["success_rate"] == pytest.approx(2 / 3)
  assert report["carried_rate"] == pytest.approx(2 / 3)
  assert report["progress"] == pytest.approx(2.5 / 3)
  assert report["time_to_success"] == pytest.approx(2 * step_dt)
  assert report["unstable_rate"] == 0.0
