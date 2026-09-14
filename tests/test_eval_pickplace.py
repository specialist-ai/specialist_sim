"""Evaluation on the pick-and-place task."""

from dataclasses import replace
from typing import Any

import numpy as np

from specialist_sim.action import hold
from specialist_sim.env import Cfg, Env
from specialist_sim.eval import eval_policy, summary
from specialist_sim.pickplace import PickPlace
from specialist_sim.rig import Observation


class _Idle:
  def reset(self, env_ids: Any, seeds: Any) -> None:
    pass

  def predict_action_chunk(self, observation: Observation) -> np.ndarray:
    return hold(observation.state[:, -1])


def test_eval_policy_scores_every_seed_and_names_what_it_ran_under() -> None:
  env = Env(PickPlace(), Cfg(image_size=32), num_envs=2)
  outcomes = eval_policy(env, _Idle(), seeds=[3, 4, 5], max_episode_steps=4)
  assert [outcome.seed for outcome in outcomes] == [3, 4, 5]
  assert all(
    outcome.steps == 4 and not outcome.success and not outcome.labels["picked"]
    for outcome in outcomes
  )
  report = summary(outcomes, env, 4)
  assert (
    report["n_episodes"] == 3
    and report["success_rate"] == 0.0
    and report["num_envs"] == 2
  )
  assert "image_size=32" in report["cfg"] and len(report["revision"]) == 12
  assert report["task"] == "PickPlace(region=(0.32, 0.24))"
  assert (
    report["revision"]
    != summary(outcomes, Env(PickPlace(), replace(env.cfg, image_size=48)), 4)[
      "revision"
    ]
  )
  assert report["tracking_error"] < 0.01 and report["picked_rate"] == 0.0
