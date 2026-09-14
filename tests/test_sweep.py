"""The bench on the pick-and-place scene."""

from dataclasses import replace
from typing import Any

import numpy as np

from specialist_sim.bench import summary, sweep
from specialist_sim.gripper import SLOTS
from specialist_sim.pickplace import Cfg, PickPlace
from specialist_sim.pickplace.env import bucket
from specialist_sim.pickplace.task import Home, Region, Task
from specialist_sim.rig import Observation
from specialist_sim.trajectory import hold

PATCH = Region(vertices=((0.50, -0.14), (0.82, -0.14), (0.82, 0.10), (0.50, 0.10)))


class _Idle:
  def reset(self, worlds: Any, seeds: Any) -> None:
    pass

  def act(self, observation: Observation) -> np.ndarray:
    return hold(observation.state[:, -1])


def test_a_sweep_scores_every_seed_and_names_what_it_ran_under() -> None:
  env = PickPlace(Cfg(image_size=32, task=Task(SLOTS, Home(PATCH))), worlds=2)
  outcomes = sweep(env, _Idle(), seeds=[3, 4, 5], budget=4)
  assert [outcome.seed for outcome in outcomes] == [3, 4, 5]
  assert all(
    outcome.steps == 4 and bucket(outcome) == "neither" for outcome in outcomes
  )
  report = summary(outcomes, env, 4)
  assert (
    report["episodes"] == 3 and report["success_rate"] == 0.0 and report["worlds"] == 2
  )
  assert "Home" in report["cfg"] and len(report["revision"]) == 12
  assert (
    report["revision"]
    != summary(outcomes, PickPlace(replace(env.cfg, image_size=48)), 4)["revision"]
  )
  assert report["tracking"] < 0.01 and report["picked_rate"] == 0.0
