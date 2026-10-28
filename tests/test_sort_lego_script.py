"""The scripted sorting demonstration clears trays, one hand or two."""

from dataclasses import replace

import numpy as np

from specialist_sim.gripper import SLOTS
from specialist_sim.sort_lego import Cfg, SortLegoEnv
from specialist_sim.sort_lego.script import scripted_policy

BLIND = Cfg(render=False)


def _run(env: SortLegoEnv, seed: int, seconds: float) -> tuple[bool, float]:
  observation, report = env.reset([seed])
  policy = scripted_policy(env, 0, np.random.default_rng(seed))
  for _ in range(round(seconds / env.cfg.clock.step_dt)):
    action = policy.predict_action_chunk(observation.state[0, -1], env.bricks(0))
    observation, report = env.step(action[None])
    if report.done[0] or policy.done:
      break
  return bool(report.success[0]), float(report.progress[0])


def test_the_right_hand_sorts_single_bricks() -> None:
  env = SortLegoEnv(replace(BLIND, driven=(SLOTS[1],), mode="single"))
  cleared = [_run(env, seed, 20.0)[0] for seed in range(5)]
  assert sum(cleared) >= 4, cleared


def test_two_hands_take_turns_on_a_mixed_tray() -> None:
  env = SortLegoEnv(replace(BLIND, bricks=3))
  success, progress = _run(env, 4, 45.0)
  assert progress >= 2 / 3, progress
