"""The scripted sorting demonstration clears trays, one hand or two."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from specialist_sim import episode
from specialist_sim.action import CHUNK_SIZE
from specialist_sim.eval import eval_policy
from specialist_sim.gripper import SLOTS
from specialist_sim.rig import Observation
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


class _Replay:
  """A policy that answers with the setpoints an episode commanded next."""

  def __init__(self, commanded: np.ndarray, per_action: int):
    self.commanded, self.per_action = commanded, per_action

  def reset(self, env_ids: object, seeds: object) -> None:
    pass

  def predict_action_chunk(self, observation: Observation) -> np.ndarray:
    due = observation.row[:, -1:] + self.per_action * (np.arange(CHUNK_SIZE) + 1)
    out: np.ndarray = self.commanded[np.minimum(due, len(self.commanded) - 1)]
    return out


def test_an_episode_commanded_again_lands_where_it_did(tmp_path: Path) -> None:
  """The setpoints an episode recorded, evaluated as a policy, redo it: the
  recorder and the evaluation agree on time. A setpoint read back off the ramp
  differs in its last bit, so the hand's path agrees to a tenth of a millimeter,
  not bitwise."""
  env = SortLegoEnv(replace(BLIND, driven=(SLOTS[1],), mode="single"))
  observation, report = env.reset([7])
  policy = scripted_policy(env, 0, np.random.default_rng(7))
  recorder = episode.Recorder()
  recorder.start("scripted", env.describe(0))
  recorder.extend(*(rows[-1:] for rows in env.rig.rows(0)))
  while not (report.done[0] or policy.done):
    observation, report = env.step(
      policy.predict_action_chunk(observation.state[0, -1], env.bricks(0))[None]
    )
    recorder.extend(*env.rig.rows(0))
  original = recorder.finish()
  assert report.success[0]
  replay = _Replay(original.commanded[:, [1]], env.cfg.clock.per_action)
  recordings = episode.Recordings(env, tmp_path, "replay", "single")
  (outcome,) = eval_policy(
    env, replay, [7], max_episode_steps=100, recording=recordings
  )
  assert outcome.success and outcome.progress == 1.0
  redone = episode.load(tmp_path / "single_000007.npz")
  assert redone.meta["outcome"]["success"] and redone.meta["source"] == "replay"
  rows = min(len(redone.qpos), len(original.qpos))
  apart = np.abs(redone.state[:rows, 1, :3] - original.state[:rows, 1, :3])
  assert apart.max() < 1e-4, "the hand redoes its path to a tenth of a millimeter"
