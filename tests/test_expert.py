"""The demonstration machinery: moves without a jolt, fingers on a window,
and a script's stages carried out and paced."""

from typing import Any

import numpy as np
import pytest

from specialist_sim import rotation
from specialist_sim.expert import PEAK, Aim, Expert, ExpertCfg, Grip, Leg, Move, Stage
from specialist_sim.trajectory import HORIZON


def test_a_move_starts_and_stops_without_a_jolt() -> None:
  """A stage's curve ends at rest with no acceleration left over, which is what
  lets the next stage begin without a corner."""
  start = np.array([0.0, 0.0, 0.0])
  goal = np.array([0.2, -0.1, 0.05])
  span = 0.8
  move = Move.plan(start, np.zeros(3), goal, span)
  assert np.allclose(move.at(0.0), start) and np.allclose(move.at(span), goal)
  assert np.allclose(move.rate_at(0.0), 0.0) and np.allclose(move.rate_at(span), 0.0)
  nudge = 1e-4
  for edge in (nudge, span - nudge):
    rise = move.rate_at(edge + nudge) - move.rate_at(edge - nudge)
    assert np.abs(rise / (2.0 * nudge)).max() < 1e-2, "the curve jolts at an end"
  fastest = max(
    float(np.linalg.norm(move.rate_at(span * part / 100.0))) for part in range(101)
  )
  assert fastest <= PEAK * np.linalg.norm(goal - start) / span + 1e-9
  arched = Move.plan(start, np.zeros(3), goal, span, bulge=0.05)
  assert arched.at(span / 2.0)[2] > move.at(span / 2.0)[2] + 0.04
  assert np.allclose(arched.at(span), goal)


def test_the_fingers_work_over_their_window() -> None:
  grip = Grip(0.4, 0.0, 0.5, 1.0)
  assert grip.at(0.0) == 0.4 and grip.at(0.5) == 0.4
  assert 0.0 < grip.at(0.75) < 0.4 and grip.at(1.0) == 0.0


def test_a_leg_lays_unit_quaternions_along_its_turn() -> None:
  start = rotation.exp(np.array([0.0, 0.0, 0.2]))
  end = rotation.exp(np.array([0.0, 0.0, 1.0]))
  leg = Leg(
    Move.plan(np.zeros(3), np.zeros(3), np.array([0.1, 0.0, 0.0]), 0.3),
    Grip(0.1, 0.1, 0.0, 1.0),
    (start, end),
  )
  out = np.zeros((HORIZON, 2, 8))
  leg.lay(out, 1, 0.02)
  assert np.allclose(np.linalg.norm(out[:, 1, 3:7], axis=1), 1.0)
  yaw = rotation.log(out[:, 1, 3:7])[:, 2]
  assert (np.diff(yaw) >= 0).all() and yaw[-1] == pytest.approx(1.0, abs=1e-6)
  assert out[-1, 1, 0] == pytest.approx(0.1)
  assert (out[:, 0] == 0).all(), "the other hand is left alone"


class _Shuttle:
  """A script with one stage: carry hand 1 ten centimetres out, then back."""

  stages = (Stage(1, "out"), Stage(1, "back", turns=False))
  reach = (np.full(3, -1.0), np.full(3, 1.0))

  def __init__(self) -> None:
    self.goals: list[np.ndarray] = []

  def opens(self, hands: np.ndarray, world: Any) -> int:
    return 0

  def aim(self, stage: Stage, hands: np.ndarray, pose: np.ndarray, world: Any) -> Aim:
    goal = pose[1, :3] + (
      np.array([0.1, 0.0, 0.0])
      if stage.name == "out"
      else -0.1 * np.array([1.0, 0.0, 0.0])
    )
    self.goals.append(goal)
    return Aim(goal=goal, opening=0.1, facing=np.eye(3), bulge=0.02)

  def arrived(
    self, stage: Stage, elapsed: float, hands: np.ndarray, world: Any
  ) -> bool | None:
    return None

  def finished(self, world: Any) -> bool:
    return True


def test_an_expert_carries_a_script_out_and_answers_to_the_plant() -> None:
  """The plant answers the command exactly, so each stage ends when its move
  does, and the trajectory never jumps between calls."""
  rng = np.random.default_rng(0)
  script = _Shuttle()
  expert = Expert(script, rng, period=0.02, tracked=5, cfg=ExpertCfg(length=(0.0, 0.0)))
  state = np.zeros((2, 8))
  state[:, 3] = 1.0
  last = None
  for _ in range(200):
    action = expert.act(state, None)
    assert action.shape == (HORIZON, 2, 8)
    assert np.allclose(np.linalg.norm(action[:, :, 3:7], axis=-1), 1.0)
    stride = np.linalg.norm(np.diff(action[:, 1, :3], axis=0), axis=-1)
    assert (stride <= expert.cfg.speed * expert.period + 1e-9).all()
    if last is not None:
      assert np.linalg.norm(action[0, 1, :3] - last[expert.tracked - 1, 1, :3]) < 0.01
    last = action
    state = action[expert.tracked - 1]
    if expert.done:
      break
  assert expert.done and expert.cycles == 1
  assert len(script.goals) == 2 and script.goals[0][0] == pytest.approx(0.1)
  assert np.allclose(state[1, :3], 0.0, atol=1e-6), "the shuttle came back"
  assert np.allclose(action, action[0]), "a finished demonstration holds where it ended"
