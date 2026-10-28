"""The cube: whether a hand can take it, and how the episode is evaluated."""

import numpy as np
import pytest

from specialist_sim import rotation
from specialist_sim.action import hold
from specialist_sim.embodiment import Embodiment
from specialist_sim.gripper import OPEN, SHUT
from specialist_sim.pickplace import Cfg, PickPlaceEnv, Report
from specialist_sim.pickplace.task import HOME_ROTATION, SETTLE_STEPS, Fixed
from specialist_sim.rig import Observation

CLEARING = SHUT + 0.4 * (OPEN - SHUT)
"""A finger angle that clears the cube."""


def _command(
  env: PickPlaceEnv, state: np.ndarray, seconds: float
) -> tuple[Observation, Report]:
  for _ in range(round(seconds / env.cfg.clock.step_dt)):
    out = env.step(hold(state)[None])
  return out


def _nearest_hand(env: PickPlaceEnv, cube: np.ndarray) -> int:
  reach = [
    np.linalg.norm(env.rig.datas[0].xpos[hand.base] - cube) for hand in env.rig.hands
  ]
  return int(np.argmin(reach))


def _grasp(env: PickPlaceEnv, observation: Observation, report: Report) -> None:
  """Take the cube with the nearer hand: reach over it, descend, close, lift."""
  cube = report.cube[0]
  hand = _nearest_hand(env, cube)
  state = observation.state[0, -1].copy()
  state[hand, 3:7] = rotation.from_matrix(HOME_ROTATION[hand])
  over = cube - HOME_ROTATION[hand] @ env.room.grasp
  for position, grip, seconds in (
    (over + (0.0, 0.0, 0.10), CLEARING, 0.9),
    (over, CLEARING, 1.2),
    (over, SHUT, 1.0),
    (over + (0.0, 0.0, 0.20), SHUT, 1.5),
  ):
    state[hand, :3], state[hand, 7] = position, grip
    _command(env, state, seconds)


@pytest.mark.parametrize("seed", [3, 1])
def test_a_closed_hand_lifts_the_cube(seed: int, embodiment: Embodiment) -> None:
  """Whether the environment is solvable at all. The command names the base,
  so reaching the cube means offsetting by where the pads meet."""
  env = PickPlaceEnv(Cfg(embodiment=embodiment, image_size=32))
  observation, report = env.reset([seed])
  _grasp(env, observation, report)
  lifted = env.rig.datas[0].xpos[env.room.cube.body][2] - env.room.cube.rest
  assert lifted > 0.05, f"the cube only rose {lifted * 1000:.1f} mm"
  assert env.step(hold(observation.state[0, -1])[None])[1].picked[0]


def test_a_settled_cube_on_the_tape_pays_once() -> None:
  """The reward marks the step the episode latches, and nothing after it."""
  env = PickPlaceEnv(Cfg(image_size=32))
  observation, report = env.reset([0])
  # The cube counts as placed only once it has been off the tape.
  env.reset(
    [0], initial_condition=Fixed(observation.state[0, -1], np.array([0.60, -0.10, 0.0]))
  )
  state = observation.state[0, -1]
  _, report = env.step(hold(state)[None])
  assert report.displaced[0] and not report.on_tape[0]
  at = env.room.cube.qpos
  env.rig.datas[0].qpos[at : at + 2] = env.room.tape
  rewards, dones = [], []
  for _ in range(SETTLE_STEPS + 3):
    _, report = env.step(hold(state)[None])
    rewards.append(float(report.reward[0]))
    dones.append(bool(report.done[0]))
  assert sum(rewards) == 1.0, f"the episode paid {sum(rewards)}"
  assert rewards.index(1.0) == SETTLE_STEPS - 1, "the dwell is not the latch"
  assert all(dones[rewards.index(1.0) :]) and report.success[0]
  assert not report.picked[0], "a cube nudged onto the tape was never carried"


def test_a_cube_off_the_table_ends_the_episode() -> None:
  env = PickPlaceEnv(Cfg(image_size=32))
  observation, _ = env.reset([0])
  at = env.room.cube.qpos
  env.rig.datas[0].qpos[at + 2] = env.room.table_top - 0.5
  _, report = env.step(hold(observation.state[0, -1])[None])
  assert report.dropped[0] and report.done[0] and report.reward[0] == 0.0


def test_a_grasp_in_one_env_is_not_credited_to_its_neighbors() -> None:
  """Contacts are per env, so only the grasping env reports a held cube."""
  env = PickPlaceEnv(Cfg(image_size=32), num_envs=2)
  observation, report = env.reset([0, 0])
  cube = report.cube[1]
  hand = _nearest_hand(env, cube)
  grasping = observation.state[1, -1].copy()
  grasping[hand, 3:7] = rotation.from_matrix(HOME_ROTATION[hand])
  over = cube - HOME_ROTATION[hand] @ env.room.grasp
  for position, grip, seconds in (
    (over + (0.0, 0.0, 0.10), CLEARING, 0.9),
    (over, CLEARING, 1.2),
    (over, SHUT, 1.0),
    (over + (0.0, 0.0, 0.20), SHUT, 1.5),
  ):
    grasping[hand, :3], grasping[hand, 7] = position, grip
    for _ in range(round(seconds / env.cfg.clock.step_dt)):
      idle = observation.state[0, -1]
      observation, report = env.step(np.stack([hold(idle), hold(grasping)]))
  lifted = env.rig.datas[1].xpos[env.room.cube.body][2] - env.room.cube.rest
  assert lifted > 0.05, f"the cube only rose {lifted * 1000:.1f} mm"
  assert report.held[1] and not report.held[0]
  assert report.picked[1] and not report.picked[0]
