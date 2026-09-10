"""The plant: does it track, stay finite, and hold a cube."""

from typing import Any

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim import pose as se3
from specialist_sim.action import hold
from specialist_sim.embodiment import EmbodimentCfg
from specialist_sim.observation import flat_row
from specialist_sim.randomize import HOME_ROTATION, place_cube
from specialist_sim.spaces import (
  FINGER_HIGH,
  FINGER_LOW,
  FINGER_SHUT,
  GRIPPERS,
  HORIZON,
  PER_HAND,
  gripper_from_opening,
)
from specialist_sim.task import TaskCfg

CLEARING = gripper_from_opening(0.4)
"""A finger angle that clears the cube."""

NEAR = np.array([0.25, -0.55, -0.260])
FAR = np.array([0.85, 0.65, 0.25])
"""The corner to corner box these tests command inside, which is the table a
hand reaches across and no lower than its top."""
LOW = np.tile([*NEAR, *(-1.0,) * 4, FINGER_LOW], (HORIZON, len(GRIPPERS)))
HIGH = np.tile([*FAR, *(1.0,) * 4, FINGER_HIGH], (HORIZON, len(GRIPPERS)))


def _command(env: PickPlaceEnv, pose: np.ndarray, seconds: float) -> dict[str, Any]:
  """Ask for one pose across the whole trajectory, for as long as named."""
  commanded = hold(pose)
  for _ in range(round(seconds * env.cfg.control_hz)):
    obs, _, _, _, _ = env.step(commanded)
  return obs


def test_the_hands_reach_a_commanded_pose(embodiment: EmbodimentCfg) -> None:
  env = PickPlaceEnv(PickPlaceCfg(embodiment=embodiment))
  obs, _ = env.reset(seed=0)
  action = flat_row(obs).astype(np.float64)
  for at in (0, PER_HAND):
    action[at] += 0.05
    action[at + 2] -= 0.05
  obs = _command(env, action, 1.0)
  for at in (0, PER_HAND):
    assert flat_row(obs)[at : at + 3] == pytest.approx(action[at : at + 3], abs=1e-3)


def test_random_commands_do_not_diverge(embodiment: EmbodimentCfg) -> None:
  """Every pose drawn independently, which slams the setpoint at the trajectory rate.

  The setpoint cannot leave the box, so neither can the hand chasing it.
  """
  env = PickPlaceEnv(PickPlaceCfg(embodiment=embodiment))
  env.reset(seed=1)
  rng = np.random.default_rng(1)
  for _ in range(100):
    obs, *_ = env.step(rng.uniform(LOW, HIGH).astype(np.float64))
    assert np.isfinite(env.scene.data.qpos).all()
    assert np.isfinite(env.scene.data.qvel).all()
    for hand in range(len(GRIPPERS)):
      at = hand * PER_HAND
      reached = flat_row(obs)[at : at + 3]
      assert (reached <= FAR + 0.05).all()
      assert (reached >= NEAR - 0.05).all()


@pytest.mark.parametrize("seed", [3, 1])
def test_a_closed_hand_lifts_the_cube(seed: int, embodiment: EmbodimentCfg) -> None:
  """Whether the environment is solvable at all.

  The command names the gripper's base, so reaching the cube means offsetting
  by where the pads meet, turned into the room by the commanded rotation. The
  nearer hand does the work; the far one cannot reach across the other.
  """
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400, embodiment=embodiment))
  obs, info = env.reset(seed=seed)
  cube = np.asarray(info["cube_pos"], np.float64)
  hand = _nearest_hand(env, cube)
  action = flat_row(obs).astype(np.float64).copy()
  at = hand * PER_HAND
  held = HOME_ROTATION[hand]
  action[at + 3 : at + 7] = se3.wxyz(held)
  over = cube - held @ env.scene.grasp
  for position, grip, seconds in (
    (over + (0.0, 0.0, 0.10), CLEARING, 0.9),
    (over, CLEARING, 1.2),
    (over, FINGER_SHUT, 1.0),
    (over + (0.0, 0.0, 0.20), FINGER_SHUT, 1.5),
  ):
    action[at : at + 3], action[at + 7] = position, grip
    _command(env, action, seconds)
  lifted = env.scene.data.xpos[env.scene.cube.body][2] - env.scene.cube.rest
  assert lifted > 0.05, f"the cube only rose {lifted * 1000:.1f} mm"


def _nearest_hand(env: PickPlaceEnv, cube: np.ndarray) -> int:
  """Which hand the cube is closer to."""
  reach = [
    np.linalg.norm(env.scene.data.xpos[hand.base] - cube) for hand in env.scene.hands
  ]
  return int(np.argmin(reach))


def test_a_settled_cube_on_the_tape_pays_once() -> None:
  """The reward marks the step the episode latches, and nothing after it."""
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400))
  obs, _ = env.reset(seed=0, options={"cube": (0.60, -0.10)})
  action = flat_row(obs).astype(np.float64)
  # The cube counts as placed only once it has been off the tape.
  env.step(hold(action))
  place_cube(env.scene, env.scene.data, env.scene.tape)
  rewards: list[float] = []
  terminations: list[bool] = []
  for _ in range(TaskCfg().settle_steps + 3):
    _, reward, terminated, _, info = env.step(hold(action))
    rewards.append(float(reward[0]))
    terminations.append(bool(terminated[0]))
  assert sum(rewards) == 1.0, f"the episode paid {sum(rewards)}"
  assert rewards.index(1.0) == TaskCfg().settle_steps - 1, "the dwell is not the latch"
  assert terminations[rewards.index(1.0) :] == [True] * (
    len(rewards) - rewards.index(1.0)
  )
  assert info["is_success"]


def test_a_cube_off_the_table_ends_the_episode() -> None:
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400))
  obs, _ = env.reset(seed=0)
  address = env.scene.cube.qpos
  env.scene.data.qpos[address + 2] = env.scene.table_top - 0.5
  _, reward, terminated, _, info = env.step(hold(flat_row(obs)))
  assert info["dropped"] and terminated and reward == 0.0


def test_the_hand_tracks_the_orientation_it_is_commanded(
  embodiment: EmbodimentCfg,
) -> None:
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400, embodiment=embodiment))
  obs, _ = env.reset(seed=0)
  action = flat_row(obs).astype(np.float64)
  turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  wanted = turn @ HOME_ROTATION[0]
  action[3:7] = se3.wxyz(wanted)
  _command(env, action, 1.5)
  reached = se3.rotation(flat_row(env.step(hold(action))[0])[3:7])
  assert np.linalg.norm(se3.log(wanted @ reached.T)) < 1e-2


def test_the_tracker_passes_through_every_pose_it_tracks() -> None:
  """The setpoint reaches each tracked pose at the substep that carries it."""
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400))
  obs, _ = env.reset(seed=0)
  commanded = hold(flat_row(obs))
  commanded[:, 2] += 0.02 * np.arange(1, HORIZON + 1)
  env.ramp.process(commanded, (0, 1))
  ramp = env.ramp.position[0, :, 0, 2]
  for leg in range(env.ramp.tracked):
    assert ramp[(leg + 1) * env.ramp.span - 1] == pytest.approx(commanded[leg, 2])
  assert (np.diff(ramp) > 0).all(), "the setpoint jumped rather than ramped"
  assert abs(ramp[0] - flat_row(obs)[2]) < abs(ramp[-1] - flat_row(obs)[2]) / 10.0


def test_a_plan_reaches_past_the_step_that_executes_it() -> None:
  """The poses beyond the control step are commanded but never tracked."""
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400))
  obs, _ = env.reset(seed=0)
  assert 0 < env.ramp.tracked < HORIZON
  kept = hold(flat_row(obs))
  discarded = kept.copy()
  discarded[env.ramp.tracked :, :3] += 0.3
  env.step(kept)
  followed = env.scene.data.qpos.copy()
  env.reset(seed=0)
  env.step(discarded)
  assert env.scene.data.qpos == pytest.approx(followed)


def test_a_plan_turns_inside_one_control_step() -> None:
  """A trajectory resolves motion a single commanded pose could not express."""
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400))
  obs, _ = env.reset(seed=0)
  commanded = hold(flat_row(obs))
  commanded[:, 2] += 0.05 * np.array([1.0, 2.0, 1.0, 0.0] * (HORIZON // 4))
  env.ramp.process(commanded, (0, 1))
  ramp = env.ramp.position[0, :, 0, 2]
  rising = np.diff(ramp) > 0
  assert rising.any() and not rising.all(), "the setpoint only went one way"


def test_a_grasp_in_one_world_is_not_credited_to_its_neighbours() -> None:
  """Contacts are per world, so only the grasping world reports a held cube."""
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400), worlds=2)
  obs, info = env.reset(seeds=[0, 0])
  cube = np.asarray(info["cube_pos"][1], np.float64)
  hand = _nearest_hand(env, cube)
  at = hand * PER_HAND
  held = HOME_ROTATION[hand]
  over = cube - held @ env.scene.grasp
  grasping = flat_row(obs, world=1).astype(np.float64).copy()
  grasping[at + 3 : at + 7] = se3.wxyz(held)
  for position, grip, seconds in (
    (over + (0.0, 0.0, 0.10), CLEARING, 0.9),
    (over, CLEARING, 1.2),
    (over, FINGER_SHUT, 1.0),
    (over + (0.0, 0.0, 0.20), FINGER_SHUT, 1.5),
  ):
    grasping[at : at + 3], grasping[at + 7] = position, grip
    for _ in range(round(seconds * env.cfg.control_hz)):
      idle = hold(flat_row(obs, world=0))
      obs, _, _, _, info = env.step(np.stack([idle, hold(grasping)]))
  lifted = env.datas[1].xpos[env.scene.cube.body][2] - env.scene.cube.rest
  assert lifted > 0.05, f"the cube only rose {lifted * 1000:.1f} mm"
  assert bool(info["held"][1]) and not bool(info["held"][0])
