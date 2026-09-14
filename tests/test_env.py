"""What an env does whatever task it stands: which hands a policy drives, which
lenses it sees, how old a frame is, and when a task's goal is won. The
pick-and-place task is the one it is stood here."""

from dataclasses import replace

import numpy as np

from specialist_sim.action import hold
from specialist_sim.camera import NOMINAL
from specialist_sim.embodiment import Embodiment
from specialist_sim.env import Cfg, Env
from specialist_sim.gripper import SLOTS
from specialist_sim.pickplace import PickPlace
from specialist_sim.pickplace.task import on_tape

SMALL = Cfg(image_size=32)


def test_a_one_hand_task_reproduces_the_two_hand_run_that_held_the_other(
  embodiment: Embodiment,
) -> None:
  """Driving the right hand alone, with the env holding the left where it was
  sent, is bitwise the run that drove both and asked the left to stay."""
  both = Env(PickPlace(), replace(SMALL, embodiment=embodiment))
  right = Env(
    PickPlace(), replace(SMALL, embodiment=type(embodiment)(), driven=(SLOTS[1],))
  )
  two, _ = both.reset([11])
  one, _ = right.reset([11])
  assert np.array_equal(two.state[:, :, 1], one.state[:, :, 0])
  assert np.array_equal(two.image[:, 1], one.image[:, 0])
  sent = two.state[0, -1, 0]
  for step in range(12):
    plan = hold(one.state[0, -1])[None]
    plan[..., 0, 0] += 0.002 * (step + 1)
    plan[..., 0, 7] -= 0.01 * (step + 1)
    whole = np.concatenate([hold(sent[None])[None], plan], axis=2)
    two, _ = both.step(whole)
    one, _ = right.step(plan)
    assert np.array_equal(two.state[:, :, 1], one.state[:, :, 0])
    assert np.array_equal(two.image[:, 1], one.image[:, 0])
  assert np.array_equal(both.rig.datas[0].qpos, right.rig.datas[0].qpos)


def test_the_idle_hand_stays_in_the_air_and_in_the_picture(
  embodiment: Embodiment,
) -> None:
  env = Env(PickPlace(), replace(SMALL, embodiment=embodiment, driven=(SLOTS[1],)))
  observation, _ = env.reset([3])
  assert observation.state.shape[2] == 1 and observation.image.shape[1] == 1
  left = env.rig.hands[0]
  parked = env.rig.datas[0].xpos[left.base].copy()
  for step in range(20):
    plan = hold(observation.state[0, -1])[None]
    plan[..., 0, 0] = np.minimum(plan[..., 0, 0] + 0.004 * (step + 1), 0.95)
    observation, _ = env.step(plan)
  moved = np.linalg.norm(env.rig.datas[0].xpos[left.base] - parked)
  assert moved < 0.001, f"the idle hand drifted {moved * 1e3:.2f} mm"
  model = env.rig.model
  seen = {
    model.body(int(body)).name
    for body in np.unique(observation.body[0, 0])
    if body >= 0
  }
  assert any(name.startswith(SLOTS[0]) for name in seen), (
    "the right lens sees the idle left hand"
  )


def test_the_idle_hand_returns_to_where_it_was_sent_after_a_bump() -> None:
  env = Env(PickPlace(), replace(SMALL, driven=(SLOTS[1],)))
  observation, _ = env.reset([5])
  sent = env.rig.states(env.rig.datas[0].qpos[None])[0, 0]
  bumped = sent.copy()
  bumped[0] += 0.01
  env.rig.cfg.embodiment.place(env.rig.datas[0], 0, bumped[:7])
  for _ in range(20):
    observation, _ = env.step(hold(observation.state[0, -1])[None])
  back = np.linalg.norm(
    env.rig.states(env.rig.datas[0].qpos[None])[0, 0, :3] - sent[:3]
  )
  assert back < 0.002, f"the bumped hand stayed {back * 1e3:.1f} mm away"


def test_a_cube_resting_on_the_tape_and_let_go_of_wins() -> None:
  env = Env(PickPlace(), SMALL)
  observation, report = env.reset([0])
  for _ in range(8):
    observation, report = env.step(hold(observation.state[0, -1])[None])
  assert not report.success[0], "a cube left where it started is no placement"
  hands = env.rig.states(env.rig.datas[0].qpos[None])[0]
  observation, report = env.reset(
    [0], start=(hands, np.array([*env.task.room.tape, 0.0]))
  )
  assert report.labels["on_tape"][0]
  for _ in range(8):
    observation, report = env.step(hold(observation.state[0, -1])[None])
    if report.done[0]:
      break
  assert report.success[0] and report.reward[0] == 1.0


def test_a_seed_names_one_start_in_the_region_and_off_the_tape() -> None:
  env = Env(PickPlace(), SMALL)

  def spot(seed: int) -> np.ndarray:
    env.reset([seed])
    return env.task.cube()[0, :2]

  spots = np.stack([spot(seed) for seed in range(16)])
  half = np.array(env.task.region) / 2.0
  assert (np.abs(spots - env.task.room.tape) <= half + 1e-9).all()
  assert not on_tape(env.task.room, np.c_[spots, np.zeros(len(spots))]).any()
  assert np.array_equal(spots[3], spot(3)), "a seed names one start"
  assert SMALL.driven == SLOTS


def test_an_env_hands_over_a_frame_as_old_as_it_asks_for() -> None:
  """A policy taught a delay is scored against one: the frame an observation
  carries is at least the env's camera latency behind the step it returns at."""
  span = 1.0 / (NOMINAL.hz * SMALL.clock.row_dt)
  for latency in (0.0, 0.12, 0.15, 0.18):
    env = Env(PickPlace(), replace(SMALL, camera_latency=latency))
    observation, report = env.reset([0])
    for _ in range(3):
      observation, report = env.step(hold(observation.state[0, -1])[None])
    age = (
      observation.row[0, -1]
      - observation.image_row[0, :]
      - latency / SMALL.clock.row_dt
    )
    assert (age >= -0.5).all(), f"a latency of {latency}s delivered {age.tolist()}"
    assert (age <= span + 0.5).all(), (
      f"a latency of {latency}s delivered {age.tolist()}"
    )


def test_an_env_may_see_a_lens_it_does_not_drive() -> None:
  env = Env(PickPlace(), replace(SMALL, driven=(SLOTS[1],), seen=(SLOTS[0],)))
  observation, _ = env.reset([0])
  assert observation.state.shape[2] == 1 and observation.image.shape[1] == 2
  assert env.rig.lenses == list(SLOTS) and env.rig.cfg.driven == (SLOTS[1],)
