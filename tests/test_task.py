"""A task names the hands a policy drives; the rig stays whole underneath."""

from dataclasses import replace

import numpy as np

from specialist_sim.camera import NOMINAL
from specialist_sim.embodiment import Embodiment
from specialist_sim.gripper import SLOTS
from specialist_sim.pickplace import Cfg, PickPlace
from specialist_sim.pickplace.task import (
  BIMANUAL,
  Fixed,
  Home,
  Region,
  Task,
  left_task,
  right_task,
)
from specialist_sim.trajectory import hold

PATCH = Region(vertices=((0.50, -0.14), (0.82, -0.14), (0.82, 0.10), (0.50, 0.10)))
"""A patch of table, off the tape on the right hand's side and beyond."""
SMALL = Cfg(image_size=32)


def test_a_one_hand_task_reproduces_the_two_hand_run_that_held_the_other(
  embodiment: Embodiment,
) -> None:
  """Driving the right hand alone, with the env holding the left where it was
  sent, is bitwise the run that drove both and asked the left to stay."""
  both = PickPlace(replace(SMALL, embodiment=embodiment, task=Task(SLOTS, Home(PATCH))))
  right = PickPlace(
    replace(SMALL, embodiment=type(embodiment)(), task=right_task(PATCH))
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
  env = PickPlace(replace(SMALL, embodiment=embodiment, task=right_task(PATCH)))
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
  env = PickPlace(replace(SMALL, task=right_task(PATCH)))
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


def test_a_kick_succeeds_when_the_cube_rests_in_the_picking_region() -> None:
  env = PickPlace(replace(SMALL, task=left_task(PATCH)))
  observation, report = env.reset([0])
  assert report.on_tape[0]
  hands = env.rig.states(env.rig.datas[0].qpos[None])[0]
  spot = np.array([0.75, -0.05])
  assert PATCH.contains(spot) and not PATCH.contains(np.array([0.40, 0.0]))
  observation, report = env.reset([0], start=Fixed(hands, np.array([*spot, 0.0])))
  assert not report.on_tape[0]
  for _ in range(8):
    observation, report = env.step(hold(observation.state[0, -1])[None])
    if report.done[0]:
      break
  assert report.success[0] and report.reward[0] == 1.0
  env.reset([0])
  for _ in range(8):
    observation, report = env.step(hold(observation.state[0, -1])[None])
  assert not report.success[0], "a cube left on the tape is no kick"


def test_a_kicked_start_opens_the_episode_at_the_pick() -> None:
  env = PickPlace(replace(SMALL, task=Task(SLOTS, Home(PATCH))))
  _, report = env.reset([0])
  assert not report.on_tape[0] and report.displaced[0]
  assert PATCH.contains(report.cube[0, :2])
  _, again = env.reset([0])
  assert np.array_equal(report.cube, again.cube), "a seed names one start"
  assert BIMANUAL.driven == SLOTS


def test_a_scene_hands_over_a_frame_as_old_as_it_asks_for() -> None:
  """A policy taught a delay is scored against one: the frame an observation
  carries is at least the scene's lag behind the step it returns at."""
  span = 1.0 / (NOMINAL.hz * SMALL.clock.row_dt)
  for lag in (0.0, 0.12, 0.15, 0.18):
    env = PickPlace(replace(SMALL, image_lag=lag))
    observation, report = env.reset([0])
    for _ in range(3):
      observation, report = env.step(hold(observation.state[0, -1])[None])
    age = observation.row[0, -1] - observation.image_row[0] - lag / SMALL.clock.row_dt
    assert (age >= -0.5).all(), f"a lag of {lag}s delivered {age.tolist()}"
    assert (age <= span + 0.5).all(), f"a lag of {lag}s delivered {age.tolist()}"


def test_a_task_may_see_a_lens_it_does_not_drive() -> None:
  env = PickPlace(replace(SMALL, task=Task((SLOTS[1],), Home(PATCH), seen=(SLOTS[0],))))
  observation, _ = env.reset([0])
  assert observation.state.shape[2] == 1 and observation.image.shape[1] == 2
  assert env.rig.lenses == list(SLOTS) and env.rig.cfg.driven == (SLOTS[1],)


def test_a_region_draws_inside_its_ring_and_outside_its_hole() -> None:
  ring = ((0.30, -0.15), (0.65, -0.15), (0.65, 0.15), (0.30, 0.15))
  region = Region(vertices=ring, hole=(0.49, 0.0, 0.08))
  spots = np.stack([region.draw(np.random.default_rng(seed)) for seed in range(200)])
  assert region.contains(spots).all()
  assert (np.hypot(spots[:, 0] - 0.49, spots[:, 1]) >= 0.08).all()
  assert (spots >= [0.30, -0.15]).all() and (spots <= [0.65, 0.15]).all()
  assert not region.contains(np.array([0.49, 0.0])) and not region.contains(
    np.array([0.7, 0.0])
  )
  again = region.draw(np.random.default_rng(3))
  assert np.array_equal(spots[3], again), "a seed names one draw"
