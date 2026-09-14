"""Stepping an env: tracking, clocks, lenses, restarts and what comes back."""

from dataclasses import replace
from typing import Any

import mujoco
import numpy as np
import pytest

from fixtures.scene import CHUNK, HOME, SMALL, SPAN, Bare, command, hands
from specialist_sim import gripper, rotation
from specialist_sim.arrays import Float64Array, UInt8Array
from specialist_sim.camera.lens import Lens
from specialist_sim.embodiment import Embodiment
from specialist_sim.env import (
  Clock,
  Env,
  Observation,
  Start,
  hold,
  interpolate,
)
from specialist_sim.gripper import HANDS, OPEN, SHUT
from specialist_sim.tasks.pick_place import PickPlace

# The box the random commands stay inside.
NEAR = np.array([0.25, -0.55, -0.40])
FAR = np.array([0.85, 0.65, 0.25])
LOW = np.array([*NEAR, *(-1.0,) * 4, SHUT])
HIGH = np.array([*FAR, *(1.0,) * 4, OPEN])


def _open(env: Env[Any]) -> Observation:
  observation, _ = env.reset(range(env.num_envs), start=Start(HOME))
  return observation


def _same(a: dict[str, Any], b: dict[str, Any]) -> None:
  assert a.keys() == b.keys()
  for key, value in a.items():
    if isinstance(value, dict):
      _same(value, b[key])
    else:
      assert np.array_equal(value, b[key]), key


def test_other_whole_clocks_step_alike(embodiment: type[Embodiment]) -> None:
  """Rows, actions and steps at other whole ratios: the hands reach where they
  are sent, and the lenses are delivered."""
  clock = Clock(per_row=6, per_action=4, per_step=8)
  env = Env(Bare(), replace(SMALL, embodiment=embodiment, clock=clock))
  observation = _open(env)
  assert observation.row.shape == (1, 8) and observation.image.shape[:2] == (1, 2)
  target = HOME.copy()
  target[1, 2] += 0.05
  for _ in range(10):
    env.step(hold(target, CHUNK)[None])
  assert np.linalg.norm(hands(env)[0, :, :3] - target[:, :3], axis=-1).max() < 0.005


def test_a_reset_fills_the_window_with_the_start(
  embodiment: type[Embodiment],
) -> None:
  env = Env(Bare(), replace(SMALL, embodiment=embodiment))
  observation, info = env.reset([0], start=Start(HOME))
  rows = env.cfg.clock.per_step
  assert observation.row.tolist() == [list(range(1 - rows, 1))]
  assert observation.state.shape == (1, rows, 2, 8)
  assert (observation.state[0] == observation.state[0, -1]).all()
  assert observation.state[0, -1, :, :3] == pytest.approx(HOME[:, :3], abs=1e-6)
  assert not info["unstable"][0] and info["tracking_error"][0] < 1e-9


def test_a_hand_s_image_is_drawn_through_the_lens_it_is_given() -> None:
  plain = _open(Env(Bare(), SMALL))
  lens = replace(Lens(), pos=(0.0, 0.0, 0.25))
  moved = _open(Env(Bare(), replace(SMALL, lenses={HANDS[0]: lens})))
  assert not np.array_equal(moved.image[0, 0], plain.image[0, 0])
  assert np.array_equal(moved.image[0, 1], plain.image[0, 1])
  alone = _open(
    Env(Bare(), replace(SMALL, driven=(HANDS[1],), image_hands=(HANDS[0],)))
  )
  assert alone.state.shape[2] == 1 and alone.image.shape[1] == 1
  assert np.array_equal(alone.image[0, 0], plain.image[0, 0]), "the left lens"


def test_the_hands_reach_a_commanded_pose_and_turn(
  embodiment: type[Embodiment],
) -> None:
  env = Env(Bare(), replace(SMALL, embodiment=embodiment))
  observation = _open(env)
  state = observation.state[0, -1].copy()
  state[:, 0] += 0.05
  state[:, 2] -= 0.05
  state[0, 3:7] = rotation.exp(np.array([0.3, -0.2, 0.9]))
  observation, info = command(env, state, 1.5)
  assert observation.state[0, -1, :, :3] == pytest.approx(state[:, :3], abs=1e-3)
  apart = rotation.multiply(
    state[0, 3:7], rotation.conjugate(observation.state[0, -1, 0, 3:7])
  )
  assert np.linalg.norm(rotation.log(apart)) < 1e-2
  assert info["tracking_error"][0] < 1e-3


def test_random_commands_do_not_diverge(embodiment: type[Embodiment]) -> None:
  """Every pose drawn independently, which slams the setpoint at the pose rate.
  The setpoint cannot leave the box, so neither can the hand chasing it."""
  env = Env(Bare(), replace(SMALL, embodiment=embodiment))
  _open(env)
  rng = np.random.default_rng(1)
  for _ in range(60):
    observation, _ = env.step(rng.uniform(LOW, HIGH, size=(1, CHUNK, 2, 8)))
    assert np.isfinite(env.sim.qpos[0]).all() and np.isfinite(env.sim.qvel[0]).all()
    reached = observation.state[0, -1, :, :3]
    assert (reached <= FAR + 0.05).all() and (reached >= NEAR - 0.05).all()


def test_an_env_mujoco_cannot_carry_is_unstable_and_left_finite(
  embodiment: type[Embodiment],
) -> None:
  """Every time, not only the first since the env was built."""
  env = Env(Bare(), replace(SMALL, embodiment=embodiment))
  _open(env)
  for _ in range(2):
    env.sim.qvel[0] = 1e30
    _, info = env.step()
    assert np.isfinite(env.sim.qpos[0]).all() and np.isfinite(env.sim.qvel[0]).all()
    assert info["unstable"][0] and info["done"][0]
    _, info = env.reset([0], start=Start(HOME))
    assert not info["unstable"][0]


def test_the_free_rig_caps_its_pull_so_a_wild_command_stays_stable() -> None:
  env = Env(Bare(), SMALL)
  _open(env)
  _, info = env.step(np.full((1, CHUNK, 2, 8), 1e6))
  assert not info["unstable"][0]
  assert np.isfinite(env.sim.qpos[0]).all() and np.isfinite(env.sim.qvel[0]).all()


def _delivered(latency: float) -> tuple[dict[int, UInt8Array], Float64Array, list[int]]:
  """Every frame a run hands over, by the row it was taken at, beside the
  plant's state at each step and the row each step was handed."""
  env = Env(Bare(), replace(SMALL, camera_latency=latency))
  observation, _ = env.reset([0], start=Start(HOME, phases=np.zeros(2)))
  action = hold(observation.state[0, -1], CHUNK)[None]
  action[..., 0, 2] += 0.06
  action[..., 1, 1] -= 0.05
  seen: dict[int, UInt8Array] = {}
  qpos = []
  taken = [int(observation.capture_row[0, 0])]
  seen[taken[0]] = observation.image[0, 0].copy()
  for _ in range(12):
    observation, _ = env.step(action)
    qpos.append(env.sim.qpos[0].copy())
    taken.append(int(observation.capture_row[0, 0]))
    seen[taken[-1]] = observation.image[0, 0].copy()
  return seen, np.asarray(qpos), taken


def _ages(taken: list[int], latency: float) -> list[float]:
  """How far past the latency each delivered frame was, in rows."""
  clock = Clock()
  behind = latency / clock.row_dt
  return [step * clock.per_step - row - behind for step, row in enumerate(taken)]


def test_a_delayed_camera_delivers_what_an_undelayed_one_saw() -> None:
  """A frame is delivered once at least the latency old, and is the frame an
  undelayed camera took at that row; the first frame is as old as the latency;
  the plant is left alone by any latency."""
  live, plain, _ = _delivered(0.0)
  for latency in (0.0, 0.03, 0.10, 0.17, 0.20):
    late, delayed, taken = _delivered(latency)
    assert np.array_equal(plain, delayed), f"a latency of {latency}s moved the plant"
    assert all(-0.5 <= age <= SPAN + 0.5 for age in _ages(taken, latency)), (
      f"a latency of {latency}s delivered {taken}"
    )
    shared = sorted(set(late) & set(live))
    assert len(shared) >= 4
    for row in shared:
      assert np.array_equal(late[row], live[row]), f"a different frame for row {row}"


def test_a_lens_fires_on_its_own_clock(monkeypatch: pytest.MonkeyPatch) -> None:
  """A lens exposes at its own period on the physics lattice and is filed
  under the row nearest each shutter."""
  clock = Clock()
  env = Env(Bare(), SMALL)
  env.reset([0], start=Start(HOME, phases=np.zeros(2)))
  drawn = 0
  assert env._renderer is not None
  render = env._renderer.render

  def counted(*poses: Any) -> Any:
    nonlocal drawn
    drawn += len(poses[0]) * len(poses[-1])
    return render(*poses)

  monkeypatch.setattr(env._renderer, "render", counted)
  steps = 12
  rows = np.array([env.step()[0].capture_row[0, 0] for _ in range(steps)])
  shots = np.rint(rows / SPAN)
  assert np.abs(rows - shots * SPAN).max() <= 0.5 + 0.5 / clock.per_row, (
    "a frame is filed under the row nearest its shutter"
  )
  gaps = np.diff(rows)
  assert (gaps > 0).all() and (gaps <= clock.per_step + SPAN).all()
  assert drawn == steps * len(env.cfg.image_hands), (
    "a lens is traced once a step: the one exposure that reaches the policy"
  )


def test_nothing_synchronizes_one_lens_to_another_or_to_a_step() -> None:
  """Each lens stands where its phase puts it, so the rows two lenses of one
  env hand over differ, the same lens of two envs differs, and few of them
  land where a step ends."""
  env = Env(Bare(), SMALL, num_envs=2)
  phases = np.array([[0.0, 0.4], [0.7, 0.1]])
  observation, _ = env.reset([0, 1], start=Start(HOME, phases=phases))
  handed = [observation.capture_row.copy()]
  for _ in range(6):
    observation, _ = env.step()
    handed.append(observation.capture_row.copy())
  rows = np.stack(handed)
  assert (rows[:, :, 0] != rows[:, :, 1]).any(axis=0).all(), "a pair sits apart"
  assert (rows[:, 0] != rows[:, 1]).any(), "two envs draw their own phase"
  ends = np.arange(len(handed)) * env.cfg.clock.per_step
  on_the_step = (rows == ends[:, None, None]).sum()
  assert on_the_step < rows.size // 2, "a shutter rarely falls on a step's end"


def test_one_env_restarts_while_the_others_carry_on(
  embodiment: type[Embodiment],
) -> None:
  """Many episodes run at once and they do not end together."""
  env = Env(Bare(), replace(SMALL, embodiment=embodiment), num_envs=3)
  nudged = HOME + 0.01 * np.arange(3)[:, None, None] * np.eye(8)[0]
  observation, _ = env.reset([0, 1, 2], start=Start(nudged, phases=np.zeros(2)))
  rows = env.cfg.clock.per_step
  for _ in range(3):
    observation, _ = env.step()
  assert observation.row[:, -1].tolist() == [3 * rows] * 3
  carried = observation.state[[0, 2], -1].copy()

  observation, _ = env.reset([1], env_ids=[1], start=Start(HOME, phases=np.zeros(2)))
  assert observation.row[1, -1] == 0, "the restarted env keeps counting"
  assert observation.row[0, -1] == 3 * rows, "the others were restarted with it"
  assert observation.capture_row[:, 0].tolist() == [3 * rows, 0, 3 * rows]
  assert np.array_equal(observation.state[[0, 2], -1], carried)

  observation, _ = env.step()
  assert observation.row[:, -1].tolist() == [4 * rows, rows, 4 * rows]


def test_an_env_in_a_batch_walks_the_path_it_walks_alone(
  embodiment: type[Embodiment],
) -> None:
  """Envs share nothing but the model, and a pause for another env's lens
  leaves an env's physics alone, so company cannot change one."""
  alone = Env(Bare(), replace(SMALL, embodiment=embodiment))
  crowd = Env(Bare(), replace(SMALL, embodiment=embodiment), num_envs=3)
  one, _ = alone.reset([7], start=Start(HOME))
  starts = np.stack([HOME, HOME, HOME])
  starts[[0, 2], :, :3] += np.array([0.02, -0.02])[:, None, None]
  many, _ = crowd.reset([5, 7, 9], start=Start(starts))
  assert np.array_equal(one.image[0], many.image[1])
  rng = np.random.default_rng(0)
  for _ in range(4):
    act_a = hold(one.state[:, -1], CHUNK)
    act_a[..., :3] += rng.normal(scale=0.01, size=(1, CHUNK, 2, 3))
    act_c = hold(many.state[:, -1], CHUNK)
    act_c[1] = act_a[0]
    one, _ = alone.step(act_a)
    many, _ = crowd.step(act_c)
    assert np.array_equal(alone.sim.qpos[0], crowd.sim.qpos[1])
    assert np.array_equal(one.image[0], many.image[1])
    assert np.array_equal(one.state[0], many.state[1])


def test_a_reset_after_stepping_is_a_fresh_env_bit_for_bit() -> None:
  """Nothing of an episode outlives its env's reset: not the solver's warm
  start, not a contact, not a frame."""
  used = Env(PickPlace(), SMALL)
  fresh = Env(PickPlace(), SMALL)
  used.reset([4])
  rng = np.random.default_rng(2)
  for _ in range(3):
    used.step(rng.uniform(LOW, HIGH, size=(1, CHUNK, 2, 8)))
  again, told = used.reset([9])
  first, made = fresh.reset([9])
  plan = hold(first.state[:, -1], CHUNK)
  plan[..., 2] -= 0.05
  for step in range(4):
    assert np.array_equal(used.sim.qpos, fresh.sim.qpos)
    assert np.array_equal(used.sim.qvel, fresh.sim.qvel)
    assert np.array_equal(again.image, first.image)
    assert np.array_equal(again.capture_row, first.capture_row)
    _same(told, made)
    if step < 3:
      again, told = used.step(plan)
      first, made = fresh.step(plan)


def test_the_rows_of_a_step_are_kept_for_a_reader() -> None:
  """Each row's state is its own ``qpos``'s hands, and its setpoint is the
  step's plan laid onto the rows; the actions past the step are never
  commanded."""
  env = Env(Bare(), SMALL, num_envs=2)
  clock = SMALL.clock
  last = _open(env).state[:, -1]
  plan = hold(last + 0.01 * np.eye(8)[0], CHUNK)
  plan[:, clock.n_action_steps :, :, :3] += 0.3
  env.step(plan)
  setpoint, _ = interpolate(last, plan[:, : clock.n_action_steps], clock)
  data = mujoco.MjData(env.model)
  for env_idx in range(2):
    rows = env.rows(env_idx)
    assert np.array_equal(rows.qpos[-1], env.sim.qpos[env_idx])
    assert np.array_equal(rows.setpoint, setpoint[env_idx])
    for qpos, state in zip(rows.qpos, rows.state, strict=True):
      data.qpos[:] = qpos
      mujoco.mj_kinematics(env.model, data)
      stood = gripper.states(
        env.hands, data.qpos[None], data.xpos[None], data.xquat[None]
      )[0]
      assert np.array_equal(stood, state)
  assert env.rows(0).timestamp == pytest.approx(
    np.arange(1, env.cfg.clock.per_step + 1) * env.cfg.clock.row_dt
  )


def test_tracing_a_recorded_row_is_what_the_lenses_see() -> None:
  env = Env(Bare(), SMALL)
  observation = _open(env)
  image, body = env.trace(env.rows(0).qpos[-1:])
  assert np.array_equal(image[0], observation.image[0])
  assert np.array_equal(body[0], observation.body[0])
  env.step(hold(HOME + 0.01 * np.eye(8)[2], CHUNK)[None])
  qpos = env.rows(0).qpos[[0, -1]]
  image, _ = env.trace(qpos)
  again, _ = env.trace(qpos[1:])
  assert np.array_equal(again[0], image[1]), "a row traces alike alone or among others"
  assert not np.array_equal(image[0], image[1])


def test_a_frame_is_the_whole_square_traced_finer_and_reduced_as_the_rig_does() -> None:
  camera = pytest.importorskip("specialist_camera")
  s = SMALL.supersample
  whole = {hand: replace(Lens(), mask_radius=None) for hand in HANDS}
  fine = replace(SMALL, image_size=SMALL.image_size * s, supersample=1, lenses=whole)
  env = Env(Bare(), SMALL)
  _open(env)
  qpos = env.rows(0).qpos[-1:]
  image = env.trace(qpos)[0][0]
  traced = Env(Bare(), fine).trace(qpos)[0][0]
  for lens in range(len(HANDS)):
    assert np.array_equal(image[lens], camera.to_image(traced[lens], s))


def test_a_one_hand_task_reproduces_the_two_hand_run_that_held_the_other(
  embodiment: type[Embodiment],
) -> None:
  """Driving the right hand alone, with the env holding the left where it was
  sent, is bitwise the run that drove both and asked the left to stay."""
  both = Env(PickPlace(), replace(SMALL, embodiment=embodiment))
  one_hand = (HANDS[1],)
  right = Env(
    PickPlace(),
    replace(SMALL, embodiment=embodiment, driven=one_hand, image_hands=one_hand),
  )
  two, _ = both.reset([11])
  one, _ = right.reset([11])
  assert np.array_equal(two.state[:, :, 1], one.state[:, :, 0])
  assert np.array_equal(two.image[:, 1], one.image[:, 0])
  sent = two.state[0, -1, 0]
  for step in range(12):
    plan = hold(one.state[0, -1], CHUNK)[None]
    plan[..., 0, 0] += 0.002 * (step + 1)
    plan[..., 0, 7] -= 0.01 * (step + 1)
    whole = np.concatenate([hold(sent[None], CHUNK)[None], plan], axis=2)
    two, _ = both.step(whole)
    one, _ = right.step(plan)
    assert np.array_equal(two.state[:, :, 1], one.state[:, :, 0])
    assert np.array_equal(two.image[:, 1], one.image[:, 0])
  assert np.array_equal(both.sim.qpos[0], right.sim.qpos[0])


def test_the_idle_hand_holds_where_it_was_sent_through_a_bump(
  embodiment: type[Embodiment],
) -> None:
  """While the other hand reaches out, the idle hand stays where it was sent,
  in the other lens's picture, and comes back there after a bump."""
  one_hand = (HANDS[1],)
  env = Env(
    PickPlace(),
    replace(SMALL, embodiment=embodiment, driven=one_hand, image_hands=one_hand),
  )
  observation, _ = env.reset([3])
  sent = hands(env)[0, 0]

  def reach_out(observation: Observation) -> Observation:
    for step in range(20):
      plan = hold(observation.state[0, -1], CHUNK)[None]
      plan[..., 0, 0] = np.minimum(plan[..., 0, 0] + 0.004 * (step + 1), 0.95)
      observation, _ = env.step(plan)
    return observation

  observation = reach_out(observation)
  moved = np.linalg.norm(hands(env)[0, 0, :3] - sent[:3])
  assert moved < 0.001, f"the idle hand drifted {moved * 1e3:.2f} mm"
  seen = {
    env.model.body(int(body)).name
    for body in np.unique(observation.body[0, 0])
    if body >= 0
  }
  assert any(name.startswith(f"{HANDS[0]}_gripper_") for name in seen), (
    "the right lens sees the idle left hand"
  )
  bumped = sent.copy()
  bumped[0] += 0.01
  env.embodiment.place(env.sim.qpos[0], 0, bumped[:7])
  reach_out(observation)
  back = np.linalg.norm(hands(env)[0, 0, :3] - sent[:3])
  assert back < 0.002, f"the bumped hand stayed {back * 1e3:.1f} mm away"
