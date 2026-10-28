"""Stepping envs of a rig: tracking error, clocks, restarts, and the shape of
what comes back, in a scene holding nothing but the hands."""

from dataclasses import replace
from typing import Any

import mujoco
import numpy as np
import pytest

from scene import bare
from specialist_sim import rotation
from specialist_sim.action import CHUNK_SIZE, hold
from specialist_sim.camera import NOMINAL
from specialist_sim.embodiment import Embodiment, Floating
from specialist_sim.gripper import GRIPPING, OPEN, SHUT, SLOTS
from specialist_sim.physics import Physics
from specialist_sim.rig import Cfg, Clock, Observation, Rig


def test_other_whole_rates_step_alike_and_an_undriven_hand_holds(
  embodiment: Embodiment,
) -> None:
  """Rows, actions and steps at other whole ratios: the driven hand reaches
  where it is sent, the other holds where it was placed, and its lens is
  still delivered."""
  clock = Clock(physics_dt=1.0 / 600.0, per_row=6, per_action=4, per_step=8)
  rig = Rig(
    replace(
      SMALL, embodiment=embodiment, clock=clock, driven=(SLOTS[1],), seen=(SLOTS[0],)
    )
  )
  rig.place(0, HOME)
  observation = rig.begin([0])
  assert observation.row.shape == (1, 8) and observation.state.shape == (1, 8, 1, 8)
  assert observation.image.shape[:2] == (1, 2)
  target = HOME[1:2].copy()
  target[0, 2] += 0.05
  for _ in range(10):
    observation = rig.step(hold(target[None], CHUNK_SIZE))
  hands = rig.states(rig.datas[0].qpos[None])[0]
  assert np.linalg.norm(hands[1, :3] - target[0, :3]) < 0.005
  assert np.linalg.norm(hands[0, :3] - HOME[0, :3]) < 0.005


SMALL = Cfg(scene=bare, image_size=32)
SPAN = 1.0 / (NOMINAL.hz * Clock().row_dt)
"""Rows between one exposure of a nominal lens and its next."""
HOME = np.array(
  [
    [0.45, 0.15, -0.10, 1.0, 0.0, 0.0, 0.0, 0.2],
    [0.45, -0.15, -0.10, 0.0, 0.0, 0.0, 1.0, 0.2],
  ]
)
NEAR = np.array([0.25, -0.55, -0.40])
FAR = np.array([0.85, 0.65, 0.25])
"""The corner to corner box the random commands stay inside."""
LOW = np.array([*NEAR, *(-1.0,) * 4, SHUT])
HIGH = np.array([*FAR, *(1.0,) * 4, OPEN])


def _begin(rig: Rig, num_envs: int = 1, nudge: float = 0.0) -> Observation:
  for env_idx in range(num_envs):
    rig.place(env_idx, HOME + nudge * env_idx)
  return rig.begin(range(num_envs))


def _command(rig: Rig, state: np.ndarray, seconds: float) -> Observation:
  for _ in range(round(seconds / rig.cfg.clock.step_dt)):
    out = rig.step(hold(state)[None])
  return out


def test_a_beginning_reports_the_hands_and_every_camera(embodiment: Embodiment) -> None:
  rig = Rig(replace(SMALL, embodiment=embodiment))
  observation = _begin(rig)
  rows, size = rig.cfg.clock.per_step, rig.cfg.image_size
  assert observation.row.tolist() == [list(range(1 - rows, 1))]
  assert observation.state.shape == (1, rows, 2, 8)
  assert (observation.state[0] == observation.state[0, -1]).all()
  assert observation.state[0, -1, :, :3] == pytest.approx(HOME[:, :3], abs=1e-6)
  assert (
    observation.image.shape == (1, 2, 1, size, size, 3)
    and observation.image.dtype == np.uint8
  )
  assert observation.body.shape == (1, 2, 1, size, size)
  assert observation.image_row.tolist() == [[[0], [0]]]
  assert not rig.unstable[0] and rig.tracking_error[0] < 1e-9
  assert (observation.image[0, 0, -1].sum(axis=2) > 0).mean() > 0.2
  assert not np.array_equal(observation.image[0, 0], observation.image[0, 1])


def test_every_lens_sees_the_scene_through_the_camera_it_is_given() -> None:
  plain = _begin(Rig(SMALL))
  camera = replace(NOMINAL, pos=(0.0, 0.0, 0.25))
  moved = _begin(Rig(replace(SMALL, cameras={SLOTS[0]: camera})))
  assert not np.array_equal(moved.image[0, 0], plain.image[0, 0])
  assert np.array_equal(moved.image[0, 1], plain.image[0, 1])


def test_the_hands_reach_a_commanded_pose(embodiment: Embodiment) -> None:
  rig = Rig(replace(SMALL, embodiment=embodiment))
  observation = _begin(rig)
  state = observation.state[0, -1].copy()
  state[:, 0] += 0.05
  state[:, 2] -= 0.05
  observation = _command(rig, state, 1.0)
  assert observation.state[0, -1, :, :3] == pytest.approx(state[:, :3], abs=1e-3)
  assert rig.tracking_error[0] < 1e-3


def test_random_commands_do_not_diverge(embodiment: Embodiment) -> None:
  """Every pose drawn independently, which slams the setpoint at the pose rate.
  The setpoint cannot leave the box, so neither can the hand chasing it."""
  rig = Rig(replace(SMALL, embodiment=embodiment))
  _begin(rig)
  rng = np.random.default_rng(1)
  for _ in range(60):
    observation = rig.step(rng.uniform(LOW, HIGH, size=(1, CHUNK_SIZE, 2, 8)))
    assert np.isfinite(rig.datas[0].qpos).all() and np.isfinite(rig.datas[0].qvel).all()
    reached = observation.state[0, -1, :, :3]
    assert (reached <= FAR + 0.05).all() and (reached >= NEAR - 0.05).all()


def test_the_hand_tracks_the_orientation_it_is_commanded(
  embodiment: Embodiment,
) -> None:
  rig = Rig(replace(SMALL, embodiment=embodiment))
  observation = _begin(rig)
  state = observation.state[0, -1].copy()
  wanted = rotation.to_matrix(rotation.exp(np.array([0.3, -0.2, 0.9])))
  state[0, 3:7] = rotation.from_matrix(wanted)
  observation = _command(rig, state, 1.5)
  reached = rotation.to_matrix(observation.state[0, -1, 0, 3:7])
  assert np.linalg.norm(rotation.log(rotation.from_matrix(wanted @ reached.T))) < 1e-2


def test_a_plan_reaches_past_the_step_that_executes_it() -> None:
  """The actions beyond the control step are commanded but never executed."""
  rig = Rig(SMALL)
  observation = _begin(rig)
  assert 0 < rig.cfg.clock.n_action_steps < CHUNK_SIZE
  kept = hold(observation.state[0, -1])[None]
  discarded = kept.copy()
  discarded[:, rig.cfg.clock.n_action_steps :, :, :3] += 0.3
  rig.step(kept)
  followed = rig.datas[0].qpos.copy()
  _begin(rig)
  rig.step(discarded)
  assert rig.datas[0].qpos == pytest.approx(followed)


def test_an_env_mujoco_cannot_carry_is_unstable_and_left_finite(
  embodiment: Embodiment,
) -> None:
  rig = Rig(replace(SMALL, embodiment=embodiment))
  _begin(rig)
  rig.datas[0].qvel[:] = 1e30
  rig.step(hold(rig.state_rows[:, -1]))
  assert np.isfinite(rig.datas[0].qpos).all() and np.isfinite(rig.datas[0].qvel).all()
  assert rig.unstable[0]
  _begin(rig)
  assert not rig.unstable[0]


def test_the_free_rig_caps_its_pull_so_a_wild_command_stays_stable() -> None:
  rig = Rig(SMALL)
  _begin(rig)
  rig.step(np.full((1, CHUNK_SIZE, 2, 8), 1e6))
  assert not rig.unstable[0]
  assert np.isfinite(rig.datas[0].qpos).all() and np.isfinite(rig.datas[0].qvel).all()


def test_a_step_reports_the_rows_it_spanned() -> None:
  rig = Rig(SMALL)
  observation = _begin(rig)
  rows = rig.cfg.clock.per_step
  for step in range(1, 4):
    observation = rig.step(hold(observation.state[0, -1])[None])
    assert observation.row[0].tolist() == list(
      range((step - 1) * rows + 1, step * rows + 1)
    )
    age = step * rows - int(observation.image_row[0, 0, -1])
    assert 0 <= age <= SPAN


def _delivered(
  latency: float, steps: int = 6
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, list[int]]:
  """Every frame a run hands over and every frame its lenses expose, both by
  the row they were taken at, beside the plant's state at each step."""
  rig = Rig(replace(SMALL, camera_latency=latency))
  observation = _begin(rig)
  action = hold(observation.state[0, -1])[None]
  action[..., 0, 2] += 0.06
  action[..., 1, 1] -= 0.05
  seen: dict[int, np.ndarray] = {}
  shot: dict[int, np.ndarray] = {}
  qpos, taken = [], []
  for _ in range(steps):
    observation = rig.step(action)
    qpos.append(rig.datas[0].qpos.copy())
    taken.append(int(observation.image_row[0, 0, -1]))
    seen[int(observation.image_row[0, 0, -1])] = observation.image[0, 0, -1].copy()
    for frame in rig.exposed[0][SLOTS[0]]:
      shot[frame.row] = frame.image.copy()
  return seen, shot, np.asarray(qpos), taken


def _ages(taken: list[int], latency: float) -> list[float]:
  """How far past the latency each delivered frame was, in rows."""
  clock = Clock()
  behind = latency / clock.row_dt
  return [
    step * clock.per_step - row - behind for step, row in enumerate(taken, start=1)
  ]


def test_a_delayed_camera_delivers_what_an_undelayed_one_saw() -> None:
  """A frame is quarantined until it is at least the latency old, and is the frame
  a camera keeping up with the rest of the stack took at that row; the plant is
  left alone by any latency."""
  _, live, plain, _ = _delivered(0.0)
  for latency in (0.0, 0.03, 0.10, 0.17, 0.20):
    late, _, delayed, taken = _delivered(latency)
    assert np.array_equal(plain, delayed), f"a latency of {latency}s moved the plant"
    assert all(-0.5 <= age <= SPAN + 0.5 for age in _ages(taken, latency)), (
      f"a latency of {latency}s delivered {taken}"
    )
    shared = sorted(set(late) & set(live))
    assert len(shared) >= 4
    for row in shared:
      assert np.array_equal(late[row], live[row]), f"a different frame for row {row}"


def test_an_episode_opens_with_a_camera_that_was_already_running() -> None:
  """The first frame is as old as the latency asks, drawn from before the episode."""
  for latency in (0.0, 0.03, 0.15, 0.17):
    rig = Rig(replace(SMALL, camera_latency=latency))
    observation = _begin(rig)
    age = -int(observation.image_row[0, 0, -1]) - latency / Clock().row_dt
    assert -0.5 <= age <= SPAN + 0.5


def test_a_lens_fires_on_its_own_clock() -> None:
  """A lens runs at the rate a capture's does, not the control step's: it
  exposes at its own period on the physics lattice, is filed under the row
  nearest each shutter, and is drawn once for each."""
  clock = Clock()
  rig = Rig(SMALL)
  observation = _begin(rig)
  drawn = 0
  assert rig._renderer is not None
  rendering = rig._renderer.render_many

  def counted(datas: Any, names: Any, rendering: Any = rendering) -> Any:
    nonlocal drawn
    drawn += len(datas) * len(names)
    return rendering(datas, names)

  rig._renderer.render_many = counted
  rows: list[int] = []
  for _ in range(12):
    observation = rig.step(hold(observation.state[0, -1])[None])
    rows.extend(frame.row for frame in rig.exposed[0][SLOTS[0]])
  gaps = np.diff(rows)
  assert set(gaps.tolist()) == {3, 4}, "a shutter falls where the rows are nearest it"
  assert float(np.mean(gaps)) == pytest.approx(SPAN, abs=0.05)
  assert rows[-1] <= 12 * clock.per_step
  assert drawn == len(rows) * len(rig.lenses), "a frame is drawn once, when exposed"


def test_nothing_synchronizes_one_lens_to_another_or_to_a_step() -> None:
  """Each lens stands where its phase puts it, so two lenses of one env sit
  apart and the same lens of two envs does too."""
  rig = Rig(SMALL, num_envs=2)
  for env_idx in range(2):
    rig.place(env_idx, HOME)
  phases = np.array([[0.0, 0.4], [0.7, 0.1]])
  observation = rig.begin([0, 1], phases)
  taken: list[list[list[float]]] = [[[], []], [[], []]]
  for _ in range(6):
    observation = rig.step(hold(observation.state[:, -1]))
    for env_idx in range(2):
      for lens, slot in enumerate(rig.lenses):
        taken[env_idx][lens].extend(frame.at for frame in rig.exposed[env_idx][slot])
  for env_idx in range(2):
    apart = np.asarray(taken[env_idx][1][:5]) - np.asarray(taken[env_idx][0][:5])
    assert np.allclose(apart, apart[0], atol=0.1), "a pair holds its offset"
    want = (phases[env_idx, 1] - phases[env_idx, 0]) % 1.0 * SPAN
    assert abs(float(apart[0]) % SPAN - want) < 0.2, "the phase sets it"
  assert taken[0][0][:5] != taken[1][0][:5], "two envs draw their own phase"
  instants = [at for per_env in taken for lens in per_env for at in lens]
  between = sum(abs(at - round(at)) > 1e-9 for at in instants)
  assert between > len(instants) // 2, (
    "most shutters fall between rows, as a capture's do"
  )


def test_one_env_restarts_while_the_others_carry_on(embodiment: Embodiment) -> None:
  """An evaluation runs many episodes at once and they do not end together."""
  rig = Rig(replace(SMALL, embodiment=embodiment), num_envs=3)
  observation = _begin(rig, 3, nudge=0.01)
  rows = rig.cfg.clock.per_step
  for _ in range(3):
    observation = rig.step(hold(observation.state[:, -1]))
  assert observation.row[:, -1].tolist() == [3 * rows] * 3
  carried = observation.state[[0, 2], -1].copy()

  rig.place(1, HOME)
  observation = rig.begin([1])
  assert observation.row[1, -1] == 0, "the restarted env keeps counting"
  assert observation.row[0, -1] == 3 * rows, "the others were restarted with it"
  assert observation.image_row[:, 0, -1].tolist() == [3 * rows, 0, 3 * rows]
  assert np.array_equal(observation.state[[0, 2], -1], carried)

  observation = rig.step(hold(observation.state[:, -1]))
  assert observation.row[:, -1].tolist() == [4 * rows, rows, 4 * rows]


def test_an_env_in_a_batch_walks_the_path_it_walks_alone(
  embodiment: Embodiment,
) -> None:
  """Envs share nothing but the model, so company cannot change one."""
  alone = Rig(replace(SMALL, embodiment=embodiment))
  crowd = Rig(replace(SMALL, embodiment=type(embodiment)()), num_envs=3)
  one = _begin(alone)
  crowd.place(0, HOME + 0.02)
  crowd.place(1, HOME)
  crowd.place(2, HOME - 0.02)
  many = crowd.begin([0, 1, 2])
  assert np.array_equal(one.image[0], many.image[1])
  for _ in range(4):
    act_a = hold(one.state[:, -1])
    act_c = hold(many.state[:, -1])
    act_c[1] = act_a[0]
    one = alone.step(act_a)
    many = crowd.step(act_c)
    assert np.array_equal(alone.datas[0].qpos, crowd.datas[1].qpos)
    assert np.array_equal(one.image[0], many.image[1])
    assert np.array_equal(one.state[0], many.state[1])


def test_an_env_that_holds_where_it_is_never_goes_unstable() -> None:
  rig = Rig(SMALL, num_envs=2)
  observation = _begin(rig, 2)
  for _ in range(6):
    observation = rig.step(hold(observation.state[:, -1]))
    assert not rig.unstable.any() and rig.tracking_error.max() < 0.002


def test_the_rows_of_a_step_are_kept_for_a_reader() -> None:
  rig = Rig(SMALL, num_envs=2)
  observation = _begin(rig, 2)
  model, rows = rig.model, rig.cfg.clock.per_step
  assert rig.qpos_rows.shape == (2, rows, model.nq)
  observation = rig.step(hold(observation.state[:, -1]))
  for env_idx, data in enumerate(rig.datas):
    assert np.array_equal(rig.qpos_rows[env_idx, -1], data.qpos)
    assert np.array_equal(rig.qvel_rows[env_idx, -1], data.qvel)
  assert np.array_equal(rig.states(rig.qpos_rows[0]), observation.state[0])
  assert rig.twists(rig.qpos_rows[0], rig.qvel_rows[0]).shape == (rows, 2, 6)


def test_the_rig_runs_under_the_physics_it_is_given() -> None:
  rig = Rig(replace(SMALL, physics=Physics(impratio=50.0)))
  model = rig.model
  assert model.opt.impratio == 50.0
  assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  pads = model.geom_condim == GRIPPING
  assert pads.any() and (model.geom_priority[pads] == 1).all()
  assert model.geom_solref[pads] == pytest.approx(
    np.tile(Physics().pad_solref, (pads.sum(), 1))
  )
  rig_dof = model.body(f"{SLOTS[0]}_rig").dofadr[0]
  assert model.dof_damping[rig_dof] == Floating().damping, (
    "an embodiment keeps its own damping"
  )


def test_tracing_only_the_delivered_exposures_delivers_the_same_frames() -> None:
  """Off for a recorder, the rig traces an exposure only if an observation
  will hand it over, which changes nothing the policy sees."""
  for latency in (0.0, 0.07):
    every = Rig(replace(SMALL, camera_latency=latency, frames=2))
    lazy = Rig(replace(SMALL, camera_latency=latency, frames=2, record=False))
    seen = [_begin(every), _begin(lazy)]
    traced = 0
    for _ in range(8):
      action = hold(seen[0].state[0, -1])[None]
      action[..., 0, 2] += 0.04
      seen = [every.step(action), lazy.step(action)]
      assert np.array_equal(seen[0].image_row, seen[1].image_row)
      assert np.array_equal(seen[0].image, seen[1].image)
      traced += sum(len(frames) for frames in every.exposed[0].values())
      assert not any(lazy.exposed[0].values()), "nothing is kept for a recorder"
    assert traced > 2 * 2 * 8, "every exposure was traced for the recorder"


def test_an_observation_carries_each_lens_newest_frames() -> None:
  rig = Rig(replace(SMALL, frames=3))
  observation = _begin(rig)
  for _ in range(4):
    observation = rig.step(hold(observation.state[0, -1])[None])
    rows = observation.image_row[0, 0]
    assert observation.image.shape[2] == 3
    assert (np.diff(rows) > 0).all() and (np.diff(rows) <= SPAN + 1).all()
    exposed = [frame.row for frame in rig.exposed[0][SLOTS[0]]]
    assert rows[-1] == exposed[-1], "the newest is the last exposure"


def test_a_rig_without_lenses_traces_nothing() -> None:
  rig = Rig(replace(SMALL, render=False))
  observation = _begin(rig)
  observation = rig.step(hold(observation.state[0, -1])[None])
  assert rig._renderer is None and rig.lenses == []
  assert observation.image.shape[:3] == (1, 0, 1) and observation.state.shape[2] == 2
