"""The interpolator: an action chunk laid onto the substeps of a control step."""

import numpy as np
import pytest

from specialist_sim import rotation
from specialist_sim.action import CHUNK_SIZE, Interpolator, hold

START = np.array([[[0.5, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0, 0.2]]])
"""One env's one hand, ``(1, 1, 8)``."""


def test_a_held_state_names_every_action() -> None:
  plan = hold(START[0])
  assert plan.shape == (CHUNK_SIZE, 1, 8)
  assert (plan == START[0, 0]).all()
  assert hold(START, 4).shape == (1, 4, 1, 8)


def test_the_interpolation_passes_through_every_action_it_executes() -> None:
  interpolator = Interpolator(
    num_envs=1, hands=1, n_action_steps=5, span=60, action_dt=0.02
  )
  interpolator.hold([0], START)
  plan = hold(START)
  plan[:, :, 0, 2] += 0.02 * np.arange(1, CHUNK_SIZE + 1)
  interpolator.process(plan)
  rise = interpolator.position[0, :, 0, 2]
  for action in range(interpolator.n_action_steps):
    assert rise[(action + 1) * interpolator.span - 1] == pytest.approx(
      plan[0, action, 0, 2]
    )
  assert (np.diff(rise) > 0).all(), "the setpoint jumped rather than ramped"
  assert rise[0] - START[0, 0, 2] == pytest.approx(0.02 / 60)
  assert interpolator.twist[0, :, 0, 2] == pytest.approx(0.02 / 0.02)
  assert interpolator.last[0, 0] == pytest.approx(
    plan[0, interpolator.n_action_steps - 1, 0]
  )


def test_a_coarser_interpolation_is_the_finer_one_sampled() -> None:
  rng = np.random.default_rng(0)
  fine = Interpolator(num_envs=1, hands=1, n_action_steps=5, span=40, action_dt=0.02)
  coarse = Interpolator(num_envs=1, hands=1, n_action_steps=5, span=10, action_dt=0.02)
  for interpolator in (fine, coarse):
    interpolator.hold([0], START)
  for _ in range(3):
    plan = hold(START) + rng.normal(scale=0.02, size=(1, CHUNK_SIZE, 1, 8))
    fine.process(plan)
    coarse.process(plan)
    for name in ("position", "quaternion", "opening", "twist"):
      sampled = getattr(fine, name)[0, 3::4]
      assert np.allclose(getattr(coarse, name)[0], sampled, atol=1e-12)


def test_a_turn_is_interpolated_along_the_geodesic() -> None:
  interpolator = Interpolator(
    num_envs=1, hands=1, n_action_steps=1, span=10, action_dt=0.1
  )
  interpolator.hold([0], START)
  plan = hold(START)
  plan[:, :, 0, 3:7] = 2.0 * rotation.exp(np.array([0.0, 0.0, 0.5]))
  interpolator.process(plan)
  assert np.linalg.norm(interpolator.quaternion[0], axis=-1) == pytest.approx(1.0)
  angles = rotation.log(interpolator.quaternion[0, :, 0])[:, 2]
  assert angles == pytest.approx(0.05 * np.arange(1, 11))
  assert interpolator.twist[0, :, 0, 3:] == pytest.approx(
    np.tile([0.0, 0.0, 5.0], (10, 1))
  )


def test_a_new_action_chunk_departs_from_where_the_last_ended() -> None:
  interpolator = Interpolator(
    num_envs=2, hands=1, n_action_steps=2, span=5, action_dt=0.1
  )
  interpolator.hold([0, 1], np.repeat(START, 2, axis=0))
  plan = hold(np.repeat(START, 2, axis=0), 4)
  plan[1, :, 0, 0] += 1.0
  interpolator.process(plan)
  ended = interpolator.last.copy()
  interpolator.process(hold(ended, 4))
  assert (interpolator.position[0, :, 0, 0] == 0.5).all()
  assert (interpolator.position[1, :, 0, 0] == 1.5).all()
  assert (interpolator.twist == 0.0).all()
