"""The interpolation: an action chunk laid onto the rows of a control step."""

import numpy as np
import pytest

from specialist_sim import rotation
from specialist_sim.env import Clock, hold, interpolate

# One env's one hand, (1, 1, 8).
START = np.array([[[0.5, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0, 0.2]]])
# Five actions a step, each of sixty rows, rows a physics step long.
CLOCK = Clock(per_row=1, per_action=60, per_step=300)


def test_the_rows_pass_through_every_action_they_execute() -> None:
  plan = hold(START, CLOCK.n_action_steps)
  plan[:, :, 0, 2] += 0.02 * np.arange(1, CLOCK.n_action_steps + 1)
  setpoint, twist = interpolate(START, plan, CLOCK)
  rise = setpoint[0, :, 0, 2]
  for action in range(CLOCK.n_action_steps):
    end = (action + 1) * CLOCK.per_action - 1
    assert rise[end] == pytest.approx(plan[0, action, 0, 2])
  assert (np.diff(rise) > 0).all(), "the setpoint jumped rather than ramped"
  assert rise[0] - START[0, 0, 2] == pytest.approx(0.02 / 60)
  action_dt = CLOCK.row_dt * CLOCK.per_action
  assert twist[0, :, 0, 2] == pytest.approx(0.02 / action_dt)
  assert setpoint[0, -1, 0] == pytest.approx(plan[0, -1, 0])


def test_a_coarser_clock_is_the_finer_one_sampled() -> None:
  rng = np.random.default_rng(0)
  fine = Clock(per_row=1, per_action=40, per_step=200)
  coarse = Clock(per_row=4, per_action=10, per_step=50)
  last = START
  for _ in range(3):
    plan = hold(START, 5) + rng.normal(scale=0.02, size=(1, 5, 1, 8))
    for many, few in zip(
      interpolate(last, plan, fine), interpolate(last, plan, coarse), strict=True
    ):
      assert np.allclose(few[0], many[0, 3::4], atol=1e-12)
    last = interpolate(last, plan, fine)[0][:, -1]


def test_a_turn_is_interpolated_along_the_geodesic() -> None:
  clock = Clock(per_row=1, per_action=10, per_step=10)
  plan = hold(START, 1)
  plan[:, :, 0, 3:7] = 2.0 * rotation.exp(np.array([0.0, 0.0, 0.5]))
  setpoint, twist = interpolate(START, plan, clock)
  assert np.linalg.norm(setpoint[0, :, 0, 3:7], axis=-1) == pytest.approx(1.0)
  angles = rotation.log(setpoint[0, :, 0, 3:7])[:, 2]
  assert angles == pytest.approx(0.05 * np.arange(1, 11))
  assert twist[0, :, 0, 3:] == pytest.approx(
    np.tile([0.0, 0.0, 0.5 / clock.step_dt], (10, 1))
  )
