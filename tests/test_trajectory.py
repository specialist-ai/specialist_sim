"""The ramp: a trajectory laid onto the substeps of a control step."""

import numpy as np
import pytest

from specialist_sim import rotation
from specialist_sim.trajectory import HORIZON, Ramp, hold

START = np.array([[[0.5, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0, 0.2]]])
"""One world's one hand, ``(1, 1, 8)``."""


def test_a_held_state_names_every_pose() -> None:
  plan = hold(START[0])
  assert plan.shape == (HORIZON, 1, 8)
  assert (plan == START[0, 0]).all()
  assert hold(START, 4).shape == (1, 4, 1, 8)


def test_the_ramp_passes_through_every_pose_it_tracks() -> None:
  ramp = Ramp(worlds=1, hands=1, tracked=5, span=60, pose_dt=0.02)
  ramp.hold([0], START)
  plan = hold(START)
  plan[:, :, 0, 2] += 0.02 * np.arange(1, HORIZON + 1)
  ramp.process(plan)
  rise = ramp.position[0, :, 0, 2]
  for leg in range(ramp.tracked):
    assert rise[(leg + 1) * ramp.span - 1] == pytest.approx(plan[0, leg, 0, 2])
  assert (np.diff(rise) > 0).all(), "the setpoint jumped rather than ramped"
  assert rise[0] - START[0, 0, 2] == pytest.approx(0.02 / 60)
  assert ramp.twist[0, :, 0, 2] == pytest.approx(0.02 / 0.02)
  assert ramp.last[0, 0] == pytest.approx(plan[0, ramp.tracked - 1, 0])


def test_a_coarser_ramp_is_the_finer_one_sampled() -> None:
  rng = np.random.default_rng(0)
  fine = Ramp(worlds=1, hands=1, tracked=5, span=40, pose_dt=0.02)
  coarse = Ramp(worlds=1, hands=1, tracked=5, span=10, pose_dt=0.02)
  for ramp in (fine, coarse):
    ramp.hold([0], START)
  for _ in range(3):
    plan = hold(START) + rng.normal(scale=0.02, size=(1, HORIZON, 1, 8))
    fine.process(plan)
    coarse.process(plan)
    for name in ("position", "quaternion", "opening", "twist"):
      sampled = getattr(fine, name)[0, 3::4]
      assert np.allclose(getattr(coarse, name)[0], sampled, atol=1e-12)


def test_a_turn_is_ramped_along_the_geodesic() -> None:
  ramp = Ramp(worlds=1, hands=1, tracked=1, span=10, pose_dt=0.1)
  ramp.hold([0], START)
  plan = hold(START)
  plan[:, :, 0, 3:7] = 2.0 * rotation.exp(np.array([0.0, 0.0, 0.5]))
  ramp.process(plan)
  assert np.linalg.norm(ramp.quaternion[0], axis=-1) == pytest.approx(1.0)
  angles = rotation.log(ramp.quaternion[0, :, 0])[:, 2]
  assert angles == pytest.approx(0.05 * np.arange(1, 11))
  assert ramp.twist[0, :, 0, 3:] == pytest.approx(np.tile([0.0, 0.0, 5.0], (10, 1)))


def test_a_new_trajectory_departs_from_where_the_last_ended() -> None:
  ramp = Ramp(worlds=2, hands=1, tracked=2, span=5, pose_dt=0.1)
  ramp.hold([0, 1], np.repeat(START, 2, axis=0))
  plan = hold(np.repeat(START, 2, axis=0), 4)
  plan[1, :, 0, 0] += 1.0
  ramp.process(plan)
  ended = ramp.last.copy()
  ramp.process(hold(ended, 4))
  assert (ramp.position[0, :, 0, 0] == 0.5).all()
  assert (ramp.position[1, :, 0, 0] == 1.5).all()
  assert (ramp.twist == 0.0).all()
