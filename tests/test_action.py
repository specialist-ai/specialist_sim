"""Decoding a command: rotations, the gripper map, and the setpoint ramp."""

import numpy as np
import pytest

from specialist_sim import pose as se3
from specialist_sim.action import Ramp, _apart, hold, turn
from specialist_sim.spaces import (
  FINGER_OPEN,
  FINGER_SHUT,
  HORIZON,
  PER_HAND,
  gripper_from_opening,
  opening_from_gripper,
)


def _rotations(count: int) -> list[np.ndarray]:
  rng = np.random.default_rng(0)
  out = []
  for _ in range(count):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    out.append(q * np.sign(np.linalg.det(q)))
  return out


@pytest.mark.parametrize("rotation", _rotations(8))
def test_a_rotation_survives_the_four_number_round_trip(rotation: np.ndarray) -> None:
  assert se3.rotation(se3.wxyz(rotation)) == pytest.approx(rotation, abs=1e-12)


def test_a_quaternion_is_stored_scalar_first() -> None:
  """A quarter turn about z, written out, so the order is pinned to a literal."""
  quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  root = np.sqrt(0.5)
  assert se3.wxyz(quarter) == pytest.approx([root, 0.0, 0.0, root], abs=1e-12)


@pytest.mark.parametrize(
  "four", [np.zeros(4), np.array([2.0, 0.0, 0.0, 0.0]), np.array([0.0, 3.0, 4.0, 0.0])]
)
def test_a_command_of_any_length_still_names_a_rotation(four: np.ndarray) -> None:
  unit = turn(four)
  assert np.linalg.norm(unit) == pytest.approx(1.0)
  rotation = se3.rotation(unit)
  assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-12)
  assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_a_scaled_command_names_the_same_rotation() -> None:
  """Length carries no meaning, so scaling a command may not turn the hand."""
  quarter = se3.wxyz(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
  for scale in (0.1, 1.0, 7.0):
    assert se3.rotation(turn(quarter * scale)) == pytest.approx(
      se3.rotation(quarter), abs=1e-12
    )


def test_the_gripper_command_spans_shut_to_open() -> None:
  assert gripper_from_opening(0.0) == pytest.approx(FINGER_SHUT)
  assert gripper_from_opening(1.0) == pytest.approx(FINGER_OPEN)
  assert opening_from_gripper(gripper_from_opening(0.37)) == pytest.approx(0.37)


def test_a_half_turn_of_error_asks_for_a_half_turn_of_correction() -> None:
  """The largest error a rotation can hold is as sound as the smallest."""
  here = np.array([[1.0, 0.0, 0.0, 0.0]])
  for axis in np.eye(3):
    want = np.concatenate([[0.0], axis])[None]
    swing = _apart(want, here)
    assert np.isfinite(swing).all()
    assert np.linalg.norm(swing) == pytest.approx(np.pi)
    assert np.allclose(swing[0] / np.linalg.norm(swing), axis)
  assert np.allclose(_apart(here, here), 0.0)


def test_a_coarser_ramp_is_the_finer_one_sampled() -> None:
  """A ramp laid on a follower's coarser clock passes through the same setpoints."""
  rng = np.random.default_rng(0)
  rates = {"control_dt": 0.1, "max_speed": 1.5, "max_turn": 10.0, "worlds": 1}
  fine = Ramp(hands=1, substeps=200, action_hz=50.0, **rates)
  coarse = Ramp(hands=1, substeps=50, action_hz=50.0, **rates)
  stride = fine.substeps // coarse.substeps
  start = np.array([[[0.5, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0, 0.2]]])
  for ramp in (fine, coarse):
    ramp.hold([0], start)
  for _ in range(3):
    plan = hold(start[0, 0]) + rng.normal(scale=0.02, size=(HORIZON, PER_HAND))
    fine.process(plan, (0,))
    coarse.process(plan, (0,))
    for name in ("position", "quaternion", "opening"):
      sampled = getattr(fine, name)[0, stride - 1 :: stride]
      assert np.allclose(getattr(coarse, name)[0], sampled, atol=1e-12)
