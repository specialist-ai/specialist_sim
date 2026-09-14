"""Quaternions: the convention, and the round trips."""

import numpy as np
import pytest

from specialist_sim import rotation


def _rotations(count: int) -> np.ndarray:
  rng = np.random.default_rng(0)
  out = []
  for _ in range(count):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    out.append(q * np.sign(np.linalg.det(q)))
  return np.stack(out)


def test_a_quaternion_is_stored_scalar_first() -> None:
  quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  root = np.sqrt(0.5)
  assert rotation.from_matrix(quarter) == pytest.approx([root, 0.0, 0.0, root])
  assert rotation.to_matrix(rotation.from_matrix(quarter)) == pytest.approx(quarter)


def test_a_rotation_survives_every_round_trip() -> None:
  matrices = _rotations(8)
  quaternions = np.stack([rotation.from_matrix(m) for m in matrices])
  assert rotation.to_matrix(quaternions) == pytest.approx(matrices, abs=1e-12)
  assert rotation.exp(rotation.log(quaternions)) == pytest.approx(
    np.where(quaternions[:, :1] < 0.0, -quaternions, quaternions), abs=1e-12
  )
  vectors = np.random.default_rng(1).normal(size=(8, 3))
  turned = np.einsum("nij,nj->ni", matrices, vectors)
  assert rotation.rotate(quaternions, vectors) == pytest.approx(turned, abs=1e-12)
  product = rotation.multiply(quaternions[:4], quaternions[4:])
  assert rotation.to_matrix(product) == pytest.approx(
    matrices[:4] @ matrices[4:], abs=1e-12
  )


def test_a_half_turn_is_as_sound_as_no_turn() -> None:
  for axis in np.eye(3):
    half = np.concatenate([[0.0], axis])
    assert rotation.log(half) == pytest.approx(np.pi * axis)
    assert np.abs(rotation.log(-half)) == pytest.approx(np.pi * axis)
  assert rotation.log(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(np.zeros(3))
  assert rotation.exp(np.zeros(3)) == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_normalizing_keeps_the_direction() -> None:
  scaled = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 4.0, 0.0]])
  unit = rotation.normalize(scaled)
  assert np.linalg.norm(unit, axis=1) == pytest.approx([1.0, 1.0])
  assert unit[1] == pytest.approx([0.0, 0.6, 0.8, 0.0])
