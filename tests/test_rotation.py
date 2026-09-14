"""Quaternions: the round trips, and the half turn."""

import numpy as np
import pytest

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array


def _rotations(count: int) -> Float64Array:
  rng = np.random.default_rng(0)
  out = []
  for _ in range(count):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    out.append(q * np.sign(np.linalg.det(q)))
  return np.stack(out)


def test_a_rotation_survives_every_round_trip() -> None:
  matrices = _rotations(8)
  quaternions = np.stack([rotation.from_matrix(m) for m in matrices])
  assert rotation.exp(rotation.log(quaternions)) == pytest.approx(
    np.where(quaternions[:, :1] < 0.0, -quaternions, quaternions), abs=1e-12
  )
  vectors = np.random.default_rng(1).normal(size=(8, 3))
  turned = np.einsum("nij,nj->ni", matrices, vectors)
  assert rotation.rotate(quaternions, vectors) == pytest.approx(turned, abs=1e-12)
  product = rotation.multiply(quaternions[:4], quaternions[4:])
  assert np.stack([rotation.to_matrix(q) for q in product]) == pytest.approx(
    matrices[:4] @ matrices[4:], abs=1e-12
  )


def test_a_half_turn_has_a_rotation_vector_of_pi() -> None:
  for axis in np.eye(3):
    half = np.concatenate([[0.0], axis])
    assert rotation.log(half) == pytest.approx(np.pi * axis)
    assert np.abs(rotation.log(-half)) == pytest.approx(np.pi * axis)
