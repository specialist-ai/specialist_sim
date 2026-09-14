"""Rotations as unit quaternions, ``wxyz``, over any leading axes."""

from __future__ import annotations

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array

__all__ = [
  "conjugate",
  "exp",
  "from_matrix",
  "log",
  "multiply",
  "normalize",
  "rotate",
  "to_matrix",
]


def normalize(q: Float64Array) -> Float64Array:
  out: Float64Array = q / np.linalg.norm(q, axis=-1, keepdims=True)
  return out


def conjugate(q: Float64Array) -> Float64Array:
  out: Float64Array = q * np.array([1.0, -1.0, -1.0, -1.0])
  return out


def multiply(a: Float64Array, b: Float64Array) -> Float64Array:
  """``a ⊗ b``."""
  w1, v1 = a[..., :1], a[..., 1:]
  w2, v2 = b[..., :1], b[..., 1:]
  w = w1 * w2 - np.sum(v1 * v2, axis=-1, keepdims=True)
  v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
  out: Float64Array = np.concatenate([w, v], axis=-1)
  return out


def rotate(q: Float64Array, v: Float64Array) -> Float64Array:
  """Each ``v`` carried through its ``q``."""
  w, u = q[..., :1], q[..., 1:]
  lift = np.cross(u, v)
  out: Float64Array = v + 2.0 * (w * lift + np.cross(u, lift))
  return out


def exp(v: Float64Array) -> Float64Array:
  """The quaternion of a rotation vector."""
  half = 0.5 * np.linalg.norm(v, axis=-1, keepdims=True)
  # sin(half) / half, which tends to one as the angle vanishes.
  scale = np.where(half > 1e-12, np.sin(half) / np.where(half > 0.0, half, 1.0), 1.0)
  out: Float64Array = np.concatenate([np.cos(half), 0.5 * scale * v], axis=-1)
  return out


def log(q: Float64Array) -> Float64Array:
  """The rotation vector of a unit quaternion, by the shorter way round."""
  q = np.where(q[..., :1] < 0.0, -q, q)
  swing = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
  angle = 2.0 * np.arctan2(swing, q[..., :1])
  # angle / swing, which tends to two as the swing vanishes.
  scale = np.where(swing > 1e-12, angle / np.where(swing > 0.0, swing, 1.0), 2.0)
  out: Float64Array = scale * q[..., 1:]
  return out


def to_matrix(q: Float64Array) -> Float64Array:
  w, x, y, z = (q[..., i] for i in range(4))
  out: Float64Array = np.stack(
    [
      np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
      np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
      np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ],
    axis=-2,
  )
  return out


def from_matrix(rotation: Float64Array) -> Float64Array:
  """One rotation matrix's quaternion."""
  out = np.empty(4)
  mujoco.mju_mat2Quat(out, np.ascontiguousarray(rotation, np.float64).ravel())
  return out
