"""Unit quaternions ``wxyz`` over any leading axes, and single poses ``(7,)``,
a position then a quaternion."""

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array


def conjugate(q: Float64Array) -> Float64Array:
  return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def multiply(a: Float64Array, b: Float64Array) -> Float64Array:
  """``a ⊗ b``."""
  w1, v1 = a[..., :1], a[..., 1:]
  w2, v2 = b[..., :1], b[..., 1:]
  w = w1 * w2 - np.sum(v1 * v2, axis=-1, keepdims=True)
  v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
  return np.concatenate([w, v], axis=-1)


def rotate(q: Float64Array, v: Float64Array) -> Float64Array:
  """Each ``v`` carried through its ``q``."""
  w, u = q[..., :1], q[..., 1:]
  lift = np.cross(u, v)
  return v + 2.0 * (w * lift + np.cross(u, lift))


def exp(v: Float64Array) -> Float64Array:
  """The quaternion of a rotation vector."""
  half = 0.5 * np.linalg.norm(v, axis=-1, keepdims=True)
  scale = np.sinc(half / np.pi)  # sin(half) / half
  return np.concatenate([np.cos(half), 0.5 * scale * v], axis=-1)


def log(q: Float64Array) -> Float64Array:
  """The rotation vector of a unit quaternion, by the shorter way round."""
  q = np.where(q[..., :1] < 0.0, -q, q)
  swing = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
  angle = 2.0 * np.arctan2(swing, q[..., :1])
  # angle / swing, which tends to 2 as the swing vanishes.
  scale = np.where(swing > 1e-12, angle / np.maximum(swing, 1e-12), 2.0)
  return scale * q[..., 1:]


def from_matrix(rotation: Float64Array) -> Float64Array:
  """One rotation matrix's quaternion."""
  out = np.empty(4)
  mujoco.mju_mat2Quat(out, np.ascontiguousarray(rotation, np.float64).ravel())
  return out


def to_matrix(q: Float64Array) -> Float64Array:
  """One quaternion's rotation matrix, ``(3, 3)``."""
  out = np.empty(9)
  mujoco.mju_quat2Mat(out, q)
  return out.reshape(3, 3)


def compose(a: Float64Array, b: Float64Array) -> Float64Array:
  """Pose ``b``, given in pose ``a``'s frame, in the frame ``a`` is given in."""
  out = np.empty(7)
  mujoco.mju_mulPose(out[:3], out[3:], a[:3], a[3:7], b[:3], b[3:7])
  return out


def inverse(pose: Float64Array) -> Float64Array:
  """The frame ``pose`` is given in, as a pose in ``pose``'s own frame."""
  out = np.empty(7)
  mujoco.mju_negPose(out[:3], out[3:], pose[:3], pose[3:7])
  return out


def interpolate(a: Float64Array, b: Float64Array, t: float) -> Float64Array:
  """The pose ``t`` of the way from ``a`` to ``b``, extrapolated past 1."""
  turn = np.empty(3)
  mujoco.mju_subQuat(turn, b[3:7], a[3:7])
  out = np.concatenate([a[:3] + t * (b[:3] - a[:3]), a[3:7]])
  mujoco.mju_quatIntegrate(out[3:], turn, t)
  return out


def pointing(fingers: Float64Array) -> Float64Array:
  """The orientation whose x (the fingers) points along ``fingers`` and whose z
  (the lens side) stands as far up as that allows. Straight down or up, z
  faces +x."""
  x = fingers / np.linalg.norm(fingers)
  up = np.array([0.0, 0.0, 1.0])
  z = up - np.dot(up, x) * x
  if np.linalg.norm(z) < 1e-6:
    z = np.array([1.0, 0.0, 0.0]) - x[0] * x
  z /= np.linalg.norm(z)
  return from_matrix(np.stack([x, np.cross(z, x), z], axis=1))
