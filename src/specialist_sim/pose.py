"""Rigid transforms as ``(4, 4)`` matrices."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array

__all__ = ["inverse", "log", "of", "rotation", "wxyz"]


def of(
  rotation: Float64Array, translation: Sequence[float] | Float64Array
) -> Float64Array:
  out = np.eye(4)
  out[:3, :3] = rotation
  out[:3, 3] = translation
  return out


def inverse(pose: Float64Array) -> Float64Array:
  rotation = pose[:3, :3]
  return of(rotation.T, -rotation.T @ pose[:3, 3])


def wxyz(rotation: Float64Array) -> Float64Array:
  out = np.empty(4)
  mujoco.mju_mat2Quat(out, np.ascontiguousarray(rotation).ravel())
  return out


def rotation(quaternion: Float64Array) -> Float64Array:
  """The rotation a quaternion names."""
  out = np.empty(9)
  mujoco.mju_quat2Mat(out, np.ascontiguousarray(quaternion, np.float64))
  return out.reshape(3, 3)


def log(rotation: Float64Array) -> Float64Array:
  """The rotation vector whose norm is the rotation's angle."""
  quat, out = wxyz(rotation), np.empty(3)
  mujoco.mju_quat2Vel(out, quat, 1.0)
  return out
