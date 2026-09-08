"""The MEI fisheye: a unit-sphere mirror behind a distorted pinhole.

Distortion order matches the camera's own lens block, ``[k1, k2, k3, p1, p2]``
with the mirror parameter ``xi`` past it. Points outside the model's domain, where the map stops being one to one,
come back as NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from specialist_sim.arrays import Float64Array

__all__ = ["MeiParams", "mei_project", "mei_unproject", "rescale"]


@dataclass(frozen=True)
class MeiParams:
  fx: float
  fy: float
  cx: float
  cy: float
  k1: float = 0.0
  k2: float = 0.0
  k3: float = 0.0
  p1: float = 0.0
  p2: float = 0.0
  xi: float = 0.0


def rescale(params: MeiParams, scale: float) -> MeiParams:
  """The same lens on an image scaled by ``scale``. Distortion and the mirror
  act on normalized coordinates."""
  return replace(
    params,
    fx=params.fx * scale,
    fy=params.fy * scale,
    cx=params.cx * scale,
    cy=params.cy * scale,
  )


def _unit(rows: Float64Array) -> Float64Array:
  norm = np.linalg.norm(rows, axis=1, keepdims=True)
  return np.asarray(rows / np.where(norm == 0.0, 1.0, norm), np.float64)


def mei_project(points: Float64Array, params: MeiParams) -> Float64Array:
  sphere = _unit(np.asarray(points, np.float64).reshape(-1, 3))
  denom = sphere[:, 2] + params.xi
  valid = (denom > 0.0) & (1.0 + params.xi * sphere[:, 2] > 0.0)
  x = np.where(valid, sphere[:, 0] / denom, np.nan)
  y = np.where(valid, sphere[:, 1] / denom, np.nan)
  r2 = x * x + y * y
  radial = 1.0 + params.k1 * r2 + params.k2 * r2 * r2 + params.k3 * r2**3
  distorted_x = x * radial + 2.0 * params.p1 * x * y + params.p2 * (r2 + 2.0 * x * x)
  distorted_y = y * radial + params.p1 * (r2 + 2.0 * y * y) + 2.0 * params.p2 * x * y
  return np.stack(
    [params.fx * distorted_x + params.cx, params.fy * distorted_y + params.cy], 1
  )


def mei_unproject(pixels: Float64Array, params: MeiParams) -> Float64Array:
  flat = np.asarray(pixels, np.float64).reshape(-1, 2)
  target_x = (flat[:, 0] - params.cx) / params.fx
  target_y = (flat[:, 1] - params.cy) / params.fy
  x, y = target_x, target_y
  for _ in range(20):
    r2 = x * x + y * y
    radial = 1.0 + params.k1 * r2 + params.k2 * r2 * r2 + params.k3 * r2**3
    tangential_x = 2.0 * params.p1 * x * y + params.p2 * (r2 + 2.0 * x * x)
    tangential_y = params.p1 * (r2 + 2.0 * y * y) + 2.0 * params.p2 * x * y
    safe = np.where(radial == 0.0, 1.0, radial)
    next_x, next_y = (target_x - tangential_x) / safe, (target_y - tangential_y) / safe
    step = max(np.max(np.abs(next_x - x)), np.max(np.abs(next_y - y)))
    x, y = next_x, next_y
    if step < 1e-12:
      break
  r2 = x * x + y * y
  # f solves f**2 * (r2 + 1) - 2*f*xi + xi**2 - 1 = 0 for the unit sphere point.
  discriminant = 1.0 + (1.0 - params.xi * params.xi) * r2
  factor = (params.xi + np.sqrt(np.where(discriminant < 0.0, np.nan, discriminant))) / (
    r2 + 1.0
  )
  return _unit(np.stack([factor * x, factor * y, factor - params.xi], axis=1))
