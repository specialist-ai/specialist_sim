"""The camera a hand carries: the MEI fisheye it projects through, where it
sits on the base, and how fast it fires. Distortion runs ``[k1, k2, k3, p1,
p2]`` with the mirror parameter ``xi`` past it, at the width it was measured at.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from specialist_sim.arrays import Float32Array, Float64Array, Int32Array

__all__ = ["NOMINAL", "Camera", "Mei", "Rays", "project", "unproject"]


@dataclass(frozen=True)
class Mei:
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

  def scaled(self, factor: float) -> Mei:
    """The same lens on an image scaled by ``factor``."""
    return replace(
      self,
      fx=self.fx * factor,
      fy=self.fy * factor,
      cx=self.cx * factor,
      cy=self.cy * factor,
    )


def _unit(rows: Float64Array) -> Float64Array:
  norm = np.linalg.norm(rows, axis=1, keepdims=True)
  out: Float64Array = rows / np.where(norm == 0.0, 1.0, norm)
  return out


def project(points: Float64Array, mei: Mei) -> Float64Array:
  """Each point's pixel, ``(n, 3)`` to ``(n, 2)``; NaN where the map is not
  one to one."""
  sphere = _unit(np.asarray(points, np.float64).reshape(-1, 3))
  denom = sphere[:, 2] + mei.xi
  valid = (denom > 0.0) & (1.0 + mei.xi * sphere[:, 2] > 0.0)
  x = np.where(valid, sphere[:, 0] / denom, np.nan)
  y = np.where(valid, sphere[:, 1] / denom, np.nan)
  r2 = x * x + y * y
  radial = 1.0 + mei.k1 * r2 + mei.k2 * r2 * r2 + mei.k3 * r2**3
  dx = x * radial + 2.0 * mei.p1 * x * y + mei.p2 * (r2 + 2.0 * x * x)
  dy = y * radial + mei.p1 * (r2 + 2.0 * y * y) + 2.0 * mei.p2 * x * y
  return np.stack([mei.fx * dx + mei.cx, mei.fy * dy + mei.cy], 1)


def unproject(pixels: Float64Array, mei: Mei) -> Float64Array:
  """Each pixel's unit bearing, ``(n, 2)`` to ``(n, 3)``, x right, y down,
  z forward."""
  flat = np.asarray(pixels, np.float64).reshape(-1, 2)
  target_x = (flat[:, 0] - mei.cx) / mei.fx
  target_y = (flat[:, 1] - mei.cy) / mei.fy
  x, y = target_x, target_y
  for _ in range(20):
    r2 = x * x + y * y
    radial = 1.0 + mei.k1 * r2 + mei.k2 * r2 * r2 + mei.k3 * r2**3
    tangential_x = 2.0 * mei.p1 * x * y + mei.p2 * (r2 + 2.0 * x * x)
    tangential_y = mei.p1 * (r2 + 2.0 * y * y) + 2.0 * mei.p2 * x * y
    safe = np.where(radial == 0.0, 1.0, radial)
    next_x, next_y = (target_x - tangential_x) / safe, (target_y - tangential_y) / safe
    step = max(np.max(np.abs(next_x - x)), np.max(np.abs(next_y - y)))
    x, y = next_x, next_y
    if step < 1e-12:
      break
  r2 = x * x + y * y
  discriminant = 1.0 + (1.0 - mei.xi * mei.xi) * r2
  factor = (mei.xi + np.sqrt(np.where(discriminant < 0.0, np.nan, discriminant))) / (
    r2 + 1.0
  )
  return _unit(np.stack([factor * x, factor * y, factor - mei.xi], axis=1))


@dataclass(frozen=True)
class Rays:
  """One bearing per pixel a lens draws, at one output size."""

  bearings: Float32Array
  """``(n, 3)`` unit bearings in the lens frame."""
  pixel: Int32Array
  """Where each bearing's ray lands in the flattened image."""


ROUND_TRIP = 1e-3
"""How far a pixel may move through the round trip and still be a bearing."""


@dataclass(frozen=True)
class Camera:
  """A lens, and its seat in the frame of the body carrying it."""

  mei: Mei = Mei(
    fx=2250.0,
    fy=2250.0,
    cx=1456.9354838709678,
    cy=1450.1612903225807,
    k1=0.61060095,
    k2=0.17060375,
    k3=-2.69401383,
    p1=-0.00210447,
    p2=-0.00060749,
    xi=1.94817,
  )
  width: int = 2880
  """The width the intrinsics were measured at."""
  pos: tuple[float, float, float] = (-0.014537817405, -0.000731615839, 0.036328428298)
  quat: tuple[float, float, float, float] = (
    -0.4290814654677777,
    0.5837380175668424,
    -0.5669007222884379,
    0.392125737370049,
  )
  """How the lens is turned: x right, y down, z forward."""
  mask: float | None = 0.5
  """How much of the frame it keeps, as a radius from the center over the
  width; ``None`` keeps the whole frame."""

  def rays(self, size: int) -> Rays:
    """Every pixel's bearing at ``size`` pixels across. A pixel earns a ray by
    surviving the round trip and lying inside the disc."""
    mei = self.mei.scaled(size / self.width)
    rows, columns = np.indices((size, size), dtype=np.float64)
    pixels = np.stack([columns.ravel(), rows.ravel()], axis=1) + 0.5
    bearings = unproject(pixels, mei)
    residual = np.linalg.norm(project(bearings, mei) - pixels, axis=1)
    valid = np.isfinite(residual) & (residual < ROUND_TRIP)
    if self.mask is not None:
      valid &= np.linalg.norm(pixels - size / 2.0, axis=1) <= self.mask * size
    return Rays(
      bearings=np.asarray(bearings[valid], np.float32),
      pixel=np.flatnonzero(valid).astype(np.int32),
    )


NOMINAL = Camera()
"""The lens every hand carries unless told otherwise."""
