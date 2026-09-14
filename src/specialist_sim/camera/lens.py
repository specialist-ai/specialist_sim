"""The MEI fisheye a hand's camera projects through, and its seat on the base."""

from dataclasses import dataclass, replace

import numpy as np

from specialist_sim.arrays import Float32Array, Float64Array, Int32Array


@dataclass(frozen=True)
class Mei:
  """MEI intrinsics; ``fx``, ``fy``, ``cx`` and ``cy`` in image widths."""

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
    return replace(
      self,
      fx=self.fx * factor,
      fy=self.fy * factor,
      cx=self.cx * factor,
      cy=self.cy * factor,
    )


def _distort(
  x: Float64Array, y: Float64Array, mei: Mei
) -> tuple[Float64Array, Float64Array, Float64Array]:
  """The radial factor and tangential x and y shifts at normalized-plane points."""
  r2 = x * x + y * y
  radial = 1.0 + mei.k1 * r2 + mei.k2 * r2 * r2 + mei.k3 * r2**3
  tangential_x = 2.0 * mei.p1 * x * y + mei.p2 * (r2 + 2.0 * x * x)
  tangential_y = mei.p1 * (r2 + 2.0 * y * y) + 2.0 * mei.p2 * x * y
  return radial, tangential_x, tangential_y


def project(points: Float64Array, mei: Mei) -> Float64Array:
  """Each point's pixel, ``(n, 3)`` to ``(n, 2)``; NaN where the map is not one
  to one."""
  sphere = points / np.linalg.norm(points, axis=1, keepdims=True)
  denom = sphere[:, 2] + mei.xi
  valid = (denom > 0.0) & (1.0 + mei.xi * sphere[:, 2] > 0.0)
  x = np.where(valid, sphere[:, 0] / denom, np.nan)
  y = np.where(valid, sphere[:, 1] / denom, np.nan)
  radial, tangential_x, tangential_y = _distort(x, y, mei)
  u = mei.fx * (x * radial + tangential_x) + mei.cx
  v = mei.fy * (y * radial + tangential_y) + mei.cy
  return np.stack([u, v], 1)


def unproject(pixels: Float64Array, mei: Mei) -> Float64Array:
  """Each pixel's unit bearing, ``(n, 2)`` to ``(n, 3)``, x right, y down, z forward."""
  target_x = (pixels[:, 0] - mei.cx) / mei.fx
  target_y = (pixels[:, 1] - mei.cy) / mei.fy
  x = target_x
  y = target_y
  for _ in range(20):
    radial, tangential_x, tangential_y = _distort(x, y, mei)
    safe = np.where(radial == 0.0, 1.0, radial)
    next_x = (target_x - tangential_x) / safe
    next_y = (target_y - tangential_y) / safe
    step = max(np.max(np.abs(next_x - x)), np.max(np.abs(next_y - y)))
    x = next_x
    y = next_y
    if step < 1e-12:
      break
  r2 = x * x + y * y
  discriminant = 1.0 + (1.0 - mei.xi * mei.xi) * r2
  root = np.sqrt(np.where(discriminant < 0.0, np.nan, discriminant))
  factor = (mei.xi + root) / (r2 + 1.0)
  return np.stack([factor * x, factor * y, factor - mei.xi], axis=1)


@dataclass(frozen=True)
class Rays:
  """The rays a lens draws, at one output size."""

  bearings: Float32Array
  """``(n, 3)`` unit bearings in the lens frame."""
  pixel: Int32Array
  """The index in the flattened image of the pixel each bearing falls in."""


@dataclass(frozen=True)
class Lens:
  """A lens, and its seat in the frame of the body carrying it."""

  params: Mei = Mei(
    fx=2250.0 / 2880.0,
    fy=2250.0 / 2880.0,
    cx=1456.9354838709678 / 2880.0,
    cy=1450.1612903225807 / 2880.0,
    k1=0.61060095,
    k2=0.17060375,
    k3=-2.69401383,
    p1=-0.00210447,
    p2=-0.00060749,
    xi=1.94817,
  )
  pos: tuple[float, float, float] = (-0.014537817405, -0.000731615839, 0.036328428298)
  quat: tuple[float, float, float, float] = (
    -0.4290814654677777,
    0.5837380175668424,
    -0.5669007222884379,
    0.392125737370049,
  )
  """``wxyz``; the lens frame is x right, y down, z forward."""
  mask_radius: float | None = 0.5
  """The kept disc's radius over the width; ``None`` keeps the whole frame."""

  def rays(self, size: int, supersample: int = 1) -> Rays:
    """The ``supersample`` by ``supersample`` rays of each pixel, ``size`` across,
    that survive the round trip and whose pixel lies inside the disc."""
    fine = size * supersample
    mei = self.params.scaled(fine)
    rows, columns = np.indices((fine, fine), dtype=np.float64)
    points = np.stack([columns.ravel(), rows.ravel()], axis=1) + 0.5
    bearings = unproject(points, mei)
    residual = np.linalg.norm(project(bearings, mei) - points, axis=1)
    valid = np.isfinite(residual) & (residual < 1e-3)
    pixels = points // supersample
    if self.mask_radius is not None:
      valid &= (
        np.linalg.norm(pixels + 0.5 - size / 2.0, axis=1) <= self.mask_radius * size
      )
    return Rays(
      bearings=np.asarray(bearings[valid], np.float32),
      pixel=(pixels[valid] @ [1, size]).astype(np.int32),
    )
