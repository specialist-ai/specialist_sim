"""One ray per pixel of a fisheye, and the disc they fall inside.

A pixel earns a ray by surviving the round trip and by lying inside the disc an
observation keeps: outside its domain the model's inversion converges on the
optical axis, and outside the disc the policy has never seen a pixel lit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from specialist_sim.arrays import BoolArray, Float32Array, Int32Array
from specialist_sim.mei import MeiParams, mei_project, mei_unproject

__all__ = ["Rays", "build_rays"]

PIXEL_CENTRE = 0.5
ROUND_TRIP = 1e-3
"""How far a pixel may move through the round trip and still be a bearing."""


@dataclass(frozen=True)
class Rays:
  """A lens at one output size."""

  bearings: Float32Array
  """``(n, 3)`` unit bearings in the lens frame: x right, y down, z forward."""
  pixel: Int32Array
  """Where each bearing's ray lands in the flattened image."""
  valid: BoolArray
  """Which pixels of the image a bearing exists for."""


def build_rays(
  params: MeiParams, width: int, height: int, radius: float | None
) -> Rays:
  """Every pixel's bearing. ``radius`` blanks the frame outside that fraction
  of its width, or keeps all of it when there is none."""
  rows, columns = np.indices((height, width), dtype=np.float64)
  pixels = np.stack([columns.ravel(), rows.ravel()], axis=1) + PIXEL_CENTRE
  bearings = mei_unproject(pixels, params)
  residual = np.linalg.norm(mei_project(bearings, params) - pixels, axis=1)
  valid = np.isfinite(residual) & (residual < ROUND_TRIP)
  if radius is not None:
    centre = np.array([width, height], np.float64) / 2.0
    valid &= np.linalg.norm(pixels - centre, axis=1) <= radius * width
  return Rays(
    bearings=np.asarray(bearings[valid], np.float32),
    pixel=np.flatnonzero(valid).astype(np.int32),
    valid=valid.reshape(height, width),
  )
