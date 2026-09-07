"""Turn a 360 photograph of the capture room into the room's skybox.

The panorama is equirectangular and in its own frame; the asset is the six-face
cross MuJoCo loads, in the room frame, so the tracer can sample it along a ray
without knowing anything about where the photograph was taken.

Two things are read out of the photograph itself to get there.

**Yaw.** Both anchor tags are in the frame, and the room is defined by them: the
midpoint is the origin and the line between them is across, so their bearings
fix how the panorama is turned. Reading each tag's centre twice -- by eye off a
pixel ruler, and as the crossing of the diagonals of its black square in a
gnomonic reprojection -- gives 48.1 and 50.6 degrees, and :data:`YAW` splits
them. The same fit puts the photographer at (0.46, 0.00, -0.14) in the room,
0.19 m above the table and 0.70 m from each anchor, which the tape's angular
width independently agrees with: 6.2 degrees across at that range is 28.5 mm,
against the 31 mm the tape geom is.

**Nothing else.** A skybox has no distance, so the room is drawn as if every
surface were infinitely far. It is a fair trade here only because the
photograph was taken inside the workspace: a camera moving 0.2 m about the
capture point sees the far wall shift by under 8 degrees, and the near surface
that would shift more -- the table -- the scene owns as a geom and draws itself.

Usage (needs pillow, which is in the dev group)::

    uv run --group dev python scripts/skybox.py room_equirect.png \
        src/specialist_sim/assets/room_skybox.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from specialist_sim.arrays import Float64Array
from specialist_sim.render.sky import FACES, LAYOUT

Image.MAX_IMAGE_PIXELS = None

YAW = 49.0
"""How far the panorama's own frame is turned from the room's, about up, in
degrees. Baking it into the asset leaves the tracer nothing to correct for."""

FACE_SIZE = 512
"""Texels across one face: 0.18 degrees each, which is finer than the 0.83
degrees a pixel of a 240-wide gripper camera covers."""


def sample(panorama: Float64Array, ray: Float64Array) -> Float64Array:
  """The panorama read along unit directions ``(..., 3)``, bilinearly.

  Longitude wraps and latitude clamps, which is what the poles want.
  """
  height, width = panorama.shape[:2]
  longitude = np.arctan2(-ray[..., 1], ray[..., 0])
  latitude = np.arcsin(np.clip(ray[..., 2], -1.0, 1.0))
  x = (longitude + np.pi) / (2.0 * np.pi) * width - 0.5
  y = (np.pi / 2.0 - latitude) / np.pi * height - 0.5
  left, top = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
  along, down = (x - left)[..., None], (y - top)[..., None]
  right, bottom = (left + 1) % width, np.clip(top + 1, 0, height - 1)
  left, top = left % width, np.clip(top, 0, height - 1)
  upper = panorama[top, left] * (1.0 - along) + panorama[top, right] * along
  lower = panorama[bottom, left] * (1.0 - along) + panorama[bottom, right] * along
  out: Float64Array = upper * (1.0 - down) + lower * down
  return out


def face(panorama: Float64Array, key: str, size: int, yaw: float) -> Float64Array:
  """One face of the cross, ``(size, size, 3)``, in the room frame."""
  centre, across, down = (np.array(axis, np.float64) for axis in FACES[key])
  grid = 2.0 * (np.arange(size) + 0.5) / size - 1.0
  ray = centre + grid[None, :, None] * across + grid[:, None, None] * down
  ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
  angle = np.radians(-yaw)
  turned = np.stack(
    [
      np.cos(angle) * ray[..., 0] - np.sin(angle) * ray[..., 1],
      np.sin(angle) * ray[..., 0] + np.cos(angle) * ray[..., 1],
      ray[..., 2],
    ],
    axis=-1,
  )
  return sample(panorama, turned)


def cross(panorama: Float64Array, size: int, yaw: float) -> Float64Array:
  """The six faces laid out as the 3x4 sheet ``LAYOUT`` names."""
  sheet = np.zeros((3 * size, 4 * size, 3), np.float64)
  for row, spec in enumerate(LAYOUT):
    for column, key in enumerate(spec):
      if key != ".":
        sheet[row * size : (row + 1) * size, column * size : (column + 1) * size] = (
          face(panorama, key, size, yaw)
        )
  return sheet


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("panorama", type=Path, help="the equirectangular photograph")
  parser.add_argument("out", type=Path, help="where to write the cross")
  parser.add_argument("--size", type=int, default=FACE_SIZE, help="texels per face")
  parser.add_argument(
    "--yaw", type=float, default=YAW, help="degrees, panorama to room"
  )
  args = parser.parse_args()

  panorama = np.asarray(Image.open(args.panorama).convert("RGB"), np.float64)
  sheet = cross(panorama, args.size, args.yaw)
  Image.fromarray(np.clip(sheet, 0.0, 255.0).astype(np.uint8)).save(
    args.out, optimize=True
  )
  print(f"wrote {args.out}, {4 * args.size}x{3 * args.size}, yawed {args.yaw} degrees")


if __name__ == "__main__":
  main()
