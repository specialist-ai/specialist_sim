"""The room behind everything: a photograph of the capture room, as six faces.

The scene owns the surfaces a hand can reach -- the table, the tape, the cube --
and a photograph owns the rest of the room. MuJoCo loads that photograph as a
skybox and stores it as six square faces; this module says which way each face
looks, and hands the faces to the tracer.

The face table is MuJoCo's own ``gridlayout`` lettering, and a compiled model
returns the faces byte for byte in that lettering's order, which is what
:data:`ORDER` restates. Nothing here rotates the photograph: ``scripts/skybox.py``
turns it into the room frame when it builds the asset.
"""

from __future__ import annotations

import mujoco
import numpy as np

from specialist_sim.arrays import Float32Array

__all__ = ["AXES", "FACES", "LAYOUT", "ORDER", "read"]

FACES = {
  "R": ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
  "L": ((-1, 0, 0), (0, -1, 0), (0, 0, -1)),
  "U": ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
  "D": ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
  "F": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
  "B": ((0, 1, 0), (-1, 0, 0), (0, 0, -1)),
}
"""Each face's centre, the way its columns run, and the way its rows run."""

ORDER = "RLUDFB"
"""The order a compiled model stores the faces in, whatever the sheet's layout."""

LAYOUT = (".U..", "LFRB", ".D..")
"""How the faces sit on the 3x4 sheet, as ``room.xml``'s ``gridlayout`` spells it."""

AXES: Float32Array = np.stack(
  [
    np.stack([np.array(axis, np.float32) for axis in FACES[key]], axis=1)
    for key in ORDER
  ]
)
"""The face table as columns of a matrix, ``(6, 3, 3)``: a ray times the
transpose of one of these is its depth into the face, then its column and row."""


def read(model: mujoco.MjModel) -> Float32Array:
  """The scene's skybox as ``(6, size, size, 3)`` in 0 to 1, in :data:`ORDER`.

  Raises:
    ValueError: if the scene carries anything but one three-channel skybox.
  """
  found = [
    index
    for index in range(model.ntex)
    if model.tex_type[index] == mujoco.mjtTexture.mjTEXTURE_SKYBOX
  ]
  if len(found) != 1:
    raise ValueError(
      f"the scene carries {len(found)} skyboxes, and the tracer wants one"
    )
  index = found[0]
  size, channels = int(model.tex_width[index]), int(model.tex_nchannel[index])
  if channels != 3:
    raise ValueError(f"the skybox has {channels} channels, and the tracer shades three")
  if int(model.tex_height[index]) != len(ORDER) * size:
    raise ValueError("the skybox is not six square faces, so its faces cannot be cut")
  start = int(model.tex_adr[index])
  faces = model.tex_data[start : start + len(ORDER) * size * size * channels]
  out: Float32Array = faces.reshape(len(ORDER), size, size, channels).astype(np.float32)
  return out / 255.0
