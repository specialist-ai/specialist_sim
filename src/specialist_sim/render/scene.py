"""The drawable scene, one rigid part per drawn geom."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from specialist_sim.arrays import Float32Array, Int32Array

__all__ = ["Part", "colors", "gather"]

DRAWN_GROUPS = (1, 2)

_CORNERS = np.array(
  [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64
)
_BOX_FACES = np.array(
  [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
   2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3], dtype=np.int32
)  # fmt: skip


@dataclass(frozen=True)
class Part:
  """One drawn geom's triangles, in the geom's own frame."""

  vertices: Float32Array
  faces: Int32Array
  albedo: Float32Array
  texture: Float32Array | None
  repeat: Float32Array
  size: Float32Array
  body: int
  geom: int


OPAQUE = 0.999
"""The least alpha this renderer draws: it traces no translucency."""
INVISIBLE = 0.05
"""The most alpha a geom may carry and still be left out as invisible."""

DECOR = 1024
"""Room in the visualizer's scene for the geoms a model does not own."""


def colors(model: mujoco.MjModel) -> Float32Array:
  """Every geom's color as the visualizer resolves it, ``(ngeom, 4)``.

  A geom is colored by its material unless it carries an ``rgba`` of its own,
  and a compiled model does not record which of the two happened. The
  visualizer decides it, so the scene it builds is what the tracer shades from
  and the rule lives in one place. A geom the visualizer leaves out comes back
  transparent, which is how it is drawn.
  """
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  option = mujoco.MjvOption()
  option.geomgroup[:] = 1
  scene = mujoco.MjvScene(model, maxgeom=model.ngeom + DECOR)
  mujoco.mjv_updateScene(
    model,
    data,
    option,
    mujoco.MjvPerturb(),
    mujoco.MjvCamera(),
    mujoco.mjtCatBit.mjCAT_ALL,
    scene,
  )
  out: Float32Array = np.zeros((model.ngeom, 4), np.float32)
  for index in range(scene.ngeom):
    drawn = scene.geoms[index]
    if drawn.objtype == mujoco.mjtObj.mjOBJ_GEOM:
      out[drawn.objid] = drawn.rgba
  return out


def gather(model: mujoco.MjModel) -> list[Part]:
  rgba = colors(model)
  parts = []
  for geom in range(model.ngeom):
    if model.geom_group[geom] not in DRAWN_GROUPS:
      continue
    if rgba[geom][3] < OPAQUE:
      if rgba[geom][3] > INVISIBLE:
        raise ValueError(f"geom {geom} is translucent and this renderer is opaque")
      continue
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
      local = _CORNERS * model.geom_size[geom]
      face = _BOX_FACES
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
      mesh = model.geom_dataid[geom]
      start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
      local = model.mesh_vert[start : start + count].astype(np.float64)
      at, num = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
      face = model.mesh_face[at : at + num].ravel().astype(np.int32)
    else:
      raise ValueError(
        f"geom {geom} is a {mujoco.mjtGeom(kind).name}, which this renderer cannot draw"
      )
    material = int(model.geom_matid[geom])
    texture = -1
    repeat = np.ones(2, np.float32)
    if material >= 0:
      texture = int(model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB])
      repeat = model.mat_texrepeat[material].astype(np.float32)
    pixels = None
    if texture >= 0:
      if kind != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError(f"textured geom {geom} is not a box")
      if model.tex_type[texture] != mujoco.mjtTexture.mjTEXTURE_2D:
        raise ValueError(f"geom {geom} carries a texture that is not two-dimensional")
      channels = int(model.tex_nchannel[texture])
      if channels != 3:
        raise ValueError(f"geom {geom} carries a {channels}-channel texture")
      width, height = int(model.tex_width[texture]), int(model.tex_height[texture])
      start = int(model.tex_adr[texture])
      stop = start + width * height * channels
      pixels = (
        model.tex_data[start:stop]
        .reshape(height, width, channels)
        .astype(np.float32)
        / 255.0
      )
    parts.append(
      Part(
        vertices=local.astype(np.float32),
        faces=face.reshape(-1, 3),
        albedo=rgba[geom][:3],
        texture=pixels,
        repeat=repeat,
        size=model.geom_size[geom].astype(np.float32),
        body=int(model.geom_bodyid[geom]),
        geom=geom,
      )
    )
  return parts
