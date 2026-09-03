"""The drawable scene, one rigid part per drawn geom."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from specialist_sim.arrays import Float32Array, Int32Array

__all__ = ["Part", "colours", "gather"]

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
  body: int
  geom: int


OPAQUE = 0.999
"""The least alpha this renderer draws: it traces no translucency."""
INVISIBLE = 0.05
"""The most alpha a geom may carry and still be left out as invisible."""

DECOR = 1024
"""Room in the visualiser's scene for the geoms a model does not own."""


def colours(model: mujoco.MjModel) -> Float32Array:
  """Every geom's colour as the visualiser resolves it, ``(ngeom, 4)``.

  A geom is coloured by its material unless it carries an ``rgba`` of its own,
  and a compiled model does not record which of the two happened. The
  visualiser decides it, so the scene it builds is what the tracer shades from
  and the rule lives in one place. A geom the visualiser leaves out comes back
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
  rgba = colours(model)
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
    parts.append(
      Part(
        vertices=local.astype(np.float32),
        faces=face.reshape(-1, 3),
        albedo=rgba[geom][:3],
        body=int(model.geom_bodyid[geom]),
        geom=geom,
      )
    )
  return parts
