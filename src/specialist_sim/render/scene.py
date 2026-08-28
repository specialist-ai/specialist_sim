"""The drawable scene as flat arrays: triangles, their geom, and its colour."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from specialist_sim.arrays import Float32Array, Int32Array

__all__ = ["Geometry", "gather"]

DRAWN_GROUPS = (1, 2)

_CORNERS = np.array(
  [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64
)
_BOX_FACES = np.array(
  [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1,
   2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3], dtype=np.int32
)  # fmt: skip


@dataclass(frozen=True)
class Geometry:
  """Every drawn triangle, in the frame of the geom that carries it."""

  vertices: Float32Array
  vertex_geom: Int32Array
  faces: Int32Array
  face_geom: Int32Array
  face_body: Int32Array
  albedo: Float32Array
  geoms: Int32Array


def gather(model: mujoco.MjModel) -> Geometry:
  vertices: list[np.ndarray] = []
  vertex_geom: list[np.ndarray] = []
  faces: list[np.ndarray] = []
  face_geom: list[np.ndarray] = []
  albedo: list[np.ndarray] = []
  geoms: list[int] = []
  for geom in range(model.ngeom):
    if model.geom_group[geom] not in DRAWN_GROUPS:
      continue
    if model.geom_rgba[geom][3] < 0.999:
      if model.geom_rgba[geom][3] > 0.05:
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
        f"geom {geom} is a {mujoco.mjtGeom(kind).name}, which is not drawn"
      )
    index = len(geoms)
    faces.append(face + sum(len(v) for v in vertices))
    face_geom.append(np.full(len(face) // 3, index, np.int32))
    vertex_geom.append(np.full(len(local), index, np.int32))
    vertices.append(local)
    albedo.append(model.geom_rgba[geom][:3])
    geoms.append(geom)
  face_geom_flat = np.concatenate(face_geom)
  bodies = np.asarray([model.geom_bodyid[geom] for geom in geoms], np.int32)
  return Geometry(
    vertices=np.concatenate(vertices).astype(np.float32),
    vertex_geom=np.concatenate(vertex_geom),
    faces=np.concatenate(faces),
    face_geom=face_geom_flat,
    face_body=bodies[face_geom_flat],
    albedo=np.asarray(albedo, np.float32),
    geoms=np.asarray(geoms, np.int32),
  )
