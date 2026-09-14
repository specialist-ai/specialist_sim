"""Each lens's view of the scene, traced through its lens on the GPU."""

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from specialist_sim.arrays import Float32Array, Float64Array, Int32Array, UInt8Array
from specialist_sim.camera import kernels
from specialist_sim.camera.lens import Lens, Rays
from specialist_sim.gripper import DRAWN

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
  size: Float32Array
  """A box's half sizes."""
  body: int
  geom: int


def _colors(model: mujoco.MjModel) -> Float32Array:
  """Every geom's color, ``(ngeom, 4)``: its material's where its own ``rgba``
  is MuJoCo's default."""
  out = np.array(model.geom_rgba, np.float32)
  material = model.geom_matid
  inherits = (material >= 0) & (out == [0.5, 0.5, 0.5, 1.0]).all(axis=1)
  out[inherits] = model.mat_rgba[material[inherits]]
  return out


def parts(model: mujoco.MjModel) -> list[Part]:
  """Every opaque box and mesh in group ``DRAWN``."""
  rgba = _colors(model)
  out = []
  for geom in range(model.ngeom):
    if model.geom_group[geom] != DRAWN or rgba[geom, 3] < 0.5:
      continue
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
      vertices = _CORNERS * model.geom_size[geom]
      faces = _BOX_FACES
    else:
      mesh = model.geom_dataid[geom]
      start = model.mesh_vertadr[mesh]
      count = model.mesh_vertnum[mesh]
      vertices = model.mesh_vert[start : start + count]
      at = model.mesh_faceadr[mesh]
      num = model.mesh_facenum[mesh]
      faces = model.mesh_face[at : at + num].ravel()
    out.append(
      Part(
        vertices=np.asarray(vertices, np.float32),
        faces=np.asarray(faces, np.int32).reshape(-1, 3),
        albedo=rgba[geom, :3],
        size=np.asarray(model.geom_size[geom], np.float32),
        body=int(model.geom_bodyid[geom]),
        geom=geom,
      )
    )
  return out


class _View:
  """One lens's view: its site, the parts it sees, and its rays."""

  def __init__(self, site: int, hidden: list[int], rays: Rays, count: int):
    self.site = site
    self.bearings = wp.array(rays.bearings, dtype=wp.vec3)
    self.pixel = wp.array(rays.pixel, dtype=wp.int32)
    seen = np.ones(count, np.int32)
    seen[hidden] = 0
    self.seen = wp.array(seen, dtype=wp.int32)


class Renderer:
  """Every lens's view of a scene, each pixel the mean of its ``supersample`` by
  ``supersample`` rays. A view is a site, its lens, and the geoms it cannot see.
  The model holds a shadow-casting key light and a shadowless fill."""

  def __init__(
    self,
    model: mujoco.MjModel,
    views: Sequence[tuple[int, Lens, Sequence[int]]],
    size: int,
    supersample: int = 1,
  ):
    wp.init()
    self.parts = parts(model)
    self._kept = [
      wp.Mesh(
        points=wp.array(part.vertices, dtype=wp.vec3),
        indices=wp.array(np.ascontiguousarray(part.faces.ravel()), dtype=wp.int32),
      )
      for part in self.parts
    ]
    self._meshes = wp.array([mesh.id for mesh in self._kept], dtype=wp.uint64)
    self._geoms = np.array([part.geom for part in self.parts], np.int32)
    self._size = size
    self._supersample = supersample
    self._albedo = wp.array(
      np.stack([part.albedo for part in self.parts]), dtype=wp.vec3
    )
    self._body = wp.array(
      np.array([part.body for part in self.parts], np.int32), dtype=wp.int32
    )
    self.views = [
      _View(
        site,
        [index for index, part in enumerate(self.parts) if part.geom in hidden],
        lens.rays(size, supersample),
        len(self.parts),
      )
      for site, lens, hidden in views
    ]
    key = int(np.flatnonzero(model.light_castshadow)[0])
    fill = int(np.flatnonzero(model.light_castshadow == 0)[0])
    self._lights = [
      wp.vec3(*model.light_dir[key]),
      wp.vec3(*model.light_diffuse[key]),
      wp.vec3(*model.light_dir[fill]),
      wp.vec3(*model.light_diffuse[fill]),
      wp.vec3(*model.light_ambient[fill]),
    ]
    # Room for the scenes and frames of the largest batch drawn so far.
    self._position = wp.zeros((0, len(self.parts)), dtype=wp.vec3)
    self._rotation = wp.zeros((0, len(self.parts)), dtype=wp.mat33)
    self._image = wp.zeros((0, size * size * 3), dtype=wp.int32)
    self._struck = wp.zeros((0, size * size), dtype=wp.int32)

  def render(
    self,
    geom_xpos: Float64Array,
    geom_xmat: Float64Array,
    site_xpos: Float64Array,
    site_xmat: Float64Array,
    which: Sequence[int],
  ) -> tuple[UInt8Array, Int32Array]:
    """The views of the lenses ``which`` indexes in ``m`` scenes, each given by
    its geoms' and sites' poses, ``(m, ngeom, 3)``, ``(m, ngeom, 9)``, ``(m,
    nsite, 3)`` and ``(m, nsite, 9)``: the images, ``(m, len(which), size, size,
    3)``, and the highest-numbered body a ray of each pixel struck, ``(m,
    len(which), size, size)``, ``-1`` for none."""
    scenes = len(geom_xpos)
    frames = scenes * len(which)
    pixels = self._size**2
    if scenes > self._position.shape[0]:
      self._position = wp.zeros((scenes, len(self.parts)), dtype=wp.vec3)
      self._rotation = wp.zeros((scenes, len(self.parts)), dtype=wp.mat33)
    if frames > self._image.shape[0]:
      self._image = wp.zeros((frames, pixels * 3), dtype=wp.int32)
      self._struck = wp.zeros((frames, pixels), dtype=wp.int32)
    # A lens draws only the pixels inside its disc.
    self._image.zero_()
    self._struck.fill_(-1)
    self._position[:scenes].assign(np.asarray(geom_xpos[:, self._geoms], np.float32))
    self._rotation[:scenes].assign(
      np.asarray(geom_xmat[:, self._geoms], np.float32).reshape(scenes, -1, 3, 3)
    )
    for scene in range(scenes):
      for index, view in enumerate(self.views[at] for at in which):
        frame = scene * len(which) + index
        wp.launch(
          kernels.trace,
          dim=len(view.pixel),
          inputs=[
            self._meshes,
            self._position[scene],
            self._rotation[scene],
            view.seen,
            wp.vec3(*site_xpos[scene, view.site]),
            wp.mat33(*site_xmat[scene, view.site]),
            view.bearings,
            view.pixel,
            self._albedo,
            self._body,
            *self._lights,
            self._image[frame],
            self._struck[frame],
          ],
        )
    size, rays = self._size, self._supersample**2
    total = self._image[:frames].numpy().reshape(scenes, len(which), size, size, 3)
    return (
      ((total + rays // 2) // rays).astype(np.uint8),
      self._struck[:frames].numpy().reshape(scenes, len(which), size, size),
    )
