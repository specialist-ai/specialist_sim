"""Each camera's view of the scene, traced through its lens on the GPU."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from specialist_sim import kernels, rotation
from specialist_sim.arrays import Float32Array, Int32Array, UInt8Array
from specialist_sim.camera import Camera

__all__ = ["HOUSING", "Part", "Renderer", "View", "parts"]

DRAWN_GROUPS = (1, 2)
HOUSING = ("x3_", "protector")
"""The described parts a lens looks out of, and so cannot see."""

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
  """A box's diffuse map, ``(height, width, 3)`` in 0 to 1, tinted by the albedo."""
  repeat: Float32Array
  """How many times the map tiles across and up a face."""
  size: Float32Array
  """A box's half sizes."""
  body: int
  geom: int


@dataclass(frozen=True)
class View:
  """One camera's frame, and which body each pixel struck, ``-1`` for none."""

  image: UInt8Array
  body: Int32Array


def _colors(model: mujoco.MjModel) -> Float32Array:
  """Every geom's color as the visualizer resolves it, ``(ngeom, 4)``: its
  material's unless it carries an ``rgba`` of its own."""
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  option = mujoco.MjvOption()
  option.geomgroup[:] = 1
  scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 1024)
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


def _texture(
  model: mujoco.MjModel, geom: int
) -> tuple[Float32Array | None, Float32Array]:
  """A geom's material's diffuse map and repeat, if it carries one."""
  material = int(model.geom_matid[geom])
  if material < 0:
    return None, np.ones(2, np.float32)
  texture = int(model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB])
  repeat = np.asarray(model.mat_texrepeat[material], np.float32)
  if texture < 0:
    return None, repeat
  width, height = int(model.tex_width[texture]), int(model.tex_height[texture])
  start = int(model.tex_adr[texture])
  pixels = model.tex_data[start : start + width * height * 3]
  return np.asarray(pixels, np.float32).reshape(height, width, 3) / 255.0, repeat


def parts(model: mujoco.MjModel) -> list[Part]:
  """Every opaque box and mesh in a drawn group."""
  rgba = _colors(model)
  out = []
  for geom in range(model.ngeom):
    if model.geom_group[geom] not in DRAWN_GROUPS or rgba[geom, 3] < 0.5:
      continue
    texture, repeat = _texture(model, geom)
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
      vertices, faces = _CORNERS * model.geom_size[geom], _BOX_FACES
    else:
      mesh = model.geom_dataid[geom]
      start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
      vertices = model.mesh_vert[start : start + count]
      at, num = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
      faces = model.mesh_face[at : at + num].ravel()
    out.append(
      Part(
        vertices=np.asarray(vertices, np.float32),
        faces=np.asarray(faces, np.int32).reshape(-1, 3),
        albedo=rgba[geom, :3],
        texture=texture,
        repeat=repeat,
        size=np.asarray(model.geom_size[geom], np.float32),
        body=int(model.geom_bodyid[geom]),
        geom=geom,
      )
    )
  return out


def _atlas(parts: Sequence[Part]) -> list[wp.array]:
  """Every texture in one run of texels, with each part's start, width,
  height, repeat and size. A part without one starts at ``-1``."""
  maps = [part.texture for part in parts if part.texture is not None]
  starts = np.cumsum([0] + [map.shape[0] * map.shape[1] for map in maps])
  address, at = [], 0
  for part in parts:
    address.append(int(starts[at]) if part.texture is not None else -1)
    at += part.texture is not None
  texels = (
    np.concatenate([map.reshape(-1, 3) for map in maps])
    if maps
    else np.zeros((1, 3), np.float32)
  )

  def sides(axis: int) -> wp.array:
    counts = [p.texture.shape[axis] if p.texture is not None else 0 for p in parts]
    return wp.array(np.array(counts, np.int32), dtype=wp.int32)

  return [
    wp.array(texels, dtype=wp.vec3),
    wp.array(np.array(address, np.int32), dtype=wp.int32),
    sides(1),
    sides(0),
    wp.array(np.stack([part.repeat for part in parts]), dtype=wp.vec2),
    wp.array(np.stack([part.size for part in parts]), dtype=wp.vec3),
  ]


def _light(found: list[int], table: np.ndarray) -> wp.vec3:
  """One light's entry in ``table``, or black where the world has no such light."""
  return wp.vec3(*table[found[0]]) if found else wp.vec3(0.0, 0.0, 0.0)


def _background(model: mujoco.MjModel) -> wp.vec3:
  """What a ray that strikes nothing reads: the skybox's first texel."""
  sky = [
    i
    for i in range(model.ntex)
    if model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX
  ]
  if not sky:
    return wp.vec3(0.0, 0.0, 0.0)
  at = int(model.tex_adr[sky[0]])
  return wp.vec3(*(model.tex_data[at : at + 3] / 255.0))


class _Lens:
  """One camera: the body it rides, its seat there, what it may see, and where
  its pixels land."""

  def __init__(
    self, body: int, camera: Camera, hidden: list[int], size: int, count: int
  ):
    self.body, self.size = body, size
    self.pos = np.asarray(camera.pos)
    self.rotation = rotation.to_matrix(np.asarray(camera.quat))
    rays = camera.rays(size)
    self.bearings = wp.array(rays.bearings, dtype=wp.vec3)
    self.pixel = wp.array(rays.pixel, dtype=wp.int32)
    seen = np.ones(count, np.int32)
    seen[hidden] = 0
    self.seen = wp.array(seen, dtype=wp.int32)


class Renderer:
  """Every camera's view of a scene, from any data of its model. ``cameras``
  names each view: the body it rides and the camera seated there; a camera
  named for a slot cannot see that hand's housing."""

  def __init__(
    self, model: mujoco.MjModel, cameras: dict[str, tuple[int, Camera]], size: int
  ):
    wp.init()
    self.parts = parts(model)
    self._meshes = [
      wp.Mesh(
        points=wp.array(part.vertices, dtype=wp.vec3),
        indices=wp.array(np.ascontiguousarray(part.faces.ravel()), dtype=wp.int32),
      )
      for part in self.parts
    ]
    self.meshes = wp.array([mesh.id for mesh in self._meshes], dtype=wp.uint64)
    self.geoms = np.array([part.geom for part in self.parts], np.int32)
    self.size = size
    self.albedo = wp.array(
      np.stack([part.albedo for part in self.parts]), dtype=wp.vec3
    )
    self.body = wp.array(
      np.array([part.body for part in self.parts], np.int32), dtype=wp.int32
    )
    self.maps = _atlas(self.parts)
    names = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, part.geom) or ""
      for part in self.parts
    ]
    self.lenses = {
      name: _Lens(
        body,
        camera,
        [
          i
          for i, geom in enumerate(names)
          if geom.startswith(tuple(f"{name}_{housing}" for housing in HOUSING))
        ],
        size,
        len(self.parts),
      )
      for name, (body, camera) in cameras.items()
    }
    key = [i for i in range(model.nlight) if model.light_castshadow[i]]
    fill = [i for i in range(model.nlight) if not model.light_castshadow[i]]
    self._lights = [
      _light(key, model.light_dir),
      _light(key, model.light_diffuse),
      _light(fill, model.light_dir),
      _light(fill, model.light_diffuse),
      _light(fill, model.light_ambient),
      _background(model),
    ]
    self._position = wp.zeros((0, len(self.parts)), dtype=wp.vec3)
    self._rotation = wp.zeros((0, len(self.parts)), dtype=wp.mat33)
    self._image = wp.zeros((0, size * size * 3), dtype=wp.uint8)
    self._struck = wp.zeros((0, size * size), dtype=wp.int32)
    """Room for the scenes and frames of the largest batch drawn so far."""

  def render(
    self, data: mujoco.MjData, names: Sequence[str] | None = None
  ) -> dict[str, View]:
    """The named views, or all of them, of one data."""
    wanted = list(self.lenses) if names is None else list(names)
    images, bodies = self.render_many([data], wanted)
    return {
      name: View(image=image, body=body)
      for name, image, body in zip(wanted, images[0], bodies[0], strict=True)
    }

  def render_many(
    self, datas: Sequence[mujoco.MjData], names: Sequence[str]
  ) -> tuple[UInt8Array, Int32Array]:
    """The named views of every data at once: the images, ``(datas, names,
    size, size, 3)``, and the body each pixel struck, ``(datas, names, size,
    size)``, ``-1`` for none. Every data's scene goes up in one copy and every
    frame comes down in one, each frame as one data alone draws it."""
    scenes, frames, pixels = len(datas), len(datas) * len(names), self.size * self.size
    if scenes > self._position.shape[0]:
      self._position = wp.zeros((scenes, len(self.parts)), dtype=wp.vec3)
      self._rotation = wp.zeros((scenes, len(self.parts)), dtype=wp.mat33)
    if frames > self._image.shape[0]:
      self._image = wp.zeros((frames, pixels * 3), dtype=wp.uint8)
      self._struck = wp.zeros((frames, pixels), dtype=wp.int32)
    # A lens draws only the pixels inside its disc, so every frame starts black
    # and struck by nothing.
    self._image.zero_()
    self._struck.fill_(-1)
    self._position[:scenes].assign(
      np.stack([data.geom_xpos[self.geoms] for data in datas]).astype(np.float32)
    )
    self._rotation[:scenes].assign(
      np.stack([data.geom_xmat[self.geoms].reshape(-1, 3, 3) for data in datas]).astype(
        np.float32
      )
    )
    for scene, data in enumerate(datas):
      for index, name in enumerate(names):
        lens = self.lenses[name]
        mount = data.xmat[lens.body].reshape(3, 3)
        origin = data.xpos[lens.body] + mount @ lens.pos
        frame = scene * len(names) + index
        wp.launch(
          kernels.trace,
          dim=len(lens.pixel),
          inputs=[
            self.meshes,
            self._position[scene],
            self._rotation[scene],
            lens.seen,
            wp.vec3(*origin),
            wp.mat33(*(mount @ lens.rotation).ravel()),
            lens.bearings,
            lens.pixel,
            self.albedo,
            *self.maps,
            self.body,
            *self._lights,
            self._image[frame],
            self._struck[frame],
          ],
        )
    size = self.size
    images: UInt8Array = (
      self._image[:frames].numpy().reshape(scenes, len(names), size, size, 3)
    )
    bodies: Int32Array = (
      self._struck[:frames].numpy().reshape(scenes, len(names), size, size)
    )
    return images, bodies
