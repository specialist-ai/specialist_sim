"""Rendering the scene through each gripper's fisheye.

Every drawn geom is rigid, so each one's triangles live in a hierarchy built
once in its own frame; a frame's poses carry the rays instead of the geometry,
and nothing is rebuilt or refit as the scene moves. The cameras share the parts
and differ only in which they may see. The floor, walls, table and fiducials are
geometry; only rays that leave that measured patch read a neutral background.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from specialist_sim.arrays import Int32Array, UInt8Array
from specialist_sim.calibration import CameraCfg
from specialist_sim.rays import build_rays
from specialist_sim.render import kernels, sky
from specialist_sim.render.scene import gather

__all__ = ["HOUSING", "Renderer", "View", "seat"]

HOUSING = ("x3_", "protector")
"""The described parts a housed lens looks out of, and so cannot see."""


@dataclass(frozen=True)
class View:
  """One camera's frame, and which body each pixel struck."""

  image: UInt8Array
  body: Int32Array


def seat(model: mujoco.MjModel, name: str) -> int:
  """The body a camera of this name rides."""
  for body in (f"{name}_lens", name):
    found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if found >= 0:
      return int(found)
  raise ValueError(f"the scene carries no body for the camera {name}")


class _Camera:
  """One lens: where it rides, what it may see, and where its pixels land.

  Pixels outside the lens disc are written once here — black, no body — and
  the trace never touches them again.
  """

  def __init__(
    self, body: int, hidden: set[int], size: int, parts: int, camera: CameraCfg
  ):
    self.body = body
    rays = build_rays(camera.at(size), size, size, camera.mask_radius)
    self.size = size
    self.bearings = wp.array(rays.bearings, dtype=wp.vec3)
    self.pixel = wp.array(rays.pixel, dtype=wp.int32)
    seen = np.ones(parts, np.int32)
    seen[sorted(hidden)] = 0
    self.seen = wp.array(seen, dtype=wp.int32)
    self.image = wp.zeros(size * size * 3, dtype=wp.uint8)
    self.body_hit = wp.array(np.full(size * size, -1, np.int32), dtype=wp.int32)


class Renderer:
  """Every camera's view of one scene, re-traced each time the scene moves."""

  def __init__(self, model: mujoco.MjModel, cameras: dict[str, CameraCfg], size: int):
    wp.init()
    self.parts = gather(model)
    self._meshes = [
      wp.Mesh(
        points=wp.array(part.vertices, dtype=wp.vec3),
        indices=wp.array(np.ascontiguousarray(part.faces.ravel()), dtype=wp.int32),
      )
      for part in self.parts
    ]
    self.meshes = wp.array([mesh.id for mesh in self._meshes], dtype=wp.uint64)
    self.geoms = np.array([part.geom for part in self.parts], np.int32)
    self.position = wp.zeros(len(self.parts), dtype=wp.vec3)
    self.rotation = wp.zeros(len(self.parts), dtype=wp.mat33)
    self.albedo = wp.array(
      np.stack([part.albedo for part in self.parts]), dtype=wp.vec3
    )
    texture_pixels = []
    texture_address = []
    texture_width = []
    texture_height = []
    for part in self.parts:
      if part.texture is None:
        texture_address.append(-1)
        texture_width.append(0)
        texture_height.append(0)
        continue
      texture_address.append(sum(len(pixels) for pixels in texture_pixels))
      texture_height.append(part.texture.shape[0])
      texture_width.append(part.texture.shape[1])
      texture_pixels.append(part.texture.reshape(-1, 3))
    packed = (
      np.concatenate(texture_pixels, axis=0)
      if texture_pixels
      else np.zeros((1, 3), np.float32)
    )
    self.texture = wp.array(packed, dtype=wp.vec3)
    self.texture_address = wp.array(np.asarray(texture_address), dtype=wp.int32)
    self.texture_width = wp.array(np.asarray(texture_width), dtype=wp.int32)
    self.texture_height = wp.array(np.asarray(texture_height), dtype=wp.int32)
    self.texture_repeat = wp.array(
      np.stack([part.repeat for part in self.parts]), dtype=wp.vec2
    )
    self.part_size = wp.array(
      np.stack([part.size for part in self.parts]), dtype=wp.vec3
    )
    self.part_body = wp.array(
      np.array([part.body for part in self.parts], np.int32), dtype=wp.int32
    )
    faces = sky.read(model)
    self.sky_size = int(faces.shape[1])
    self.sky = wp.array(faces.reshape(-1, 3), dtype=wp.vec3)
    self.sky_axes = wp.array(sky.AXES, dtype=wp.mat33)
    self.cameras = {
      name: _Camera(
        seat(model, name),
        self._housing(model, name, camera),
        size,
        len(self.parts),
        camera,
      )
      for name, camera in cameras.items()
    }
    self._key, self._fill, self._ambient = self._lights(model)

  def render(self, data: mujoco.MjData) -> dict[str, View]:
    self.position.assign(np.ascontiguousarray(data.geom_xpos[self.geoms], np.float32))
    self.rotation.assign(
      np.ascontiguousarray(data.geom_xmat[self.geoms].reshape(-1, 3, 3), np.float32)
    )
    out = {}
    for slot, camera in self.cameras.items():
      wp.launch(
        kernels.trace,
        dim=len(camera.pixel),
        inputs=[
          self.meshes,
          self.position,
          self.rotation,
          camera.seen,
          wp.vec3(*data.xpos[camera.body]),
          wp.mat33(*data.xmat[camera.body]),
          camera.bearings,
          camera.pixel,
          self.albedo,
          self.texture,
          self.texture_address,
          self.texture_width,
          self.texture_height,
          self.texture_repeat,
          self.part_size,
          self.part_body,
          *self._key,
          *self._fill,
          self._ambient,
          self.sky,
          self.sky_axes,
          self.sky_size,
          camera.image,
          camera.body_hit,
        ],
      )
      out[slot] = View(
        image=camera.image.numpy().reshape(camera.size, camera.size, 3).copy(),
        body=camera.body_hit.numpy().reshape(camera.size, camera.size).copy(),
      )
    return out

  def _housing(self, model: mujoco.MjModel, name: str, camera: CameraCfg) -> set[int]:
    """What a lens sits inside, and so cannot see."""
    if not camera.housed:
      return set()
    housing = tuple(f"{name}_{part}" for part in HOUSING)
    hidden = {
      index
      for index, part in enumerate(self.parts)
      if (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, part.geom) or ""
      ).startswith(housing)
    }
    if not hidden:
      raise ValueError(f"{name} is housed, and the scene draws none of {housing}")
    return hidden

  def _lights(
    self, model: mujoco.MjModel
  ) -> tuple[tuple[wp.vec3, wp.vec3], tuple[wp.vec3, wp.vec3], wp.vec3]:
    still = [i for i in range(model.nlight) if model.light_bodyid[i] != 0]
    if still:
      raise ValueError(
        "lights are read once at construction, so they must ride the world body"
      )
    casting = [i for i in range(model.nlight) if model.light_castshadow[i]]
    flat = [i for i in range(model.nlight) if not model.light_castshadow[i]]
    if len(casting) != 1 or len(flat) != 1:
      raise ValueError("the scene wants one shadowing light and one fill")
    key, fill = casting[0], flat[0]
    return (
      (wp.vec3(*model.light_dir[key]), wp.vec3(*model.light_diffuse[key])),
      (wp.vec3(*model.light_dir[fill]), wp.vec3(*model.light_diffuse[fill])),
      wp.vec3(*model.light_ambient[fill]),
    )
