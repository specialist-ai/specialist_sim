"""Rendering the scene through each gripper's fisheye.

Every drawn geom is rigid, so each one's triangles live in a hierarchy built
once in its own frame; a frame's poses carry the rays instead of the geometry,
and nothing is rebuilt or refit as the scene moves. The cameras share the parts
and differ only in which they may see.

Worlds share them too. Because a hierarchy stands in its geom's own frame rather
than the room's, several worlds are the same geometry under different poses: a
world costs an image and a pose per part, and every world is traced in one launch
per camera.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from specialist_sim.arrays import Int32Array, UInt8Array
from specialist_sim.calibration import CameraCfg
from specialist_sim.rays import build_rays
from specialist_sim.render import kernels
from specialist_sim.render.scene import gather

__all__ = ["Renderer", "View", "seat"]


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
    self,
    body: int,
    hidden: set[int],
    size: int,
    parts: int,
    camera: CameraCfg,
    worlds: int,
  ):
    self.body = body
    rays = build_rays(camera.at(size), size, size, camera.mask_radius)
    self.size = size
    self.bearings = wp.array(rays.bearings, dtype=wp.vec3)
    self.pixel = wp.array(rays.pixel, dtype=wp.int32)
    seen = np.ones(parts, np.int32)
    seen[sorted(hidden)] = 0
    self.seen = wp.array(seen, dtype=wp.int32)
    self.image = wp.zeros((worlds, size * size * 3), dtype=wp.uint8)
    self.body_hit = wp.array(
      np.full((worlds, size * size), -1, np.int32), dtype=wp.int32
    )


class Renderer:
  """Every camera's view of every world, re-traced each time the worlds move."""

  def __init__(
    self,
    model: mujoco.MjModel,
    cameras: dict[str, CameraCfg],
    size: int,
    worlds: int = 1,
  ):
    wp.init()
    self.worlds = worlds
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
    self.position = wp.zeros((worlds, len(self.parts)), dtype=wp.vec3)
    self.rotation = wp.zeros((worlds, len(self.parts)), dtype=wp.mat33)
    self.origin = wp.zeros(worlds, dtype=wp.vec3)
    self.facing = wp.zeros(worlds, dtype=wp.mat33)
    self.albedo = wp.array(
      np.stack([part.albedo for part in self.parts]), dtype=wp.vec3
    )
    self.part_body = wp.array(
      np.array([part.body for part in self.parts], np.int32), dtype=wp.int32
    )
    self.cameras = {
      name: _Camera(
        seat(model, name),
        self._housing(model, name, camera),
        size,
        len(self.parts),
        camera,
        worlds,
      )
      for name, camera in cameras.items()
    }
    self._key, self._fill, self._ambient = self._lights(model)

  def render(self, datas: Sequence[mujoco.MjData]) -> list[dict[str, View]]:
    """What each of ``datas`` sees, one entry per data, in one launch per camera."""
    if not 0 < len(datas) <= self.worlds:
      raise ValueError(f"a renderer of {self.worlds} worlds cannot draw {len(datas)}")
    count = len(datas)
    self.position.assign(
      np.ascontiguousarray([data.geom_xpos[self.geoms] for data in datas], np.float32)
    )
    self.rotation.assign(
      np.ascontiguousarray(
        [data.geom_xmat[self.geoms].reshape(-1, 3, 3) for data in datas], np.float32
      )
    )
    seen: list[dict[str, View]] = [{} for _ in range(count)]
    for slot, camera in self.cameras.items():
      self.origin.assign(
        np.ascontiguousarray([data.xpos[camera.body] for data in datas], np.float32)
      )
      self.facing.assign(
        np.ascontiguousarray(
          [data.xmat[camera.body].reshape(3, 3) for data in datas], np.float32
        )
      )
      wp.launch(
        kernels.trace,
        dim=(count, len(camera.pixel)),
        inputs=[
          self.meshes,
          self.position,
          self.rotation,
          camera.seen,
          self.origin,
          self.facing,
          camera.bearings,
          camera.pixel,
          self.albedo,
          self.part_body,
          *self._key,
          *self._fill,
          self._ambient,
          camera.image,
          camera.body_hit,
        ],
      )
      images = camera.image.numpy()[:count].reshape(count, camera.size, camera.size, 3)
      bodies = camera.body_hit.numpy()[:count].reshape(count, camera.size, camera.size)
      for world in range(count):
        seen[world][slot] = View(image=images[world].copy(), body=bodies[world].copy())
    return seen

  def _housing(self, model: mujoco.MjModel, name: str, camera: CameraCfg) -> set[int]:
    """What a lens sits inside, and so cannot see."""
    if not camera.housed:
      return set()
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_base")
    if base < 0:
      raise ValueError(f"{name} is housed, and the scene has no {name}_base body")
    return {index for index, part in enumerate(self.parts) if part.body == base}

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
