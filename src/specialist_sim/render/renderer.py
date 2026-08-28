"""Rendering the scene through each gripper's fisheye.

Every camera traces the same world-space vertices; they differ only in which
triangles they may hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import warp as wp

from specialist_sim.arrays import Int32Array, UInt8Array
from specialist_sim.calibration import CameraCfg
from specialist_sim.rays import build_rays
from specialist_sim.render import kernels
from specialist_sim.render.scene import Geometry, gather

__all__ = ["Renderer", "View"]


@dataclass(frozen=True)
class View:
  """One camera's frame, and which body each pixel struck."""

  image: UInt8Array
  body: Int32Array


def _seat(model: mujoco.MjModel, name: str, camera: CameraCfg) -> int:
  """Put a camera on the body it rides, and give back that body."""
  for body in (f"{name}_lens", name):
    found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if found >= 0:
      camera.mount(model, body)
      return int(found)
  raise ValueError(f"the scene carries no body for the camera {name}")


class _Camera:
  """One lens: where it rides, what it may hit, and where its pixels land."""

  def __init__(
    self,
    body: int,
    hidden: set[int],
    size: int,
    geometry: Geometry,
    camera: CameraCfg,
  ):
    self.body = body
    rays = build_rays(camera.at(size), size, size, camera.mask_radius)
    self.size = size
    self.bearings = wp.array(rays.bearings, dtype=wp.vec3)
    self.pixel = wp.array(rays.pixel, dtype=wp.int32)
    keep = ~np.isin(geometry.face_geom, sorted(hidden))
    self.faces = wp.array(geometry.faces.reshape(-1, 3)[keep].ravel(), dtype=wp.int32)
    self.face_geom = wp.array(geometry.face_geom[keep], dtype=wp.int32)
    self.face_body = wp.array(geometry.face_body[keep], dtype=wp.int32)
    self.radiance = wp.zeros(size * size, dtype=wp.vec3)
    self.image = wp.zeros(size * size * 3, dtype=wp.uint8)
    self.segment = wp.zeros(size * size, dtype=wp.int32)
    self.mesh: wp.Mesh | None = None


class Renderer:
  """Every camera's view of one scene, re-traced each time the scene moves."""

  def __init__(self, model: mujoco.MjModel, cameras: dict[str, CameraCfg], size: int):
    wp.init()
    self.geometry = gather(model)
    self.vertices = wp.array(self.geometry.vertices, dtype=wp.vec3)
    self.vertex_geom = wp.array(self.geometry.vertex_geom, dtype=wp.int32)
    self.points = wp.zeros(len(self.geometry.vertices), dtype=wp.vec3)
    self.position = wp.zeros(len(self.geometry.geoms), dtype=wp.vec3)
    self.rotation = wp.zeros(len(self.geometry.geoms), dtype=wp.mat33)
    self.albedo = wp.array(self.geometry.albedo, dtype=wp.vec3)
    self.all_faces = wp.array(self.geometry.faces.reshape(-1), dtype=wp.int32)
    self.occluders: wp.Mesh | None = None
    self.cameras = {
      name: _Camera(
        _seat(model, name, camera),
        self._housing(model, name),
        size,
        self.geometry,
        camera,
      )
      for name, camera in cameras.items()
    }
    self._key, self._fill, self._ambient = self._lights(model)

  def _place(self, data: mujoco.MjData) -> None:
    """Carry every vertex into the world, where the geom holding it sits."""
    geoms = self.geometry.geoms
    self.position.assign(np.ascontiguousarray(data.geom_xpos[geoms], np.float32))
    self.rotation.assign(
      np.ascontiguousarray(data.geom_xmat[geoms].reshape(-1, 3, 3), np.float32)
    )
    wp.launch(
      kernels.place,
      dim=len(self.geometry.vertices),
      inputs=[
        self.vertices,
        self.vertex_geom,
        self.position,
        self.rotation,
        self.points,
      ],
    )

  def render(self, data: mujoco.MjData) -> dict[str, View]:
    self._place(data)
    if self.occluders is None:
      self.occluders = wp.Mesh(points=self.points, indices=self.all_faces)
    else:
      self.occluders.refit()
    out = {}
    for slot, camera in self.cameras.items():
      if camera.mesh is None:
        # A hierarchy keeps the shape it is built with; a refit only moves its bounds.
        camera.mesh = wp.Mesh(points=self.points, indices=camera.faces)
      else:
        camera.mesh.refit()
      camera.radiance.zero_()
      camera.segment.fill_(-1)
      wp.launch(
        kernels.trace,
        dim=len(camera.pixel),
        inputs=[
          camera.mesh.id,
          self.occluders.id,
          wp.vec3(*data.xpos[camera.body]),
          wp.mat33(*data.xmat[camera.body]),
          camera.bearings,
          camera.pixel,
          camera.face_geom,
          self.albedo,
          camera.face_body,
          *self._key,
          *self._fill,
          self._ambient,
          camera.radiance,
          camera.segment,
        ],
      )
      wp.launch(
        kernels.encode, dim=camera.size**2, inputs=[camera.radiance, camera.image]
      )
      out[slot] = View(
        image=camera.image.numpy().reshape(camera.size, camera.size, 3).copy(),
        body=camera.segment.numpy().reshape(camera.size, camera.size).copy(),
      )
    return out

  def _housing(self, model: mujoco.MjModel, name: str) -> set[int]:
    """What a lens sits inside, and so cannot see."""
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_base")
    return {
      index
      for index, geom in enumerate(self.geometry.geoms)
      if model.geom_bodyid[geom] == base
    }

  def _lights(
    self, model: mujoco.MjModel
  ) -> tuple[tuple[wp.vec3, wp.vec3], tuple[wp.vec3, wp.vec3], wp.vec3]:
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
