"""One ray per pixel, shaded by a key light with a shadow and a flat fill.

Every drawn geom is rigid, so its triangles live in a hierarchy built once in
the geom's own frame; a ray is carried into that frame instead of the geometry
being carried into the world. Nothing is ever refit.
"""

import warp as wp

REACH = 50.0
"""How far a ray is followed, in metres."""

SHADOW_BIAS = 1.0e-3
"""How far a shadow ray starts off its own surface, so a face cannot shade
itself; shadows cast within this distance of a surface are lost."""


@wp.func
def nearest(
  meshes: wp.array(dtype=wp.uint64),
  position: wp.array(dtype=wp.vec3),
  rotation: wp.array(dtype=wp.mat33),
  seen: wp.array(dtype=wp.int32),
  origin: wp.vec3,
  ray: wp.vec3,
  limit: float,
):
  """The closest part a ray strikes: which, how far, and the world normal."""
  best = limit
  struck = int(-1)
  normal = wp.vec3(0.0, 0.0, 0.0)
  for part in range(meshes.shape[0]):
    if seen[part] == 0:
      continue
    to_part = wp.transpose(rotation[part])
    hit = wp.mesh_query_ray(
      meshes[part], to_part * (origin - position[part]), to_part * ray, best
    )
    if hit.result and hit.t < best:
      best = hit.t
      struck = part
      normal = rotation[part] * hit.normal
  return struck, best, normal


@wp.func
def shadowed(
  meshes: wp.array(dtype=wp.uint64),
  position: wp.array(dtype=wp.vec3),
  rotation: wp.array(dtype=wp.mat33),
  origin: wp.vec3,
  ray: wp.vec3,
) -> bool:
  for part in range(meshes.shape[0]):
    to_part = wp.transpose(rotation[part])
    if wp.mesh_query_ray_anyhit(
      meshes[part], to_part * (origin - position[part]), to_part * ray, REACH
    ):
      return True
  return False


@wp.kernel
def trace(
  meshes: wp.array(dtype=wp.uint64),
  position: wp.array(dtype=wp.vec3),
  rotation: wp.array(dtype=wp.mat33),
  seen: wp.array(dtype=wp.int32),
  origin: wp.vec3,
  camera: wp.mat33,
  bearings: wp.array(dtype=wp.vec3),
  pixel: wp.array(dtype=wp.int32),
  albedo: wp.array(dtype=wp.vec3),
  part_body: wp.array(dtype=wp.int32),
  key_dir: wp.vec3,
  key_diffuse: wp.vec3,
  fill_dir: wp.vec3,
  fill_diffuse: wp.vec3,
  ambient: wp.vec3,
  image: wp.array(dtype=wp.uint8),
  body: wp.array(dtype=wp.int32),
):
  i = wp.tid()
  ray = camera * bearings[i]
  struck, along, normal = nearest(meshes, position, rotation, seen, origin, ray, REACH)
  p = pixel[i]
  if struck < 0:
    for channel in range(3):
      image[p * 3 + channel] = wp.uint8(0)
    body[p] = -1
    return
  if wp.dot(normal, ray) > 0.0:
    normal = -normal
  point = origin + ray * along
  light = ambient
  lambert = wp.dot(normal, -key_dir)
  if lambert > 0.0:
    if not shadowed(meshes, position, rotation, point + normal * SHADOW_BIAS, -key_dir):
      light += key_diffuse * lambert
  lambert = wp.dot(normal, -fill_dir)
  if lambert > 0.0:
    light += fill_diffuse * lambert
  shade = wp.cw_mul(albedo[struck], light)
  for channel in range(3):
    image[p * 3 + channel] = wp.uint8(wp.clamp(shade[channel], 0.0, 1.0) * 255.0 + 0.5)
  body[p] = part_body[struck]
