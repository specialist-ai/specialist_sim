"""One ray per pixel, shaded by a key light with a shadow and a flat fill.

Every drawn geom is rigid, so its triangles live in a hierarchy built once in
the geom's own frame; a ray is carried into that frame instead of the geometry
being carried into the world. Nothing is ever refit.

A ray that strikes nothing reads the skybox instead, which is a photograph of
the capture room and so already carries its own light: the lights shade geoms,
never the sky.
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


@wp.func
def sky(
  faces: wp.array(dtype=wp.vec3),
  axes: wp.array(dtype=wp.mat33),
  size: int,
  ray: wp.vec3,
) -> wp.vec3:
  """The room photograph along ``ray``, bilinearly.

  The face is the axis the ray leans hardest along; ``axes`` carries that face's
  centre, columns and rows as the columns of a matrix, so the transpose turns a
  ray into a depth and a texel. Samples clamp at a face's edge rather than
  crossing into its neighbour, which shows as a seam only under magnification.
  """
  lean = wp.vec3(wp.abs(ray[0]), wp.abs(ray[1]), wp.abs(ray[2]))
  face = int(0)
  if lean[0] >= lean[1] and lean[0] >= lean[2]:
    if ray[0] <= 0.0:
      face = 1
  elif lean[1] >= lean[2]:
    face = 5
    if ray[1] <= 0.0:
      face = 4
  else:
    face = 2
    if ray[2] <= 0.0:
      face = 3

  across = wp.transpose(axes[face]) * ray
  half = 0.5 * float(size)
  x = across[1] / across[0] * half + half - 0.5
  y = across[2] / across[0] * half + half - 0.5
  left = int(wp.floor(x))
  top = int(wp.floor(y))
  along = x - float(left)
  down = y - float(top)

  base = face * size * size
  x0 = wp.clamp(left, 0, size - 1)
  x1 = wp.clamp(left + 1, 0, size - 1)
  y0 = wp.clamp(top, 0, size - 1) * size
  y1 = wp.clamp(top + 1, 0, size - 1) * size
  upper = faces[base + y0 + x0] * (1.0 - along) + faces[base + y0 + x1] * along
  lower = faces[base + y1 + x0] * (1.0 - along) + faces[base + y1 + x1] * along
  return upper * (1.0 - down) + lower * down


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
  faces: wp.array(dtype=wp.vec3),
  axes: wp.array(dtype=wp.mat33),
  size: int,
  image: wp.array(dtype=wp.uint8),
  body: wp.array(dtype=wp.int32),
):
  i = wp.tid()
  ray = camera * bearings[i]
  struck, along, normal = nearest(meshes, position, rotation, seen, origin, ray, REACH)
  p = pixel[i]
  if struck < 0:
    room = sky(faces, axes, size, ray)
    for channel in range(3):
      image[p * 3 + channel] = wp.uint8(wp.clamp(room[channel], 0.0, 1.0) * 255.0 + 0.5)
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
