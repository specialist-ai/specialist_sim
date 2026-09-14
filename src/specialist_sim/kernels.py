"""One ray per pixel, shaded by a key light with a shadow and a flat fill."""

import warp as wp

REACH = 50.0
"""How far a ray is followed, in metres."""

SHADOW_BIAS = 1.0e-3
"""How far a shadow ray starts off its own surface, so a face cannot shade itself."""


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
def textured(
  pixels: wp.array(dtype=wp.vec3),
  address: wp.array(dtype=wp.int32),
  width: wp.array(dtype=wp.int32),
  height: wp.array(dtype=wp.int32),
  repeat: wp.array(dtype=wp.vec2),
  half: wp.array(dtype=wp.vec3),
  part: int,
  point: wp.vec3,
  normal: wp.vec3,
  tint: wp.vec3,
) -> wp.vec3:
  """A box's diffuse map at a point on its surface, read bilinearly on the
  face the normal leans hardest along; its tint where it carries no map."""
  start = address[part]
  if start < 0:
    return tint
  lean = wp.vec3(wp.abs(normal[0]), wp.abs(normal[1]), wp.abs(normal[2]))
  u = float(0.0)
  v = float(0.0)
  if lean[2] >= lean[0] and lean[2] >= lean[1]:
    u = point[0] / (2.0 * half[part][0]) + 0.5
    v = point[1] / (2.0 * half[part][1]) + 0.5
  elif lean[1] >= lean[0]:
    u = point[0] / (2.0 * half[part][0]) + 0.5
    v = point[2] / (2.0 * half[part][2]) + 0.5
  else:
    u = point[1] / (2.0 * half[part][1]) + 0.5
    v = point[2] / (2.0 * half[part][2]) + 0.5
  u = u * repeat[part][0]
  v = v * repeat[part][1]
  u = u - wp.floor(u)
  v = v - wp.floor(v)
  columns = width[part]
  rows = height[part]
  x = u * float(columns)
  y = (1.0 - v) * float(rows)
  left = int(wp.floor(x)) % columns
  top = int(wp.floor(y)) % rows
  right = (left + 1) % columns
  bottom = (top + 1) % rows
  along = x - wp.floor(x)
  down = y - wp.floor(y)
  upper = pixels[start + top * columns + left] * (1.0 - along)
  upper += pixels[start + top * columns + right] * along
  lower = pixels[start + bottom * columns + left] * (1.0 - along)
  lower += pixels[start + bottom * columns + right] * along
  return wp.cw_mul(tint, upper * (1.0 - down) + lower * down)


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
  pixels: wp.array(dtype=wp.vec3),
  address: wp.array(dtype=wp.int32),
  width: wp.array(dtype=wp.int32),
  height: wp.array(dtype=wp.int32),
  repeat: wp.array(dtype=wp.vec2),
  half: wp.array(dtype=wp.vec3),
  part_body: wp.array(dtype=wp.int32),
  key_dir: wp.vec3,
  key_diffuse: wp.vec3,
  fill_dir: wp.vec3,
  fill_diffuse: wp.vec3,
  ambient: wp.vec3,
  background: wp.vec3,
  image: wp.array(dtype=wp.uint8),
  body: wp.array(dtype=wp.int32),
):
  i = wp.tid()
  ray = camera * bearings[i]
  struck, along, normal = nearest(meshes, position, rotation, seen, origin, ray, REACH)
  p = pixel[i]
  shade = background
  body[p] = -1
  if struck >= 0:
    if wp.dot(normal, ray) > 0.0:
      normal = -normal
    point = origin + ray * along
    light = ambient
    lambert = wp.dot(normal, -key_dir)
    if lambert > 0.0:
      if not shadowed(
        meshes, position, rotation, point + normal * SHADOW_BIAS, -key_dir
      ):
        light += key_diffuse * lambert
    lambert = wp.dot(normal, -fill_dir)
    if lambert > 0.0:
      light += fill_diffuse * lambert
    to_part = wp.transpose(rotation[struck])
    surface = textured(
      pixels,
      address,
      width,
      height,
      repeat,
      half,
      struck,
      to_part * (point - position[struck]),
      to_part * normal,
      albedo[struck],
    )
    shade = wp.cw_mul(surface, light)
    body[p] = part_body[struck]
  for channel in range(3):
    image[p * 3 + channel] = wp.uint8(wp.clamp(shade[channel], 0.0, 1.0) * 255.0 + 0.5)
