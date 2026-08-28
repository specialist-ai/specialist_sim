"""One ray per pixel, shaded by a key light with a shadow and a flat fill."""

import warp as wp


@wp.kernel
def place(
  local: wp.array(dtype=wp.vec3),
  vertex_geom: wp.array(dtype=wp.int32),
  position: wp.array(dtype=wp.vec3),
  rotation: wp.array(dtype=wp.mat33),
  out: wp.array(dtype=wp.vec3),
):
  i = wp.tid()
  geom = vertex_geom[i]
  out[i] = rotation[geom] * local[i] + position[geom]


@wp.kernel
def trace(
  mesh: wp.uint64,
  occluders: wp.uint64,
  origin: wp.vec3,
  camera: wp.mat33,
  bearings: wp.array(dtype=wp.vec3),
  pixel: wp.array(dtype=wp.int32),
  face_geom: wp.array(dtype=wp.int32),
  albedo: wp.array(dtype=wp.vec3),
  face_body: wp.array(dtype=wp.int32),
  key_dir: wp.vec3,
  key_diffuse: wp.vec3,
  fill_dir: wp.vec3,
  fill_diffuse: wp.vec3,
  ambient: wp.vec3,
  radiance: wp.array(dtype=wp.vec3),
  body: wp.array(dtype=wp.int32),
):
  i = wp.tid()
  ray = camera * bearings[i]
  hit = wp.mesh_query_ray(mesh, origin, ray, 50.0)
  if not hit.result:
    return
  normal = hit.normal
  if wp.dot(normal, ray) > 0.0:
    normal = -normal
  point = origin + ray * hit.t
  light = ambient
  lambert = wp.dot(normal, -key_dir)
  if lambert > 0.0:
    if not wp.mesh_query_ray_anyhit(occluders, point + normal * 1.0e-3, -key_dir, 50.0):
      light += key_diffuse * lambert
  lambert = wp.dot(normal, -fill_dir)
  if lambert > 0.0:
    light += fill_diffuse * lambert
  shade = wp.cw_mul(albedo[face_geom[hit.face]], light)
  radiance[pixel[i]] = shade
  body[pixel[i]] = face_body[hit.face]


@wp.kernel
def encode(radiance: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.uint8)):
  i = wp.tid()
  colour = radiance[i]
  for channel in range(3):
    out[i * 3 + channel] = wp.uint8(wp.clamp(colour[channel], 0.0, 1.0) * 255.0 + 0.5)
