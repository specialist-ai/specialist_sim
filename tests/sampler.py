"""The tracer's skybox sampler, called on its own.

Warp's kernel annotations are not Python types, so the launch lives here rather
than in the test module, and is left out of the type check the way
``render/kernels.py`` is.
"""

import numpy as np
import warp as wp

from specialist_sim.render import kernels, sky


@wp.kernel
def _sampled(
  faces: wp.array(dtype=wp.vec3),
  axes: wp.array(dtype=wp.mat33),
  size: int,
  rays: wp.array(dtype=wp.vec3),
  out: wp.array(dtype=wp.vec3),
):
  index = wp.tid()
  out[index] = kernels.sky(faces, axes, size, rays[index])


def read(faces, rays):
  """What the sampler reads along each ray of ``(n, 3)``, given ``(6, s, s, 3)``."""
  wp.init()
  out = wp.zeros(len(rays), dtype=wp.vec3)
  wp.launch(
    _sampled,
    dim=len(rays),
    inputs=[
      wp.array(np.asarray(faces, np.float32).reshape(-1, 3), dtype=wp.vec3),
      wp.array(sky.AXES, dtype=wp.mat33),
      int(np.shape(faces)[1]),
      wp.array(np.asarray(rays, np.float32), dtype=wp.vec3),
      out,
    ],
  )
  return out.numpy()
