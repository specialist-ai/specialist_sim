"""A bare room for the env's own tests, a slab far below the hands and
nothing else, with the hands' own homes; a lens over any room; and a policy
and a command that hold the hands."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from specialist_sim import gripper, rotation
from specialist_sim.arrays import Float64Array, Int32Array, UInt8Array
from specialist_sim.camera.lens import Lens, Mei
from specialist_sim.camera.render import Renderer
from specialist_sim.camera.shutter import CAMERA_HZ
from specialist_sim.env import Cfg, Clock, Env, Observation, Task, hold
from specialist_sim.gripper import DRAWN

# Where the hands rest over the slab.
HOME = np.array(
  [
    [0.45, 0.15, -0.10, 1.0, 0.0, 0.0, 0.0, 0.2],
    [0.45, -0.15, -0.10, 0.0, 0.0, 0.0, 1.0, 0.2],
  ]
)
# Small frames, each reaching a policy the instant its shutter falls.
SMALL = Cfg(image_size=32, camera_latency=0.0)
# The room without lenses: physics and scoring only.
BLIND = Cfg(image_hands=())
# How many actions a test's chunk holds, more than a step executes.
CHUNK = 16
# Rows between one exposure of a lens and its next.
SPAN = 1.0 / (CAMERA_HZ * Clock().row_dt)


@dataclass(frozen=True)
class Bare(Task):
  """A slab below the hands, and nothing to do."""

  def spec(self) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
      name="slab",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=[0.5, 0.0, -0.5],
      size=[1.0, 1.0, 0.02],
      rgba=[0.8, 0.8, 0.75, 1.0],
      group=DRAWN,
    )
    return spec

  def draw(
    self, rng: np.random.Generator, driven: tuple[str, ...]
  ) -> tuple[Float64Array, Float64Array]:
    return HOME, np.zeros((0, 7))


def overhead(pos: tuple[float, float, float], focal: float, size: int) -> Lens:
  """A pinhole at ``pos`` looking straight down, ``focal`` pixels long at
  ``size`` pixels across, centered and unmasked."""
  return Lens(
    params=Mei(fx=focal / size, fy=focal / size, cx=0.5, cy=0.5),
    pos=pos,
    # Straight down, a half turn about x, so the image's x runs along the room's.
    quat=(0.0, 1.0, 0.0, 0.0),
    mask_radius=None,
  )


def look(
  model: mujoco.MjModel, data: mujoco.MjData, lens: Lens, size: int
) -> tuple[UInt8Array, Int32Array]:
  """One frame of ``lens`` seated in the room, of the room as ``data``
  stands: the image and the body each pixel struck."""
  turned = rotation.to_matrix(np.asarray(lens.quat))
  image: UInt8Array
  body: Int32Array
  image, body = Renderer(model, [(0, lens, ())], size).render(
    data.geom_xpos[None],
    data.geom_xmat[None],
    np.array([[lens.pos]]),
    turned.reshape(1, 1, 9),
    [0],
  )
  return image[0, 0], body[0, 0]


def command(
  env: Env[Any], state: Float64Array, seconds: float
) -> tuple[Observation, dict[str, Any]]:
  """Hold one env's hands at ``state``, ``(hands, 8)``, for ``seconds``: the
  last step's observation and info."""
  for _ in range(round(seconds / env.cfg.clock.step_dt)):
    out = env.step(hold(state, CHUNK)[None])
  return out


class Idle:
  """A policy that holds every hand where it stands, and the ``(env, seed)``
  pairs of each reset it is told of."""

  def __init__(self) -> None:
    self.resets: list[list[tuple[int, int]]] = []

  def reset(self, env_ids: Sequence[int], seeds: Sequence[int]) -> None:
    self.resets.append(list(zip(env_ids, seeds, strict=True)))

  def predict_action_chunk(self, observation: Observation) -> Float64Array:
    return hold(observation.state[:, -1], CHUNK)


def hands(env: Env[Any]) -> Float64Array:
  """Every env's hands as they stand, ``(num_envs, hands, 8)``."""
  sim = env.sim
  return gripper.states(env.hands, sim.qpos, sim.xpos, sim.xquat)
