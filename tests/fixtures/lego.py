"""Where the LEGO room's bricks are."""

from typing import Any

import numpy as np

from specialist_sim.arrays import BoolArray
from specialist_sim.env import Env
from specialist_sim.tasks.sort_lego.measures import BRICK_COLORS


def brick_q(env: Env[Any], brick: int) -> int:
  """Where one brick's pose starts in ``qpos``."""
  return int(env.model.joint(f"brick_{brick}_joint").qposadr[0])


def brick_geoms(env: Env[Any]) -> BoolArray:
  """Whether each geom is a brick's, ``(ngeom,)``."""
  model = env.model
  bodies = [model.body(f"brick_{i}").id for i in range(len(BRICK_COLORS))]
  return np.isin(model.geom_bodyid, bodies)
