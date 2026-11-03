"""The LEGO sorting task."""

from dataclasses import dataclass
from typing import ClassVar, Literal

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.env import Lighting, Objects, Task, Verdict
from specialist_sim.tasks import desk
from specialist_sim.tasks.sort_lego import initial, scene
from specialist_sim.tasks.sort_lego.measures import (
  BRICK_COLORS,
  COLORS,
  TRAY_CENTER,
  in_bin,
  on_tray,
)


@dataclass(frozen=True)
class SortLego(Task):
  """Sort every brick that starts on the tray into the bin of its color. It is
  won once they are all there; no label fails it."""

  mode: Literal["drop", "pile"] = "drop"
  """How the tray is laid out."""
  bricks: int | None = None
  """How many bricks a tray starts with. ``None`` draws 1 to 8."""

  # A brighter key than the room's, both lights from behind the operator.
  lighting: ClassVar[Lighting] = Lighting(
    key=(0.4, 0.0, -1.0),
    key_diffuse=0.7,
    fill=(0.6, 0.5, -0.6),
    fill_diffuse=0.25,
    ambient=0.35,
  )
  latched: ClassVar[tuple[str, ...]] = ("missorted", "lost")
  focus: ClassVar[Float64Array] = np.array([*TRAY_CENTER, desk.TOP])

  @property
  def settle(self) -> float:
    return initial.MODES[self.mode].settle

  def spec(self) -> mujoco.MjSpec:
    return scene.spec()

  def draw(
    self, rng: np.random.Generator, driven: tuple[str, ...]
  ) -> tuple[Float64Array, Float64Array]:
    return initial.starts(driven), initial.draw(self.mode, rng, self.bricks)

  def judge(self, objects: Objects) -> Verdict:
    own = np.array([COLORS.index(color) for color in BRICK_COLORS])
    active = in_bin(objects.laid[..., :3]) < 0
    landed = in_bin(objects.poses[..., :3])
    fallen = desk.fallen(objects.poses[..., 2])
    loose = active & (landed < 0)
    counts = {
      "started": active.sum(-1),
      "sorted": (active & (landed == own)).sum(-1),
      "missorted": (active & (landed >= 0) & (landed != own)).sum(-1),
      "on_tray": (loose & ~fallen & on_tray(objects.poses[..., :3])).sum(-1),
      "lost": (loose & fallen).sum(-1),
    }
    return Verdict(
      progress=counts["sorted"] / counts["started"],
      labels={"missorted": counts["missorted"] > 0, "lost": counts["lost"] > 0},
      counts=counts,
    )
