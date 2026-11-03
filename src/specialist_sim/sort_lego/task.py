"""The LEGO sorting task: a tray of bricks on a desk between two bins, and
where each brick is, read off the physics. A brick is sorted in its own color's
bin, missorted in the other, lost off the desk, or on the tray. A brick in a
hand over its bin is none of these.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import mujoco
import numpy as np

from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.env import Task, Verdict
from specialist_sim.gripper import SLOTS
from specialist_sim.rig import Lighting, Rig
from specialist_sim.sort_lego import assets, initial, scene
from specialist_sim.sort_lego.assets import ASSETS
from specialist_sim.sort_lego.initial import Layout

__all__ = ["COLORS", "Score", "SortLego", "in_bin", "on_tray"]

LIGHTING = Lighting(
  key=(-0.4, 0.0, -1.0),
  key_from=(0.4, 0.0, 1.2),
  key_diffuse=0.7,
  fill=(-0.6, -0.5, -0.6),
  fill_diffuse=0.25,
  ambient=0.35,
)
"""A key over the desk, brighter and nearer than the room's."""
COLORS = ("red",) * 6 + ("blue",) * 6
"""Six bricks of each color: a layout puts some on the tray and parks the rest
in their bins."""
LOST = 0.05
"""How far below the desk top a brick has fallen off the desk, meters."""
CLEAR_TRIES = 50
"""Layouts drawn under hands that stand where they are, each from the next
seed, before one with a brick in a gripper is laid anyway."""
CLEARANCE = 0.002
"""How near a brick may start to a hand standing where it is, meters."""


def in_bin(position: Float64Array, bins: str) -> str | None:
  """The bin of kind ``bins`` a point lies inside, below its rim, or None."""
  return scene.BINS[bins].holding(position)


def on_tray(position: Float64Array) -> bool:
  half = np.asarray(scene.TRAY_SIZE) / 2
  return bool(
    np.all(np.abs(position[:2] - scene.TRAY_CENTER) < half)
    and position[2] < scene.DESK_TOP + 0.05
  )


@dataclass(frozen=True)
class Score:
  """How a tray stands, counted over the bricks that started on it."""

  started: int
  sorted: int
  missorted: int
  on_tray: int
  lost: int

  @property
  def cleared(self) -> bool:
    """Whether every brick that started on the tray is sorted: success."""
    return self.sorted == self.started

  @property
  def progress(self) -> float:
    """The share of the started bricks that are sorted."""
    return self.sorted / self.started

  def as_dict(self) -> dict[str, int | float | bool]:
    return {**asdict(self), "cleared": self.cleared, "progress": self.progress}


@dataclass
class SortLego(Task):
  """Sort every brick that starts on the tray into the bin of its color. It is
  won once they are all there; nothing short of instability ends it."""

  mode: str = "drop"
  bricks: int | None = None
  """How many bricks a tray starts with. ``None`` lets the mode draw its count."""
  colors: tuple[str, ...] = COLORS
  swapped: bool = False
  """Whether every brick and bin is drawn in the other color."""
  bins: str = "akrobins"
  """The kind of bin, a name in ``scene.BINS``: ``akrobins``, the rig's, or
  ``clear``, clear bins with a color card on each."""

  name: ClassVar[str] = "sort_lego"
  lighting: ClassVar[Lighting] = LIGHTING
  assets: ClassVar[tuple[Path, ...]] = (ASSETS,)
  latched: ClassVar[tuple[str, ...]] = ("missorted", "lost")

  def __post_init__(self) -> None:
    if self.bins not in scene.BINS:
      raise ValueError(f"no bins {self.bins!r}, only {tuple(scene.BINS)}")

  def spec(self) -> mujoco.MjSpec:
    return assets.spec(self.colors, self.bins, self.swapped)

  @property
  def focus(self) -> Float64Array:
    return np.array([*scene.TRAY_CENTER, scene.DESK_TOP])

  def bind(self, rig: Rig) -> None:
    super().bind(rig)
    model = rig.model
    self.layouts: list[Layout | None] = [None] * rig.num_envs
    """The layout each env's episode opened with."""
    count = len(self.colors)
    self._brick_q = np.array(
      [model.joint(f"brick_{i}_joint").qposadr[0] for i in range(count)]
    )
    bricks = [model.body(f"brick_{i}").id for i in range(count)]
    self._brick_geoms = np.isin(model.geom_bodyid, bricks)
    hands = [hand.base for hand in rig.hands]
    roots = model.body_rootid[model.geom_bodyid]
    self._hand_geoms = np.isin(roots, model.body_rootid[hands])

  def open(self, env_idx: int, seed: int, start: Any) -> Float64Array:
    """A tray the seed lays, under hands where the task puts them or, given
    ``start``, there: ``(hands, 8)``, or one start per env. Seeds that would
    put a brick in a hand are passed over, and ``layouts`` names the seed laid."""
    rig = self.rig
    hands = (
      initial.starts(rig.cfg.driven)
      if start is None
      else np.broadcast_to(start, (rig.num_envs, len(SLOTS), 8))[env_idx]
    )
    layout = initial.draw(self.mode, seed, self.colors, self.bins, self.bricks)
    for tries in range(1, CLEAR_TRIES):
      if start is None or not self._overlaps(env_idx, layout, hands):
        break
      layout = initial.draw(
        self.mode, seed + tries, self.colors, self.bins, self.bricks
      )
    self.layouts[env_idx] = layout
    self.place(env_idx, layout.poses, hands)
    return np.zeros(len(rig.lenses))

  @property
  def rest(self) -> float:
    return initial.SETTLE[self.mode]

  def judge(self) -> Verdict:
    scores = [self.score(env_idx) for env_idx in range(self.rig.num_envs)]

    def counted(name: str) -> BoolArray:
      return np.array([getattr(score, name) > 0 for score in scores])

    return Verdict(
      progress=np.array([score.progress for score in scores]),
      solved=np.array([score.cleared for score in scores]),
      failed=np.zeros(self.rig.num_envs, np.bool_),
      labels={"missorted": counted("missorted"), "lost": counted("lost")},
    )

  def score(self, env_idx: int) -> Score:
    """How one env's tray stands now, over the bricks that started on it. An
    env that has yet to open an episode has nothing to sort."""
    layout = self.layouts[env_idx]
    if layout is None:
      return Score(1, 0, 0, 0, 0)
    counts = {"sorted": 0, "missorted": 0, "on_tray": 0, "lost": 0}
    poses = self.poses(env_idx)[layout.active]
    for pose, color in zip(poses, np.array(self.colors)[layout.active], strict=True):
      landed = in_bin(pose[:3], self.bins)
      if landed is not None:
        counts["sorted" if landed == color else "missorted"] += 1
      elif pose[2] < scene.DESK_TOP - LOST:
        counts["lost"] += 1
      elif on_tray(pose[:3]):
        counts["on_tray"] += 1
    return Score(started=int(layout.active.sum()), **counts)

  def poses(self, env_idx: int) -> Float64Array:
    """Every brick's pose in one env, ``(k, 7)``."""
    qpos = self.rig.datas[env_idx].qpos
    return np.stack([qpos[at : at + 7] for at in self._brick_q])

  def place(self, env_idx: int, bricks: Float64Array, hands: Float64Array) -> None:
    """Stand every brick, ``(k, 7)``, and every hand, ``(hands, 8)``, in one
    env, from rest. ``Env.reset`` opens an episode there."""
    self.rig.place(env_idx, hands)
    qpos = self.rig.datas[env_idx].qpos
    for at, pose in zip(self._brick_q, bricks, strict=True):
      qpos[at : at + 7] = pose

  def _overlaps(self, env_idx: int, layout: Layout, hands: Float64Array) -> bool:
    """Whether a hand at ``hands`` comes within ``CLEARANCE`` of a brick where
    ``layout`` puts it."""
    self.place(env_idx, layout.poses, hands)
    model, data = self.rig.model, self.rig.datas[env_idx]
    mujoco.mj_forward(model, data)
    for contact in data.contact[: data.ncon]:
      one, other = contact.geom
      crossing = (self._brick_geoms[one] and self._hand_geoms[other]) or (
        self._hand_geoms[one] and self._brick_geoms[other]
      )
      if crossing and contact.dist < CLEARANCE:
        return True
    return False
