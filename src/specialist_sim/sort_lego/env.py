"""The LEGO sorting environment: the rig standing at the 0927 desk, and every
env's tray judged after every step. Success is every brick that started on the
tray sorted into the bin of its color, and progress is the share sorted.

Time runs on 150 Hz rows, the lattice a dataset packs, with actions at 50 Hz
and the lens firing every fifth row, on the rows exactly, as the dataset build
pairs frames with states.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import mujoco
import numpy as np

from specialist_sim import rig
from specialist_sim.arrays import BoolArray, Float64Array, Int64Array
from specialist_sim.camera import NOMINAL, Camera
from specialist_sim.embodiment import Embodiment, Floating
from specialist_sim.gripper import SLOTS
from specialist_sim.physics import Physics
from specialist_sim.rig import Clock, Observation, Rig
from specialist_sim.sort_lego import assets, initial, scene, task
from specialist_sim.sort_lego.initial import Layout

__all__ = ["CLOCK", "COLORS", "LENS", "Cfg", "Report", "SortLegoEnv"]

CLOCK = Clock(physics_dt=1.0 / 1500.0, per_row=10, per_action=3, per_step=36)
"""150 Hz rows of ten physics steps, an action every third row, and a control
step of 36 rows: the 0.24 s a checkpoint at 50 Hz, 30 fps and two observation
steps asks once in, reaching back over a 0.2 s stale frame."""
LENS = replace(NOMINAL, hz=30.0)
"""The rear lens at exactly 30 Hz, so with no phase it fires on every fifth row."""
COLORS = ("red",) * 6 + ("blue",) * 6
"""Six bricks of each color: a layout puts some on the tray and parks the rest
in their bins."""
CLEAR_TRIES = 50
"""Layouts drawn under hands that stand where they are, each from the next
seed, before one with a brick in a gripper is laid anyway."""
CLEARANCE = 0.002
"""How near a brick may start to a hand standing where it is, meters."""


@dataclass(frozen=True)
class Cfg:
  clock: Clock = CLOCK
  embodiment: Embodiment = field(default_factory=Floating)
  physics: Physics = Physics()
  camera: Camera = LENS
  image_size: int = 720
  """Pixels across a lens frame: ``specialist_lerobot.datasets.views.LENS``."""
  camera_latency: float = 0.0
  frames: int = 1
  render: bool = True
  record: bool = False
  """Whether the rig keeps every exposure for a recorder. A dataset build
  traces the lenses from the recorded ``qpos`` instead."""
  driven: tuple[str, ...] = SLOTS
  mode: str = "drop"
  bricks: int | None = None
  """How many bricks a tray starts with. ``None`` lets the mode draw its count."""
  colors: tuple[str, ...] = COLORS
  swapped: bool = False
  """Whether every brick and bin is drawn in the other color."""
  bins: str = "akrobins"
  """The kind of bin, a name in ``scene.BINS``: ``akrobins``, the rig's, or
  ``clear``, the 0927 takes' bins with a card on each."""

  def __post_init__(self) -> None:
    if self.bins not in scene.BINS:
      raise ValueError(f"no bins {self.bins!r}, only {tuple(scene.BINS)}")

  def rig(self) -> rig.Cfg:
    return rig.Cfg(
      scene=partial(assets.spec, self.colors, self.bins, self.swapped),
      embodiment=self.embodiment,
      clock=self.clock,
      physics=self.physics,
      cameras={slot: self.camera for slot in SLOTS},
      image_size=self.image_size,
      driven=self.driven,
      camera_latency=self.camera_latency,
      frames=self.frames,
      render=self.render,
      record=self.record,
    )


@dataclass(frozen=True)
class Report:
  """How every env's tray stands after a reset or a step, counted over the
  bricks that started on it."""

  started: Int64Array
  sorted: Int64Array
  missorted: Int64Array
  on_tray: Int64Array
  lost: Int64Array
  success: BoolArray
  """Whether every started brick is sorted, latched."""
  unstable: BoolArray
  """Whether the physics could not carry the env, latched."""
  tracking_error: Float64Array
  """The largest gap between a driven hand and its setpoint, in meters."""
  progress: Float64Array
  """The share of the started bricks sorted."""
  done: BoolArray

  @property
  def labels(self) -> dict[str, BoolArray]:
    """What the evaluation counts of this scene beyond success."""
    return {"missorted": self.missorted > 0, "lost": self.lost > 0}


class SortLegoEnv:
  assets = (assets.ASSETS,)

  def __init__(self, cfg: Cfg | None = None, num_envs: int = 1):
    self.cfg = cfg = cfg or Cfg()
    self.rig = Rig(cfg.rig(), num_envs)
    self.num_envs = num_envs
    model = self.rig.model
    self.layouts: list[Layout | None] = [None] * num_envs
    """The layout each env's episode opened with."""
    count = len(cfg.colors)
    self._brick_q = np.array(
      [model.joint(f"brick_{i}_joint").qposadr[0] for i in range(count)]
    )
    bricks = [model.body(f"brick_{i}").id for i in range(count)]
    self._brick_geoms = np.isin(model.geom_bodyid, bricks)
    hands = [hand.base for hand in self.rig.hands]
    roots = model.body_rootid[model.geom_bodyid]
    self._hand_geoms = np.isin(roots, model.body_rootid[hands])
    self._success = np.zeros(num_envs, np.bool_)

  def reset(
    self,
    seeds: Sequence[int],
    env_ids: Sequence[int] | None = None,
    hands: Float64Array | None = None,
  ) -> tuple[Observation, Report]:
    """Open an episode in every env, or only in the ones named, each on the
    layout its seed draws. The hands start where the task puts them or, given
    ``hands``, there: ``(hands, 8)`` for every env, or ``(envs, hands, 8)`` one
    start per env opened. Then seeds that would put a brick in a hand are passed
    over, and ``layouts`` names the seed laid."""
    cfg = self.cfg
    chosen = list(range(self.num_envs)) if env_ids is None else list(env_ids)
    given = None if hands is None else np.asarray(hands, np.float64)
    starts = np.broadcast_to(
      initial.starts(cfg.driven) if given is None else given,
      (len(chosen), len(SLOTS), 8),
    )
    for env_idx, seed, start in zip(chosen, seeds, starts, strict=True):
      layout = initial.draw(cfg.mode, seed, cfg.colors, cfg.bins, cfg.bricks)
      for tries in range(1, CLEAR_TRIES):
        if given is None or not self._overlaps(env_idx, layout, start):
          break
        layout = initial.draw(cfg.mode, seed + tries, cfg.colors, cfg.bins, cfg.bricks)
      self.layouts[env_idx] = layout
      self.place(env_idx, layout.poses, start)
    self.rig.settle(chosen, initial.SETTLE[cfg.mode])
    lenses = len(self.rig.lenses)
    observation = self.rig.begin(chosen, np.zeros((len(chosen), lenses)))
    self._success[chosen] = False
    return observation, self._report()

  def step(self, action: Float64Array) -> tuple[Observation, Report]:
    """Track the leading actions of every env's action chunk, ``(num_envs,
    chunk_size, driven, 8)``, and judge where that leaves each tray."""
    observation = self.rig.step(action)
    return observation, self._report()

  def score(self, env_idx: int) -> task.Score:
    """How one env's tray stands now."""
    layout = self.layouts[env_idx]
    assert layout is not None
    return task.score(
      self.bricks(env_idx), self.cfg.colors, layout.active, self.cfg.bins
    )

  def bricks(self, env_idx: int) -> Float64Array:
    """Every brick's pose in one env, ``(k, 7)``."""
    qpos = self.rig.datas[env_idx].qpos
    return np.stack([qpos[at : at + 7] for at in self._brick_q])

  def describe(self, env_idx: int) -> dict[str, Any]:
    """The ``meta`` an episode of one env starts with: the scene, the layout
    as ``initial.draw`` lays it again from ``(mode, seed, bricks)``, and what
    it ran under."""
    layout = self.layouts[env_idx]
    assert layout is not None
    cfg = self.cfg
    return {
      "scene": "sort_lego",
      "layout": {"mode": layout.mode, "seed": layout.seed, "bricks": layout.bricks},
      "active": layout.active.tolist(),
      "colors": list(cfg.colors),
      "swapped": cfg.swapped,
      "bins": cfg.bins,
      "driven": list(cfg.driven),
      "row_hz": 1.0 / cfg.clock.row_dt,
      "physics": cfg.physics.as_dict(),
    }

  def place(self, env_idx: int, bricks: Float64Array, hands: Float64Array) -> None:
    """Stand every brick, ``(k, 7)``, and every hand, ``(hands, 8)``, in one env_idx,
    from rest. ``rig.begin`` opens an episode there."""
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

  def _report(self) -> Report:
    scores = [
      self.score(env_idx)
      if self.layouts[env_idx] is not None
      else task.Score(1, 0, 0, 0, 0)
      for env_idx in range(self.num_envs)
    ]
    self._success |= np.array([score.cleared for score in scores])
    unstable = self.rig.unstable.copy()

    def counted(name: str) -> Int64Array:
      return np.array([getattr(score, name) for score in scores], np.int64)

    return Report(
      started=counted("started"),
      sorted=counted("sorted"),
      missorted=counted("missorted"),
      on_tray=counted("on_tray"),
      lost=counted("lost"),
      success=self._success.copy(),
      unstable=unstable,
      tracking_error=self.rig.tracking_error.copy(),
      progress=np.array([score.progress for score in scores]),
      done=self._success | unstable,
    )
