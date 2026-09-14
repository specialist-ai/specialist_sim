"""One environment, standing any task's scene. A task lays each episode and
judges it; the rig, the latches and what ends an episode are the same whatever
the scene, and live here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, ClassVar

import mujoco
import numpy as np

from specialist_sim import rig as rigs
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.camera import Camera
from specialist_sim.embodiment import Embodiment, Floating
from specialist_sim.gripper import SLOTS
from specialist_sim.physics import Physics
from specialist_sim.rig import Clock, Lighting, Observation, Rig

__all__ = ["Cfg", "Env", "Report", "Task", "Verdict"]


@dataclass(frozen=True)
class Cfg:
  """What a scene runs under. Every task names its own defaults."""

  clock: Clock = rigs.Cfg.clock
  physics: Physics = rigs.Cfg.physics
  embodiment: Embodiment = field(default_factory=Floating)
  lighting: Lighting | None = None
  """The light the scene stands under, or ``None`` for the task's own."""
  cameras: dict[str, Camera] = field(default_factory=dict)
  """The lens each hand carries, by slot; a slot left out carries the nominal one."""
  image_size: int = rigs.Cfg.image_size
  camera_latency: float = rigs.Cfg.camera_latency
  driven: tuple[str, ...] = SLOTS
  """The hands the action names; every other hand holds where it was last sent."""
  seen: tuple[str, ...] = ()
  """Slots whose lens the observation carries beyond the driven hands' own."""
  render: bool = rigs.Cfg.render


@dataclass(frozen=True)
class Verdict:
  """What a task makes of every env, one entry per env."""

  progress: Float64Array
  """How far each env's task has come, from 0 to 1: one where it is solved."""
  solved: BoolArray
  """Whether the goal holds right now."""
  failed: BoolArray
  """Whether the episode can no longer be won."""
  labels: dict[str, BoolArray]
  """Whatever else the task counts about each env, by its own name."""


@dataclass(frozen=True)
class Report(Verdict):
  """How every env's episode stands after a reset or a step."""

  success: BoolArray
  """Whether the goal has held ``Task.dwell`` steps together, latched."""
  unstable: BoolArray
  """Whether the simulation went unstable, latched."""
  tracking_error: Float64Array
  """The largest gap between a driven hand and its setpoint, in meters."""
  reward: Float64Array
  """One on the step success latches."""
  done: BoolArray


class Task(ABC):
  """A scene, how an episode of it opens, and what counts as doing well in it."""

  name: ClassVar[str]
  """What an episode calls this scene, and what stands it again. A task is a
  dataclass, so its fields are the knobs an episode records."""
  lighting: ClassVar[Lighting] = Lighting()
  """The light this scene wants, where a caller names none."""
  assets: ClassVar[tuple[Path, ...]] = ()
  """The directories whose files decide what the scene is."""
  dwell: ClassVar[int] = 1
  """Control steps the goal must hold together before success latches."""
  latched: ClassVar[tuple[str, ...]] = ()
  """Labels that mean "this happened", which the env holds once they do."""

  rig: Rig
  """The rig standing the scene, from ``bind``."""

  def __init_subclass__(cls, **named: Any) -> None:
    super().__init_subclass__(**named)
    if not getattr(cls, "name", ""):
      raise TypeError(f"{cls.__name__} must name its scene")

  @abstractmethod
  def spec(self) -> mujoco.MjSpec:
    """The spec the hands are attached into."""

  @property
  @abstractmethod
  def focus(self) -> Float64Array:
    """The room point an overview of this scene looks at."""

  def bind(self, rig: Rig) -> None:
    """Read out of the compiled model whatever laying and judging need."""
    self.rig = rig

  @abstractmethod
  def open(self, env_idx: int, seed: int, start: Any) -> Float64Array:
    """Lay one env's episode from its seed, or where ``start`` says, and say
    where each lens stands in its period, ``(lenses,)`` in ``[0, 1)``."""

  @abstractmethod
  def judge(self) -> Verdict:
    """Where every env stands now. A reset judges every env, including ones
    mid-episode, so this reads the state and advances nothing."""


class Env[T: Task]:
  """``num_envs`` copies of a task's scene, reset and stepped together."""

  def __init__(self, task: T, cfg: Cfg | None = None, num_envs: int = 1):
    self.task: T = task
    cfg = Cfg() if cfg is None else cfg
    self.cfg = cfg = replace(cfg, lighting=cfg.lighting or task.lighting)
    self.num_envs = num_envs
    self.rig = Rig(
      rigs.Cfg(scene=task.spec, **{f.name: getattr(cfg, f.name) for f in fields(Cfg)}),
      num_envs,
    )
    self._held = np.zeros(num_envs, np.int64)
    """Control steps each env's goal has held together."""
    self._success = np.zeros(num_envs, np.bool_)
    self._marks = {name: np.zeros(num_envs, np.bool_) for name in task.latched}
    """Each latched label, held once it has been true."""
    task.bind(self.rig)

  def reset(
    self,
    seeds: Sequence[int],
    env_ids: Sequence[int] | None = None,
    start: Any = None,
  ) -> tuple[Observation, Report]:
    """Open an episode in every env, or only in the ones named, each laid from
    the seed beside it, or where ``start`` says, which the task reads."""
    chosen = list(range(self.num_envs)) if env_ids is None else list(env_ids)
    phases = np.stack(
      [
        self.task.open(env_idx, seed, start)
        for env_idx, seed in zip(chosen, seeds, strict=True)
      ]
    )
    observation = self.rig.begin(chosen, phases)
    self._held[chosen] = 0
    self._success[chosen] = False
    for mark in self._marks.values():
      mark[chosen] = False
    return observation, self._report(self.task.judge(), np.zeros(self.num_envs))

  def step(self, action: Float64Array) -> tuple[Observation, Report]:
    """Track the leading actions of every env's action chunk, ``(num_envs,
    chunk_size, driven, 8)``, and judge where that leaves each env."""
    observation = self.rig.step(action)
    verdict = self.task.judge()
    self._held = np.where(verdict.solved, self._held + 1, 0)
    won = (self._held >= self.task.dwell) & ~self._success
    self._success |= won
    return observation, self._report(verdict, won.astype(np.float64))

  def _report(self, verdict: Verdict, reward: Float64Array) -> Report:
    labels = dict(verdict.labels)
    for name, mark in self._marks.items():
      mark |= labels[name]
      labels[name] = mark.copy()
    return Report(
      **{**vars(verdict), "labels": labels},
      success=self._success.copy(),
      unstable=self.rig.unstable.copy(),
      tracking_error=self.rig.tracking_error.copy(),
      reward=reward,
      done=self._success | verdict.failed | self.rig.unstable,
    )
