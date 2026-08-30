"""A task: the hands a policy drives, where the episode opens, and what ends it.

The rig is always whole. Hands a task does not drive are reset, held in the
air and seen by every lens, but absent from the observation and the action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.randomise import Kicked, OnTape, Region, Start
from specialist_sim.scene import Scene
from specialist_sim.spaces import GRIPPERS

__all__ = [
  "BIMANUAL",
  "Goal",
  "Judge",
  "PlacedOnTape",
  "RestingIn",
  "Task",
  "TaskCfg",
  "left_task",
  "right_task",
  "task_for",
]


@dataclass(frozen=True)
class TaskCfg:
  """The thresholds the cube's report is judged against."""

  place_tolerance: float = 0.010
  rest_height: float = 0.005
  rest_speed: float = 0.02
  rest_spin: float = 0.5
  upright_cosine: float = 0.966
  drop_depth: float = 0.10
  settle_steps: int = 5


class Goal(Protocol):
  def reached(self, report: dict[str, Any]) -> bool: ...


@dataclass(frozen=True)
class PlacedOnTape:
  """The cube back on the tape, at rest and let go of, having been off it."""

  def reached(self, report: dict[str, Any]) -> bool:
    return bool(report["displaced"] and report["settled"])


@dataclass(frozen=True)
class RestingIn:
  """The cube at rest off the tape, inside the region a pick begins in."""

  region: Region

  def reached(self, report: dict[str, Any]) -> bool:
    return bool(
      report["resting"]
      and not report["on_target"]
      and not report["held"]
      and self.region.contains(report["cube_pos"][:2])
    )


@dataclass(frozen=True)
class Task:
  driven: tuple[str, ...]
  start: Start
  goal: Goal
  seen: tuple[str, ...] = ()
  """Lenses the observation carries beyond the driven hands' own."""
  cfg: TaskCfg = field(default_factory=TaskCfg)

  def __post_init__(self) -> None:
    for name, hands in (("drives", self.driven), ("sees", self.seen)):
      order = tuple(slot for slot in GRIPPERS if slot in hands)
      if (name == "drives" and not hands) or order != hands:
        raise ValueError(
          f"a task {name} some of {GRIPPERS}, in that order, not {hands}"
        )

  @property
  def lenses(self) -> tuple[str, ...]:
    return tuple(slot for slot in GRIPPERS if slot in self.driven or slot in self.seen)


BIMANUAL = Task(GRIPPERS, OnTape(), PlacedOnTape())


def right_task(region: Region) -> Task:
  return Task((GRIPPERS[1],), Kicked(region), PlacedOnTape())


def left_task(region: Region) -> Task:
  return Task((GRIPPERS[0],), OnTape(), RestingIn(region))


def task_for(driven: tuple[str, ...], region: Region) -> Task:
  """The named task a policy driving ``driven`` is evaluated on."""
  named = {
    GRIPPERS: BIMANUAL,
    (GRIPPERS[1],): right_task(region),
    (GRIPPERS[0],): left_task(region),
  }
  if tuple(driven) not in named:
    raise ValueError(f"no task drives {driven}; the tasks drive {list(named)}")
  return named[tuple(driven)]


class Judge:
  """One world's judge: the goal, latched once it has held for ``settle_steps``."""

  def __init__(self, scene: Scene, data: mujoco.MjData, task: Task):
    self.scene, self.data, self.goal, self.cfg = scene, data, task.goal, task.cfg
    self.reset()

  def reset(self) -> None:
    self._settled_for = 0
    self._succeeded = False
    self._displaced = False

  def evaluate(self) -> tuple[float, bool, dict[str, Any]]:
    report = self.info()
    settled = self.goal.reached(report)
    self._settled_for = self._settled_for + 1 if settled else 0
    won = self._settled_for >= self.cfg.settle_steps and not self._succeeded
    self._succeeded |= won
    report["is_success"] = self._succeeded
    return float(won), self._succeeded or report["dropped"], report

  def info(self) -> dict[str, Any]:
    """What the cube is doing, as both reset and step report it."""
    scene, cfg = self.scene, self.cfg
    data, cube = self.data, scene.cube
    position: Float64Array = data.xpos[cube.body]
    velocity = data.cvel[cube.body]
    upright = float(np.abs(data.xmat[cube.body].reshape(3, 3)[2]).max())
    on_target = self.on_target(position)
    self._displaced |= not on_target
    report = {
      "is_success": self._succeeded,
      "on_target": on_target,
      "displaced": self._displaced,
      "resting": bool(
        abs(position[2] - cube.rest) < cfg.rest_height
        and np.linalg.norm(velocity[3:]) < cfg.rest_speed
        and np.linalg.norm(velocity[:3]) < cfg.rest_spin
        and upright > cfg.upright_cosine
      ),
      "held": self._held(),
      "dropped": bool(position[2] < scene.table_top - cfg.drop_depth),
      "cube_pos": np.asarray(position, np.float32).copy(),
    }
    report["settled"] = bool(on_target and report["resting"] and not report["held"])
    return report

  def on_target(self, position: Float64Array) -> bool:
    """Whether a cube at ``position`` stands on the tape."""
    scene, cfg = self.scene, self.cfg
    return bool(
      abs(position[0] - scene.tape[0]) <= scene.tape_half[0] + cfg.place_tolerance
      and abs(position[1] - scene.tape[1]) <= scene.tape_half[1] + cfg.place_tolerance
    )

  def _held(self) -> bool:
    pairs = self.data.contact.geom[: self.data.ncon]
    touching = pairs[(pairs == self.scene.cube.geom).any(axis=1)]
    return bool(np.isin(touching, self.scene.pads).any())
