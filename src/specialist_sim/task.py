"""A task: the hands a policy drives, where the cube starts, and what ends it.

The rig is always whole. Hands a task does not drive are reset, held in the
air and seen by every lens, but absent from the observation and the action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import mujoco
import numpy as np

from specialist_sim.randomise import Region, Start
from specialist_sim.scene import Scene
from specialist_sim.spaces import GRIPPERS

__all__ = [
  "BIMANUAL",
  "Goal",
  "PickPlace",
  "PlacedOnTape",
  "RestingIn",
  "Task",
  "TaskCfg",
  "left_task",
  "right_task",
]


@dataclass(frozen=True)
class TaskCfg:
  place_tolerance: float = 0.010
  rest_height: float = 0.005
  rest_speed: float = 0.02
  rest_spin: float = 0.5
  upright_cosine: float = 0.966
  drop_depth: float = 0.10
  settle_steps: int = 5


class Goal(Protocol):
  @property
  def cfg(self) -> TaskCfg: ...

  def reached(self, report: dict[str, Any], disturbed: bool) -> bool: ...


@dataclass(frozen=True)
class PlacedOnTape:
  """The cube back on the tape, at rest and let go of, having been off it."""

  cfg: TaskCfg = field(default_factory=TaskCfg)

  def reached(self, report: dict[str, Any], disturbed: bool) -> bool:
    return bool(disturbed and report["settled"])


@dataclass(frozen=True)
class RestingIn:
  """The cube at rest off the tape, inside the region a pick begins in."""

  region: Region
  cfg: TaskCfg = field(default_factory=TaskCfg)

  def reached(self, report: dict[str, Any], disturbed: bool) -> bool:
    return bool(
      report["resting"]
      and not report["on_target"]
      and not report["held"]
      and self.region.contains(report["cube_pos"][:2])
    )


@dataclass(frozen=True)
class Task:
  driven: tuple[str, ...]
  start: Start | None
  """Where the cube opens off the tape; ``None`` opens it on the tape."""
  goal: Goal

  def __post_init__(self) -> None:
    order = tuple(slot for slot in GRIPPERS if slot in self.driven)
    if not self.driven or order != self.driven:
      raise ValueError(
        f"a task drives some of {GRIPPERS}, in that order, not {self.driven}"
      )


BIMANUAL = Task(GRIPPERS, None, PlacedOnTape())


def right_task(region: Region) -> Task:
  return Task((GRIPPERS[1],), region, PlacedOnTape())


def left_task(region: Region) -> Task:
  return Task((GRIPPERS[0],), None, RestingIn(region))


class PickPlace:
  """One world's judge: the goal, latched once it has held for ``settle_steps``."""

  def __init__(self, scene: Scene, data: mujoco.MjData, goal: Goal):
    self.scene, self.data, self.goal = scene, data, goal
    self.cfg = goal.cfg
    self.reset()

  def reset(self) -> None:
    self._settled_for = 0
    self._succeeded = False
    self._disturbed = False

  def evaluate(self) -> tuple[float, bool, dict[str, Any]]:
    report = self.info()
    self._disturbed |= not report["on_target"]
    settled = self.goal.reached(report, self._disturbed)
    self._settled_for = self._settled_for + 1 if settled else 0
    won = self._settled_for >= self.cfg.settle_steps and not self._succeeded
    self._succeeded |= won
    report["is_success"] = self._succeeded
    return float(won), self._succeeded or report["dropped"], report

  def info(self) -> dict[str, Any]:
    """What the cube is doing, as both reset and step report it."""
    scene, cfg = self.scene, self.cfg
    data, cube = self.data, scene.cube
    position = data.xpos[cube.body]
    velocity = data.cvel[cube.body]
    upright = float(np.abs(data.xmat[cube.body].reshape(3, 3)[2]).max())
    report = {
      "is_success": self._succeeded,
      "on_target": bool(
        abs(position[0] - scene.tape[0]) <= scene.tape_half[0] + cfg.place_tolerance
        and abs(position[1] - scene.tape[1]) <= scene.tape_half[1] + cfg.place_tolerance
      ),
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
    report["settled"] = bool(
      report["on_target"] and report["resting"] and not report["held"]
    )
    return report

  def _held(self) -> bool:
    pairs = self.data.contact.geom[: self.data.ncon]
    touching = pairs[(pairs == self.scene.cube.geom).any(axis=1)]
    return bool(np.isin(touching, self.scene.pads).any())
