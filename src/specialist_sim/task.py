"""Pick and place: the cube is taken off the tape and put back on it.

The cube begins on the tape, so sitting there is the state the episode is handed
rather than something it achieved. It counts as placed once it has been off the
tape and is on it again, at rest and let go of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from specialist_sim.scene import Scene

__all__ = ["TaskCfg", "PickPlace"]


@dataclass(frozen=True)
class TaskCfg:
  place_tolerance: float = 0.010
  rest_height: float = 0.005
  rest_speed: float = 0.02
  rest_spin: float = 0.5
  upright_cosine: float = 0.966
  drop_depth: float = 0.10
  settle_steps: int = 5


class PickPlace:
  def __init__(self, scene: Scene, cfg: TaskCfg):
    self.scene, self.cfg = scene, cfg
    self._settled_for = 0
    self._succeeded = False
    self._disturbed = False

  def reset(self) -> None:
    self._settled_for = 0
    self._succeeded = False
    self._disturbed = False

  def evaluate(self) -> tuple[float, bool, dict[str, Any]]:
    report = self.info()
    self._disturbed |= not report["on_target"]
    settled = self._disturbed and report["settled"]
    self._settled_for = self._settled_for + 1 if settled else 0
    won = self._settled_for >= self.cfg.settle_steps and not self._succeeded
    self._succeeded |= won
    report["is_success"] = self._succeeded
    return float(won), self._succeeded or report["dropped"], report

  def info(self) -> dict[str, Any]:
    """What the cube is doing, as both reset and step report it."""
    scene, cfg = self.scene, self.cfg
    data, cube = scene.data, scene.cube
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
    """Whether a pad and the cube share a contact."""
    pairs = self.scene.data.contact.geom[: self.scene.data.ncon]
    touching = pairs[(pairs == self.scene.cube.geom).any(axis=1)]
    return bool(np.isin(touching, list(self.scene.pads)).any())
