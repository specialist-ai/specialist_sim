"""What writes the plant's command every substep, whatever carries the hands.

The environment lays a ramp once per control step and hands it to a controller
every physics substep, with the plant's joint coordinates and the buffers a
substep is driven by. A controller writes the finger servos and whatever else
moves its carrier: a wrench on a free rig, or targets for an arm's joints.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import mujoco
import numpy as np

from specialist_sim.arrays import Float64Array

if TYPE_CHECKING:
  from specialist_sim.action import Ramp
  from specialist_sim.embodiment import Hand
  from specialist_sim.scene import Scene

__all__ = ["CTRL", "XFRC", "Controller", "ControllerCfg", "FingerServos"]

CTRL = int(mujoco.mjtState.mjSTATE_CTRL)
"""The actuator inputs: every controller writes these."""
XFRC = int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
"""A wrench on each body, which a controller pulling on a free rig writes."""


class Controller(Protocol):
  drives: int
  """The state components a substep is driven by: ``CTRL``, or ``CTRL | XFRC``."""

  def reset(self, datas: Sequence[mujoco.MjData], worlds: Sequence[int]) -> None:
    """Take up the named worlds where their scenes put them."""
    ...

  def command(
    self,
    ramp: Ramp,
    substep: int,
    qpos: Float64Array,
    qvel: Float64Array,
    ctrl: Float64Array,
    xfrc: Float64Array,
  ) -> None:
    """Write what every world wants for one substep of the ramp.

    ``qpos`` is ``(worlds, nq)`` and ``qvel`` ``(worlds, nv)``; ``ctrl`` is
    ``(worlds, nu)`` and ``xfrc`` is ``(worlds, nbody, 6)``, both zero wherever a
    controller never writes.
    """
    ...


class ControllerCfg(Protocol):
  """A controller's settings, built against the scene they drive."""

  def build(self, scene: Scene, ramp: Ramp) -> Controller: ...


class FingerServos:
  """The finger servos of every hand: the left to an opening, the right mirrored."""

  def __init__(self, hands: Sequence[Hand]):
    self.left = np.asarray([hand.servos[0] for hand in hands], np.int64)
    self.right = np.asarray([hand.servos[1] for hand in hands], np.int64)

  def command(self, ctrl: Float64Array, opening: Float64Array) -> None:
    """``opening`` is ``(worlds, hands)``, signed as the left finger turns."""
    ctrl[:, self.left] = opening
    ctrl[:, self.right] = -opening
