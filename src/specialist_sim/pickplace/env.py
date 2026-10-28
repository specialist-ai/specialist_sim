"""The pick-and-place environment: the rig standing in the room, and the task
judged on every world after every step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from specialist_sim import rig
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.bench import Outcome
from specialist_sim.camera import Camera
from specialist_sim.embodiment import Embodiment, Floating, OutOfReach
from specialist_sim.pickplace import room as rooms
from specialist_sim.pickplace.room import ASSETS, Room
from specialist_sim.pickplace.task import (
  BIMANUAL,
  SETTLE_STEPS,
  Judgment,
  Start,
  Task,
  look,
  on_tape,
)
from specialist_sim.rig import Clock, Observation, Rig

__all__ = ["ATTEMPTS", "BUCKETS", "PICK_LIFT", "Cfg", "PickPlace", "Report", "bucket"]

BUCKETS = ("broke", "clean", "slid", "lost", "neither")
"""How an episode of this scene ended."""


def bucket(outcome: Outcome) -> str:
  """Which of ``BUCKETS`` an episode fell in."""
  if outcome.broke:
    return "broke"
  carried = outcome.labels["picked"]
  if outcome.success:
    return "clean" if carried else "slid"
  return "lost" if carried else "neither"


ATTEMPTS = 20
"""How many starts a reset draws for a carrier that cannot reach one."""
PICK_LIFT = 0.010
"""How far off the table the cube must be carried for an episode to count as
a pick rather than a nudge."""


@dataclass(frozen=True)
class Cfg:
  clock: Clock = rig.Cfg.clock
  embodiment: Embodiment = field(default_factory=Floating)
  cameras: dict[str, Camera] = field(default_factory=dict)
  image_size: int = rig.Cfg.image_size
  image_lag: float = rig.Cfg.image_lag
  """Seconds a frame takes to reach a policy once the shutter has fallen."""
  task: Task = BIMANUAL
  fiducials: bool = False
  """Whether the tag stands are in the room."""

  def rig(self) -> rig.Cfg:
    return rig.Cfg(
      world=partial(rooms.spec, self.fiducials),
      embodiment=self.embodiment,
      clock=self.clock,
      cameras=self.cameras,
      image_size=self.image_size,
      image_lag=self.image_lag,
      driven=self.task.driven,
      seen=self.task.seen,
    )


@dataclass(frozen=True)
class Report(Judgment):
  """How every world's episode stands after a reset or a step."""

  picked: BoolArray
  """Whether the cube has been carried clear of the table, latched."""
  success: BoolArray
  """Whether the goal has held for ``SETTLE_STEPS`` steps, latched."""
  broke: BoolArray
  """Whether the physics could not carry the world, latched."""
  tracking: Float64Array
  """The largest gap between a driven hand and its setpoint, in meters."""
  reward: Float64Array
  """One on the step success latches."""
  done: BoolArray

  @property
  def labels(self) -> dict[str, BoolArray]:
    """What the bench counts of this scene beyond the task being solved."""
    return {"dropped": self.dropped, "picked": self.picked}


class PickPlace:
  assets = (ASSETS,)

  def __init__(self, cfg: Cfg | None = None, worlds: int = 1):
    self.cfg = cfg = cfg or Cfg()
    self.rig = Rig(cfg.rig(), worlds)
    self.room = Room(self.rig.model, self.rig.hands)
    self.worlds = worlds
    self._displaced = np.zeros(worlds, np.bool_)
    self._picked = np.zeros(worlds, np.bool_)
    self._settled = np.zeros(worlds, np.int64)
    self._success = np.zeros(worlds, np.bool_)

  def reset(
    self,
    seeds: Sequence[int],
    worlds: Sequence[int] | None = None,
    start: Start | None = None,
  ) -> tuple[Observation, Report]:
    """Open an episode in every world, or only in the ones named, each drawn
    from its seed by the task's start, or by ``start``."""
    chosen = list(range(self.worlds)) if worlds is None else list(worlds)
    start = start or self.cfg.task.start
    phases = np.empty((len(chosen), len(self.rig.lenses)))
    for index, (world, seed) in enumerate(zip(chosen, seeds, strict=True)):
      rng = np.random.default_rng(seed)
      for _ in range(ATTEMPTS):
        hands, cube = start.draw(self.room, rng)
        try:
          self.rig.place(world, hands)
        except OutOfReach:
          continue
        break
      else:
        raise OutOfReach(
          f"world {world} drew {ATTEMPTS} starts the carrier cannot reach"
        )
      self.room.place_cube(self.rig.datas[world], cube)
      drawn = rng.uniform(size=len(self.rig.slots))
      phases[index] = [drawn[self.rig.slots.index(slot)] for slot in self.rig.lenses]
    observation = self.rig.begin(chosen, phases)
    self._displaced[chosen] = self._picked[chosen] = self._success[chosen] = False
    self._settled[chosen] = 0
    return observation, self._report(self._look(), np.zeros(self.worlds))

  def step(self, action: Float64Array) -> tuple[Observation, Report]:
    """Track the leading poses of every world's trajectory, ``(worlds, poses,
    driven, 8)``, and judge where that leaves the cube."""
    observation = self.rig.step(action)
    judgment = self._look()
    reached = self.cfg.task.goal.reached(judgment)
    self._settled = np.where(reached, self._settled + 1, 0)
    won = (self._settled >= SETTLE_STEPS) & ~self._success
    self._success |= won
    return observation, self._report(judgment, won.astype(np.float64))

  def _look(self) -> Judgment:
    seen = [look(self.room, data) for data in self.rig.datas]
    cube = np.stack([one[0] for one in seen])
    standing = on_tape(self.room, cube)
    held = np.array([one[2] for one in seen])
    self._displaced |= ~standing
    self._picked |= held & (cube[:, 2] > self.room.cube.rest + PICK_LIFT)
    return Judgment(
      cube=cube,
      on_tape=standing,
      resting=np.array([one[1] for one in seen]),
      held=held,
      dropped=np.array([one[3] for one in seen]),
      displaced=self._displaced.copy(),
    )

  def _report(self, judgment: Judgment, reward: Float64Array) -> Report:
    return Report(
      **vars(judgment),
      picked=self._picked.copy(),
      success=self._success.copy(),
      broke=self.rig.broke.copy(),
      tracking=self.rig.tracking.copy(),
      reward=reward,
      done=self._success | judgment.dropped | self.rig.broke,
    )
