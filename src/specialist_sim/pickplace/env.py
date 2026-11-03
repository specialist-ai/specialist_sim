"""The pick-and-place environment: the rig standing in the room, and the task's
predicates evaluated on every env after every step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from specialist_sim import rig
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.camera import Camera
from specialist_sim.embodiment import Embodiment, Floating, OutOfReach
from specialist_sim.eval import Outcome
from specialist_sim.pickplace import room as rooms
from specialist_sim.pickplace.room import ASSETS, Room
from specialist_sim.pickplace.task import (
  BIMANUAL,
  SETTLE_STEPS,
  InitialCondition,
  Predicates,
  Task,
  evaluate,
  on_tape,
)
from specialist_sim.rig import Clock, Observation, Rig

__all__ = [
  "ATTEMPTS",
  "BUCKETS",
  "PICK_LIFT",
  "Cfg",
  "PickPlaceEnv",
  "Report",
  "bucket",
]

BUCKETS = ("unstable", "clean", "slid", "lost", "neither")
"""How an episode of this env ended."""


def bucket(outcome: Outcome) -> str:
  """Which of ``BUCKETS`` an episode fell in."""
  if outcome.unstable:
    return "unstable"
  carried = outcome.labels["picked"]
  if outcome.success:
    return "clean" if carried else "slid"
  return "lost" if carried else "neither"


ATTEMPTS = 20
"""How many initial conditions a reset draws for an embodiment that cannot
reach one."""
PICK_LIFT = 0.010
"""How far off the table the cube must be carried for an episode to count as
a pick rather than a nudge."""


@dataclass(frozen=True)
class Cfg:
  clock: Clock = rig.Cfg.clock
  embodiment: Embodiment = field(default_factory=Floating)
  cameras: dict[str, Camera] = field(default_factory=dict)
  image_size: int = rig.Cfg.image_size
  camera_latency: float = rig.Cfg.camera_latency
  """Seconds a frame takes to reach a policy once the shutter has fallen."""
  task: Task = BIMANUAL
  fiducials: bool = False
  """Whether the tag stands are in the room."""

  def rig(self) -> rig.Cfg:
    return rig.Cfg(
      scene=partial(rooms.spec, self.fiducials),
      embodiment=self.embodiment,
      clock=self.clock,
      cameras=self.cameras,
      image_size=self.image_size,
      camera_latency=self.camera_latency,
      driven=self.task.driven,
      seen=self.task.seen,
    )


@dataclass(frozen=True)
class Report(Predicates):
  """How every env's episode stands after a reset or a step."""

  picked: BoolArray
  """Whether the cube has been carried clear of the table, latched."""
  success: BoolArray
  """Whether the goal has held for ``SETTLE_STEPS`` steps, latched."""
  unstable: BoolArray
  """Whether the simulation went unstable, latched."""
  tracking_error: Float64Array
  """The largest gap between a driven hand and its setpoint, in meters."""
  reward: Float64Array
  """One on the step success latches."""
  done: BoolArray

  @property
  def progress(self) -> Float64Array:
    """The task has no milestone short of success."""
    out: Float64Array = self.success.astype(np.float64)
    return out

  @property
  def labels(self) -> dict[str, BoolArray]:
    """What evaluation counts of this env beyond the task being solved."""
    return {"dropped": self.dropped, "picked": self.picked}


class PickPlaceEnv:
  assets = (ASSETS,)

  def __init__(self, cfg: Cfg | None = None, num_envs: int = 1):
    self.cfg = cfg = cfg or Cfg()
    self.rig = Rig(cfg.rig(), num_envs)
    self.room = Room(self.rig.model, self.rig.hands)
    self.num_envs = num_envs
    self._displaced = np.zeros(num_envs, np.bool_)
    self._picked = np.zeros(num_envs, np.bool_)
    self._settled = np.zeros(num_envs, np.int64)
    self._success = np.zeros(num_envs, np.bool_)

  def reset(
    self,
    seeds: Sequence[int],
    env_ids: Sequence[int] | None = None,
    initial_condition: InitialCondition | None = None,
  ) -> tuple[Observation, Report]:
    """Open an episode in every env, or only in the ones named, each drawn
    from its seed by the task's initial condition, or by ``initial_condition``."""
    chosen = list(range(self.num_envs)) if env_ids is None else list(env_ids)
    initial_condition = initial_condition or self.cfg.task.initial_condition
    phases = np.empty((len(chosen), len(self.rig.lenses)))
    for index, (env_idx, seed) in enumerate(zip(chosen, seeds, strict=True)):
      rng = np.random.default_rng(seed)
      for _ in range(ATTEMPTS):
        hands, cube = initial_condition.draw(self.room, rng)
        try:
          self.rig.place(env_idx, hands)
        except OutOfReach:
          continue
        break
      else:
        raise OutOfReach(
          f"env {env_idx} drew {ATTEMPTS} initial conditions the embodiment cannot"
          " reach"
        )
      self.room.place_cube(self.rig.datas[env_idx], cube)
      drawn = rng.uniform(size=len(self.rig.slots))
      phases[index] = [drawn[self.rig.slots.index(slot)] for slot in self.rig.lenses]
    observation = self.rig.begin(chosen, phases)
    self._displaced[chosen] = self._picked[chosen] = self._success[chosen] = False
    self._settled[chosen] = 0
    return observation, self._report(self._evaluate(), np.zeros(self.num_envs))

  def step(self, action: Float64Array) -> tuple[Observation, Report]:
    """Track the leading actions of every env's action chunk, ``(num_envs,
    chunk_size, driven, 8)``, and evaluate where that leaves the cube."""
    observation = self.rig.step(action)
    predicates = self._evaluate()
    reached = self.cfg.task.goal.reached(predicates)
    self._settled = np.where(reached, self._settled + 1, 0)
    won = (self._settled >= SETTLE_STEPS) & ~self._success
    self._success |= won
    return observation, self._report(predicates, won.astype(np.float64))

  def _evaluate(self) -> Predicates:
    seen = [evaluate(self.room, data) for data in self.rig.datas]
    cube = np.stack([one[0] for one in seen])
    standing = on_tape(self.room, cube)
    held = np.array([one[2] for one in seen])
    self._displaced |= ~standing
    self._picked |= held & (cube[:, 2] > self.room.cube.rest + PICK_LIFT)
    return Predicates(
      cube=cube,
      on_tape=standing,
      resting=np.array([one[1] for one in seen]),
      held=held,
      dropped=np.array([one[3] for one in seen]),
      displaced=self._displaced.copy(),
    )

  def _report(self, predicates: Predicates, reward: Float64Array) -> Report:
    return Report(
      **vars(predicates),
      picked=self._picked.copy(),
      success=self._success.copy(),
      unstable=self.rig.unstable.copy(),
      tracking_error=self.rig.tracking_error.copy(),
      reward=reward,
      done=self._success | predicates.dropped | self.rig.unstable,
    )
