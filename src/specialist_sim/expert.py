"""A demonstration written as stages: each carries one hand along a
minimum-jerk curve to a goal a script names and works the fingers over a
window of the move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array
from specialist_sim.gripper import OPEN, SHUT
from specialist_sim.trajectory import hold

__all__ = [
  "Aim",
  "Expert",
  "ExpertCfg",
  "Grip",
  "Leg",
  "Move",
  "Script",
  "Stage",
]

PEAK = 1.875
"""What a minimum-jerk move's fastest moment is, as a multiple of its average."""
REACHED = 0.010
"""How near a goal counts as arrived."""
DWELL = 1.5
"""How long past its own move a stage may run before giving up on the plant."""
BRIEF = 0.25
"""The least time a stage takes, so a move that barely travels still eases in
and out."""


@dataclass(frozen=True)
class ExpertCfg:
  """How a demonstration paces itself. Rates of the fingers are in radians a
  second."""

  across: tuple[float, float] = (0.15, 2.1)
  """How long a move takes: a fixed part plus a share of the distance."""
  speed: float = 0.70
  """The fastest a hand travels, whatever the time its distance asks for."""
  tempo: tuple[float, float] = (0.85, 1.15)
  """How fast one episode runs against another."""
  tire: float = 0.10
  """How much an episode's cycles stretch between its first and its last."""
  flow: float = 0.045
  """How fast a hand is still going as one stage hands over to the next, as a
  share of its top speed."""
  finger_rate: float = 2.5 * (OPEN - SHUT)
  """The fastest the fingers move when nothing between them holds them back."""
  let_go: float = 0.6 * (OPEN - SHUT)
  """The fastest they open on something they still hold."""
  pace: tuple[float, float] = (1.0, 1.35)
  """How far a cycle stretches the time each of its stages takes."""
  length: tuple[float, float] = (80.0, 95.0)
  """How long an episode runs before it stops starting cycles."""
  rescue: float = 12.0
  """How long past its length an episode keeps cycling to end as the task
  wants, after which it stops wherever things lie."""


@dataclass(frozen=True)
class Stage:
  """One stage of a cycle."""

  hand: int
  name: str
  window: tuple[float, float] = (0.0, 1.0)
  """When in the move the fingers work, on the move's own clock."""
  loaded: bool = False
  """Whether something is between the pads, which opens gently."""
  turns: bool = True
  """Whether the wrist turns, or keeps the angle it was handed."""


@dataclass(frozen=True)
class Aim:
  """What a script answers when a stage begins."""

  goal: Float64Array
  """Where the hand's base goes."""
  opening: float
  """The finger angle it arrives with."""
  facing: Float64Array
  """The rotation the wrist turns to, ``(3, 3)``."""
  speed: float | None = None
  """A speed the move is timed by, instead of its distance, under the top speed."""
  bulge: float = 0.0
  """How high the move arches over the straight line between its ends."""
  landing: Float64Array | None = None
  """The direction the move arrives moving in, at the handover speed."""
  turning: float = 1.0
  """The share of the move over which the wrist turns."""


class Script(Protocol):
  """A scene's rules: what each stage of a cycle does there."""

  @property
  def stages(self) -> tuple[Stage, ...]: ...

  @property
  def reach(self) -> tuple[Float64Array, Float64Array]:
    """The corner to corner box a hand's base is kept inside."""
    ...

  def opens(self, hands: Float64Array, world: Any) -> int:
    """The stage a cycle begins at, given how the world stands."""
    ...

  def aim(
    self, stage: Stage, hands: Float64Array, pose: Float64Array, world: Any
  ) -> Aim:
    """Where a stage sends its hand, from the plant's state, ``(hands, 8)``,
    and the last commanded one."""
    ...

  def arrived(
    self, stage: Stage, elapsed: float, hands: Float64Array, world: Any
  ) -> bool | None:
    """Whether the plant has answered a stage whose move has run, or
    ``None`` where nearness to the goal decides."""
    ...

  def finished(self, world: Any) -> bool:
    """Whether the world stands as the task wants it."""
    ...


def _eased(part: float) -> float:
  """The minimum-jerk schedule: flat at both ends, fastest in the middle."""
  held = min(max(part, 0.0), 1.0)
  return float(held**3 * (10.0 - 15.0 * held + 6.0 * held**2))


@dataclass(frozen=True)
class Move:
  """A minimum-jerk curve through a hand's position, leaving and arriving at
  the speed it is given, with a vertical bulge that lifts the path."""

  start: Float64Array
  rate: Float64Array
  span: float
  cubic: Float64Array
  quartic: Float64Array
  quintic: Float64Array
  bulge: float = 0.0

  @classmethod
  def plan(
    cls,
    start: Float64Array,
    rate: Float64Array,
    goal: Float64Array,
    span: float,
    bulge: float = 0.0,
    landing: Float64Array | None = None,
  ) -> Move:
    end = np.zeros(3) if landing is None else landing
    gap = goal - start - rate * span
    lead = (end - rate) * span
    return cls(
      start,
      rate,
      span,
      (10.0 * gap - 4.0 * lead) / span**3,
      (7.0 * lead - 15.0 * gap) / span**4,
      (6.0 * gap - 3.0 * lead) / span**5,
      bulge,
    )

  def at(self, when: float) -> Float64Array:
    lapse = min(max(when, 0.0), self.span)
    out: Float64Array = (
      self.start
      + self.rate * lapse
      + self.cubic * lapse**3
      + self.quartic * lapse**4
      + self.quintic * lapse**5
    )
    out[2] += self.bulge * np.sin(np.pi * self.progress(when))
    return out

  def rate_at(self, when: float) -> Float64Array:
    lapse = min(max(when, 0.0), self.span)
    out: Float64Array = (
      self.rate
      + 3.0 * self.cubic * lapse**2
      + 4.0 * self.quartic * lapse**3
      + 5.0 * self.quintic * lapse**4
    )
    part = min(max(when / self.span, 0.0), 1.0)
    climb = 30.0 * part**2 * (1.0 - part) ** 2 / self.span
    out[2] += self.bulge * np.pi * np.cos(np.pi * self.progress(when)) * climb
    return out

  def progress(self, when: float) -> float:
    """How far through the move it is, on the eased schedule."""
    return _eased(when / self.span)


@dataclass(frozen=True)
class Grip:
  """What the fingers do while a move runs, on the move's own clock."""

  start: float
  end: float
  begins: float
  ends: float

  def at(self, part: float) -> float:
    if part <= self.begins:
      return self.start
    if part >= self.ends:
      return self.end
    within = (part - self.begins) / (self.ends - self.begins)
    return self.start + (self.end - self.start) * _eased(within)


@dataclass
class Leg:
  """One hand's motion over one stage: where it goes, what the fingers do,
  the wrist it turns from and to, and how much of the move the turn takes."""

  move: Move
  grip: Grip
  turn: tuple[Float64Array, Float64Array]
  turning: float = 1.0
  elapsed: float = 0.0

  def lay(self, out: Float64Array, hand: int, period: float) -> None:
    """Write this leg's poses for one hand into a trajectory, ``(poses, hands, 8)``."""
    whens = self.elapsed + np.arange(1, len(out) + 1) * period
    spin = rotation.log(
      rotation.multiply(rotation.conjugate(self.turn[0]), self.turn[1])
    )
    parts = np.array(
      [_eased(float(when) / (self.turning * self.move.span)) for when in whens]
    )
    out[:, hand, 3:7] = rotation.multiply(
      self.turn[0], rotation.exp(parts[:, None] * spin)
    )
    for row, when in enumerate(whens):
      out[row, hand, :3] = self.move.at(float(when))
      out[row, hand, 7] = self.grip.at(float(when) / self.move.span)


class Expert:
  """The rule-based policy: a script's stages, paced and carried out."""

  def __init__(
    self,
    script: Script,
    rng: np.random.Generator,
    period: float,
    tracked: int,
    cfg: ExpertCfg | None = None,
  ):
    self.script, self.rng, self.period, self.tracked = script, rng, period, tracked
    """The script, the stream every draw comes from, the pose period of the
    trajectory it emits, and how many of those poses a control step executes."""
    self.cfg = cfg or ExpertCfg()
    self.stage = 0
    self.cycles = 0
    """How many cycles have finished."""
    self.done = False
    self._length = float(rng.uniform(*self.cfg.length))
    self._tempo = float(rng.uniform(*self.cfg.tempo))
    self._pace = float(rng.uniform(*self.cfg.pace))
    self._carry = np.zeros(3)
    self._clock = 0.0
    self._pose: Float64Array | None = None
    self._leg: Leg | None = None

  def act(self, state: Float64Array, world: Any) -> Float64Array:
    """The trajectory the demonstration commands, ``(HORIZON, hands, 8)``,
    from every hand's state, ``(hands, 8)``."""
    reached = np.asarray(state, np.float64)
    if self._pose is None:
      self._pose = reached.copy()
    if not self.done:
      self._settle(reached, world)
    out = hold(self._pose)
    if self._leg is not None:
      self._leg.lay(out, self.script.stages[self.stage].hand, self.period)
      self._leg.elapsed += self.tracked * self.period
      self._clock += self.tracked * self.period
      out[:, :, :3] = np.clip(out[:, :, :3], *self.script.reach)
      out[:, :, 7] = np.clip(out[:, :, 7], SHUT, OPEN)
      self._pose = out[self.tracked - 1].copy()
    return out

  def _settle(self, reached: Float64Array, world: Any) -> None:
    """Plan the stage that is due, and leave it once its move has run and the
    plant has answered."""
    if self._leg is not None and self._arrived(reached, world):
      self._step_on(world)
    if self._leg is None and not self.done:
      if self.stage == 0:
        self.stage = self.script.opens(reached, world)
      self._plan(reached, world)

  def _step_on(self, world: Any) -> None:
    """Retire the stage, and the cycle with it once the last one is done."""
    assert self._leg is not None
    stages = self.script.stages
    hand = stages[self.stage].hand
    rate = self._leg.move.rate_at(self._leg.elapsed)
    limit = self.cfg.flow * self.cfg.speed
    fast = float(np.linalg.norm(rate))
    if fast > limit:
      rate = rate * (limit / fast)
    self._leg = None
    self.stage += 1
    if self.stage < len(stages):
      self._carry = rate if stages[self.stage].hand == hand else np.zeros(3)
      return
    self._carry = np.zeros(3)
    self.stage = 0
    self.cycles += 1
    self._pace = float(self.rng.uniform(*self.cfg.pace))
    self.done = self._clock >= self._length and (
      self.script.finished(world) or self._clock >= self._length + self.cfg.rescue
    )

  def _arrived(self, reached: Float64Array, world: Any) -> bool:
    assert self._leg is not None
    leg, stage = self._leg, self.script.stages[self.stage]
    if leg.elapsed > leg.move.span + DWELL:
      return True
    if leg.elapsed < leg.move.span:
      return False
    answer = self.script.arrived(stage, leg.elapsed, reached, world)
    if answer is not None:
      return answer
    return (
      float(np.linalg.norm(reached[stage.hand, :3] - leg.move.at(leg.move.span)))
      < REACHED
    )

  def _plan(self, reached: Float64Array, world: Any) -> None:
    """Lay a curve from where the command stands to where the stage sends it."""
    assert self._pose is not None
    stage = self.script.stages[self.stage]
    aim = self.script.aim(stage, reached, self._pose, world)
    start = self._pose[stage.hand, :3].copy()
    held = float(self._pose[stage.hand, 7])
    span = self._paced() * max(
      BRIEF,
      self._travel(start, aim.goal, aim.speed, aim.bulge),
      self._gripping(stage, held, aim.opening),
    )
    landing = (
      None if aim.landing is None else aim.landing * self.cfg.flow * self.cfg.speed
    )
    self._leg = Leg(
      Move.plan(start, self._carry, aim.goal, span, aim.bulge, landing),
      Grip(held, aim.opening, *stage.window),
      (self._pose[stage.hand, 3:7].copy(), rotation.from_matrix(aim.facing)),
      aim.turning,
    )

  def _paced(self) -> float:
    """How much this cycle stretches its stages, and how the episode tires."""
    gone = min(self._clock / max(self._length, self.period), 1.0)
    return self._pace * self._tempo * (1.0 + self.cfg.tire * gone)

  def _travel(
    self, start: Float64Array, goal: Float64Array, speed: float | None, bulge: float
  ) -> float:
    """How long the hand takes to cross: by the speed a stage names, or by its
    distance under a speed ceiling."""
    reach = float(np.linalg.norm(goal - start)) + 2.0 * bulge
    if speed is not None:
      return PEAK * reach / min(speed, self.cfg.speed)
    fixed, per = self.cfg.across
    return max(fixed + per * reach, PEAK * reach / self.cfg.speed)

  def _gripping(self, stage: Stage, held: float, opening: float) -> float:
    """The least time the fingers need, over the part of the move they act in."""
    cfg = self.cfg
    swing = opening - held
    rate = cfg.let_go if swing > 0.0 and stage.loaded else cfg.finger_rate
    low, high = stage.window
    return PEAK * abs(swing) / rate / max(high - low, 1e-3)
