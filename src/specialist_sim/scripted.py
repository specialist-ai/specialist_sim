"""A scripted policy written as stages: each carries one hand along a
minimum-jerk curve to a waypoint a script names and works the fingers over a
window of the move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from specialist_sim import rotation
from specialist_sim.action import hold
from specialist_sim.arrays import Float64Array
from specialist_sim.gripper import OPEN, SHUT

__all__ = [
  "Grip",
  "Move",
  "Script",
  "ScriptedPolicy",
  "ScriptedPolicyCfg",
  "Segment",
  "Stage",
  "Waypoint",
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
class ScriptedPolicyCfg:
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
class Waypoint:
  """Where a stage sends its hand: what a script answers when the stage begins."""

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
  """An env's rules: what each stage of a cycle does there."""

  @property
  def stages(self) -> tuple[Stage, ...]: ...

  @property
  def reach(self) -> tuple[Float64Array, Float64Array]:
    """The corner to corner box a hand's base is kept inside."""
    ...

  def opens(self, hands: Float64Array, env_state: Any) -> int:
    """The stage a cycle begins at, given the env state."""
    ...

  def waypoint(
    self, stage: Stage, hands: Float64Array, pose: Float64Array, env_state: Any
  ) -> Waypoint:
    """Where a stage sends its hand, from the plant's state, ``(hands, 8)``,
    and the last commanded one."""
    ...

  def arrived(
    self, stage: Stage, elapsed: float, hands: Float64Array, env_state: Any
  ) -> bool | None:
    """Whether the plant has answered a stage whose move has run, or
    ``None`` where nearness to the goal decides."""
    ...

  def finished(self, env_state: Any) -> bool:
    """Whether the env state is as the task wants it."""
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
class Segment:
  """One hand's motion over one stage: where it goes, what the fingers do,
  the wrist it turns from and to, and how much of the move the turn takes."""

  move: Move
  grip: Grip
  turn: tuple[Float64Array, Float64Array]
  turning: float = 1.0
  elapsed: float = 0.0

  def lay(self, out: Float64Array, hand: int, action_dt: float) -> None:
    """Write this segment's actions for one hand into an action chunk,
    ``(chunk_size, hands, 8)``."""
    whens = self.elapsed + np.arange(1, len(out) + 1) * action_dt
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


class ScriptedPolicy:
  """The rule-based policy: a script's stages, paced and carried out."""

  def __init__(
    self,
    script: Script,
    rng: np.random.Generator,
    action_dt: float,
    n_action_steps: int,
    cfg: ScriptedPolicyCfg | None = None,
  ):
    self.script, self.rng = script, rng
    self.action_dt, self.n_action_steps = action_dt, n_action_steps
    """The script, the stream every draw comes from, the action period of the
    chunks it emits, and how many of those actions a control step executes."""
    self.cfg = cfg or ScriptedPolicyCfg()
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
    self._segment: Segment | None = None

  def predict_action_chunk(self, state: Float64Array, env_state: Any) -> Float64Array:
    """The action chunk the demonstration commands, ``(CHUNK_SIZE, hands, 8)``,
    from every hand's state, ``(hands, 8)``."""
    reached = np.asarray(state, np.float64)
    if self._pose is None:
      self._pose = reached.copy()
    if not self.done:
      self._settle(reached, env_state)
    out = hold(self._pose)
    if self._segment is not None:
      self._segment.lay(out, self.script.stages[self.stage].hand, self.action_dt)
      self._segment.elapsed += self.n_action_steps * self.action_dt
      self._clock += self.n_action_steps * self.action_dt
      out[:, :, :3] = np.clip(out[:, :, :3], *self.script.reach)
      out[:, :, 7] = np.clip(out[:, :, 7], SHUT, OPEN)
      self._pose = out[self.n_action_steps - 1].copy()
    return out

  def _settle(self, reached: Float64Array, env_state: Any) -> None:
    """Plan the stage that is due, and leave it once its move has run and the
    plant has answered."""
    if self._segment is not None and self._arrived(reached, env_state):
      self._step_on(env_state)
    if self._segment is None and not self.done:
      if self.stage == 0:
        self.stage = self.script.opens(reached, env_state)
      self._plan(reached, env_state)

  def _step_on(self, env_state: Any) -> None:
    """Retire the stage, and the cycle with it once the last one is done."""
    assert self._segment is not None
    stages = self.script.stages
    hand = stages[self.stage].hand
    rate = self._segment.move.rate_at(self._segment.elapsed)
    limit = self.cfg.flow * self.cfg.speed
    fast = float(np.linalg.norm(rate))
    if fast > limit:
      rate = rate * (limit / fast)
    self._segment = None
    self.stage += 1
    if self.stage < len(stages):
      self._carry = rate if stages[self.stage].hand == hand else np.zeros(3)
      return
    self._carry = np.zeros(3)
    self.stage = 0
    self.cycles += 1
    self._pace = float(self.rng.uniform(*self.cfg.pace))
    self.done = self._clock >= self._length and (
      self.script.finished(env_state) or self._clock >= self._length + self.cfg.rescue
    )

  def _arrived(self, reached: Float64Array, env_state: Any) -> bool:
    assert self._segment is not None
    segment, stage = self._segment, self.script.stages[self.stage]
    if segment.elapsed > segment.move.span + DWELL:
      return True
    if segment.elapsed < segment.move.span:
      return False
    answer = self.script.arrived(stage, segment.elapsed, reached, env_state)
    if answer is not None:
      return answer
    return (
      float(
        np.linalg.norm(reached[stage.hand, :3] - segment.move.at(segment.move.span))
      )
      < REACHED
    )

  def _plan(self, reached: Float64Array, env_state: Any) -> None:
    """Lay a curve from where the command stands to where the stage sends it."""
    assert self._pose is not None
    stage = self.script.stages[self.stage]
    waypoint = self.script.waypoint(stage, reached, self._pose, env_state)
    start = self._pose[stage.hand, :3].copy()
    held = float(self._pose[stage.hand, 7])
    span = self._paced() * max(
      BRIEF,
      self._travel(start, waypoint.goal, waypoint.speed, waypoint.bulge),
      self._gripping(stage, held, waypoint.opening),
    )
    landing = (
      None
      if waypoint.landing is None
      else waypoint.landing * self.cfg.flow * self.cfg.speed
    )
    self._segment = Segment(
      Move.plan(start, self._carry, waypoint.goal, span, waypoint.bulge, landing),
      Grip(held, waypoint.opening, *stage.window),
      (self._pose[stage.hand, 3:7].copy(), rotation.from_matrix(waypoint.facing)),
      waypoint.turning,
    )

  def _paced(self) -> float:
    """How much this cycle stretches its stages, and how the episode tires."""
    gone = min(self._clock / max(self._length, self.action_dt), 1.0)
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
