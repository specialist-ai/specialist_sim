"""A demonstration written as rules: one hand kicks the cube off the tape, the
other picks it up and puts it back.

The cube starts on the tape, is swept somewhere else, and is then picked up and
set down where it began, so a cycle both poses the task and solves it. An
episode repeats the cycle, each at its own pace, until its length is up.

Every goal is read off the scene at the moment a stage begins, so the
demonstration answers to where the cube actually is rather than to a script of
times. A stage carries the hand along a minimum-jerk quintic and hands its speed
to the stage that follows, so the hand flows through the cube rather than
stopping over it; the fingers close and open partway through a move rather than
in a stage of their own. Stages that travel arch over the table on the way, by
about a fifth of what they cross, rather than climbing, crossing and descending.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from specialist_sim import pose as se3
from specialist_sim.action import slerp
from specialist_sim.arrays import Float64Array
from specialist_sim.env import PickPlaceEnv
from specialist_sim.expert_cfg import ExpertCfg
from specialist_sim.randomise import HOME_ROTATION, home
from specialist_sim.spaces import (
  FINGER_OPEN,
  FINGER_SHUT,
  GRIPPERS,
  HORIZON,
  PER_HAND,
  opening_from_gripper,
)

__all__ = ["Expert", "ExpertCfg", "PICKING"]

KICK, PICK = 0, 1
"""The hand that sweeps the cube off the tape, and the hand that puts it there."""

PICKING = "reach"
"""The stage the pick begins at, once the kick has scattered the cube."""

_PEAK = 1.875
"""What a minimum-jerk move's fastest moment is, as a multiple of its average."""

_REACHED = 0.010
"""How near a goal counts as arrived."""
_SEATED = 0.003
"""How near the table a carried cube must be before it is let go."""
_BRIEF = 0.25
"""The least time a stage takes, so a move that barely travels still eases in
and out of it."""
_DWELL = 1.5
"""How long past its own move a stage may run before the demonstration gives
up waiting for the plant."""


@dataclass(frozen=True)
class _Stage:
  """One stage of the cycle, and everything the machinery asks of it.

  ``arched`` stages travel and so rise over the table on the way — a strike
  stays low, sweeping through the cube rather than over it. ``loaded`` marks
  the one crossing made with the cube between the pads, which rises where an
  empty hand skims and lets go gently where an empty hand opens freely.
  ``window`` is when in a move the fingers do their work, on the move's own
  clock: a hand shuts as it arrives on the cube and opens as it sets the cube
  down, both while it is still going. ``at_cube`` moves end on the cube, so
  the wrist must be still by then; a stage that does not ``turn`` keeps the
  wrist it was handed — set on the cube, or standing still.
  """

  hand: int
  name: str
  arched: bool = False
  loaded: bool = False
  window: tuple[float, float] = (0.0, 1.0)
  at_cube: bool = False
  turns: bool = True


_STAGES = (
  _Stage(KICK, "poise", arched=True, window=(0.0, 0.5), at_cube=True),
  _Stage(KICK, "strike", turns=False),
  _Stage(KICK, "withdraw", arched=True),
  _Stage(PICK, "reach", arched=True, window=(0.55, 1.0), at_cube=True),
  _Stage(PICK, "carry", arched=True, loaded=True, window=(0.80, 1.0), turns=False),
  _Stage(PICK, "away", arched=True, window=(0.0, 0.4)),
  _Stage(PICK, "wait", turns=False),
)

_FROM_TAPE = next(step for step, stage in enumerate(_STAGES) if stage.name == PICKING)
"""Where a cycle starts when the cube is not on the tape to begin with."""


def _eased(part: float) -> float:
  """The minimum-jerk schedule: flat at both ends, fastest in the middle."""
  held = min(max(part, 0.0), 1.0)
  return float(held**3 * (10.0 - 15.0 * held + 6.0 * held**2))


@dataclass(frozen=True)
class _Move:
  """A minimum-jerk curve through a hand's position.

  It leaves and arrives at whatever speed it is given, with no acceleration at
  either end, so one move joins the next without a corner however fast the hand
  is going. An arching move rides a vertical bulge on top of the straight line
  between its ends; the bulge follows the eased schedule, which is flat at both
  ends, so it lifts the path without disturbing either join.
  """

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
  ) -> _Move:
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
    """How far through the move it is, on the eased schedule. The hand's
    rotation and its arch both ride this schedule."""
    return _eased(when / self.span)


@dataclass(frozen=True)
class _Grip:
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


class Expert:
  """The rule-based policy, driven by the scene rather than by a clock."""

  def __init__(
    self, env: PickPlaceEnv, rng: np.random.Generator, cfg: ExpertCfg | None = None
  ):
    self.scene, self.judge, self.rng = env.scene, env.judge, rng
    self.cfg = cfg or ExpertCfg()
    self.period = 1.0 / env.cfg.action_hz
    self.tracked = env.ramp.tracked
    """How many of the poses it emits a control step executes."""
    self.stage = 0
    self.cycles = 0
    """How many kick-and-replace cycles have finished."""
    self.done = False
    self._grip = max(0.0, self.scene.pinch - self.cfg.bite)
    """The opening a hand holds the cube at."""
    self._length = float(rng.uniform(*self.cfg.length))
    self._tempo = float(rng.uniform(*self.cfg.tempo))
    self._pace = float(rng.uniform(*self.cfg.pace))
    self._wrist = np.zeros(len(HOME_ROTATION))
    self._sweep = self.cfg.kick[0]
    self._carry = np.zeros(3)
    self._clock = 0.0
    self._elapsed = 0.0
    self._pose: Float64Array | None = None
    self._move: _Move | None = None
    self._fingers = _Grip(0.0, 0.0, 0.0, 1.0)
    self._turn = (np.zeros(4), np.zeros(4))
    self._turning = 1.0
    self._bearing = np.array([1.0, 0.0, 0.0])
    # The rules below reason in the fraction of its travel the hand has closed,
    # so the box they are held to names that column in the same units.
    self._fingers_at = slice(PER_HAND - 1, None, PER_HAND)
    near, far = self.cfg.reach
    self._low = np.tile([*near, *(-1.0,) * 4, 0.0], (HORIZON, len(GRIPPERS)))
    self._high = np.tile([*far, *(1.0,) * 4, 1.0], (HORIZON, len(GRIPPERS)))
    """The box the rules aim inside: the hand's reach, any turn, and the
    fraction of its travel the fingers have closed."""

  def act(self, state: Float64Array, cube: Float64Array) -> Float64Array:
    """The trajectory the demonstration commands from where it stands."""
    reached = np.asarray(state, np.float64).copy()
    reached[self._fingers_at.start :: PER_HAND] = [
      opening_from_gripper(float(angle))
      for angle in reached[self._fingers_at.start :: PER_HAND]
    ]
    if self._pose is None:
      self._pose = reached.copy()
    if not self.done:
      self._settle(reached, np.asarray(cube, np.float64))
    out = np.tile(self._pose, (HORIZON, 1))
    if self._move is not None:
      at = _STAGES[self.stage].hand * PER_HAND
      whens = self._elapsed + np.arange(1, HORIZON + 1) * self.period
      turning = self._turning * self._move.span
      spin = np.empty(3)
      mujoco.mju_subQuat(spin, self._turn[1], self._turn[0])
      turns = slerp(
        self._turn[0], spin, np.array([_eased(float(w) / turning) for w in whens])
      )
      for row, when in enumerate(whens):
        out[row, at : at + 3] = self._move.at(float(when))
        out[row, at + PER_HAND - 1] = self._fingers.at(float(when) / self._move.span)
        out[row, at + 3 : at + 7] = turns[row]
      self._elapsed += self.tracked * self.period
      self._clock += self.tracked * self.period
      out = np.clip(out, self._low, self._high)
      self._pose = out[self.tracked - 1].copy()
    # Every trajectory leaves by here, so the fraction the rules reason in
    # becomes the angle the interface names exactly once.
    out[:, self._fingers_at] = FINGER_SHUT + out[:, self._fingers_at] * (
      FINGER_OPEN - FINGER_SHUT
    )
    return out

  def _settle(self, reached: Float64Array, cube: Float64Array) -> None:
    """Plan the stage that is due, and leave it once its move has run and the
    plant has answered.

    Arrival is read off the hand, not off what it was told, so a stage cannot
    end before the plant has caught up with the command.
    """
    if self._move is not None and self._arrived(reached, cube):
      self._step_on(cube)
    if self._move is None and not self.done:
      if self.stage == 0 and not self._on_tape(cube):
        # A kick takes the cube off the tape, so a cube already off it has none
        # to take.
        self.stage = _FROM_TAPE
      self._elapsed = 0.0
      self._plan(reached, cube)

  def _on_tape(self, cube: Float64Array) -> bool:
    return self.judge.on_target(cube)

  def _step_on(self, cube: Float64Array) -> None:
    """Retire the stage, and the cycle with it once the last one is done.

    An episode stops on a cube that stands where the task wants it, so a cycle
    whose pick came up short is followed by another that puts it right.
    """
    assert self._move is not None
    hand = _STAGES[self.stage].hand
    # A stage that ends on its deadline rather than on arrival is still moving
    # at whatever the middle of a move runs at, which is not a handover speed.
    rate = self._move.rate_at(self._elapsed)
    limit = self.cfg.flow * self.cfg.speed
    fast = float(np.linalg.norm(rate))
    if fast > limit:
      rate = rate * (limit / fast)
    self._move = None
    self.stage += 1
    if self.stage < len(_STAGES):
      self._carry = rate if _STAGES[self.stage].hand == hand else np.zeros(3)
      return
    self._carry = np.zeros(3)
    self.stage = 0
    self.cycles += 1
    self._pace = float(self.rng.uniform(*self.cfg.pace))
    self.done = self._clock >= self._length and (
      self._on_tape(cube) or self._clock >= self._length + self.cfg.rescue
    )

  def _arrived(self, reached: Float64Array, cube: Float64Array) -> bool:
    cfg, move = self.cfg, self._move
    assert move is not None
    stage = _STAGES[self.stage]
    if self._elapsed > move.span + _DWELL:
      return True
    if self._elapsed < move.span:
      return False
    at = stage.hand * PER_HAND
    if stage.loaded:
      return abs(float(cube[2]) - self.scene.cube.rest) < _SEATED
    if stage.name == "wait":
      return (
        self._elapsed >= cfg.linger
        and float(reached[at + PER_HAND - 1]) > self.scene.pinch
      )
    apart = reached[at : at + 3] - move.at(move.span)
    return float(np.linalg.norm(apart)) < _REACHED

  def _plan(self, reached: Float64Array, cube: Float64Array) -> None:
    """Lay a curve from where the command stands to where the stage sends it."""
    assert self._pose is not None
    stage = _STAGES[self.stage]
    hand = stage.hand
    self._turned(stage)
    at = hand * PER_HAND
    aim, opening, speed = self._aim(stage, reached, cube)
    # Out of reach, a goal would buy the stage the time its distance asks for
    # and the plant would clip it anyway.
    goal = np.clip(aim, self.cfg.reach[0], self.cfg.reach[1])
    start = np.asarray(self._pose[at : at + 3], np.float64).copy()
    held = float(self._pose[at + PER_HAND - 1])
    bulge = self._arch(stage, start, goal)
    span = self._paced() * max(
      _BRIEF,
      self._travel(start, goal, speed, bulge),
      self._gripping(stage, held, opening),
    )
    landing = self._onward(stage, cube, goal)
    self._turn = (
      np.asarray(self._pose[at + 3 : at + 7], np.float64).copy(),
      se3.wxyz(self._facing(hand)),
    )
    self._turning = self.cfg.squaring if stage.at_cube else 1.0
    self._fingers = _Grip(held, opening, *stage.window)
    self._move = _Move.plan(start, self._carry, goal, span, bulge, landing)

  def _turned(self, stage: _Stage) -> None:
    """Draw the wrist this hand travels to the stage that is starting with.

    A hand turns only while it is going somewhere: the stages that are set on
    the cube keep the wrist they took it with, so a grasp is never twisted out
    of the fingers and a sweep holds its line, and the stage that stands still
    keeps the one it arrived on rather than turning on the spot.
    """
    if stage.turns:
      self._wrist[stage.hand] = self.rng.normal(0.0, self.cfg.wrist)

  def _facing(self, hand: int) -> Float64Array:
    """The orientation a hand holds this stage: its rest attitude, turned."""
    turn = float(self._wrist[hand])
    about = np.array(
      [
        [np.cos(turn), -np.sin(turn), 0.0],
        [np.sin(turn), np.cos(turn), 0.0],
        [0.0, 0.0, 1.0],
      ]
    )
    spun: Float64Array = about @ HOME_ROTATION[hand]
    return spun

  def _paced(self) -> float:
    """How much this cycle stretches its stages, and how the episode wearies."""
    gone = min(self._clock / max(self._length, self.period), 1.0)
    return self._pace * self._tempo * (1.0 + self.cfg.tire * gone)

  def _onward(
    self, stage: _Stage, cube: Float64Array, goal: Float64Array
  ) -> Float64Array | None:
    """The speed a stage arrives at, aimed where the next one leads.

    A hand that is still going as it takes the cube or lets it go moves the way
    a person's does, and the stage that follows picks the speed up rather than
    starting again from nothing.
    """
    tape = np.asarray(self.scene.tape, np.float64)
    if stage.name == "reach":
      onward = tape - cube[:2]
    elif stage.loaded:
      onward = home(self.scene, stage.hand)[:2] - tape
    elif stage.name == "poise":
      onward = self._bearing[:2]
    elif stage.name == "strike":
      onward = home(self.scene, stage.hand)[:2] - goal[:2]
    else:
      return None
    length = float(np.linalg.norm(onward))
    if length < 1e-9:
      return None
    pace = self.cfg.flow * self.cfg.speed / length
    return np.array([onward[0] * pace, onward[1] * pace, 0.0])

  def _arch(self, stage: _Stage, start: Float64Array, goal: Float64Array) -> float:
    """How high a stage rides over the table between its ends."""
    if not stage.arched:
      return 0.0
    if not stage.loaded:
      return self.cfg.skim
    share, fixed = self.cfg.lift
    flat = float(np.linalg.norm(goal[:2] - start[:2]))
    return float(min(share * flat + fixed, self.cfg.arch))

  def _travel(
    self, start: Float64Array, goal: Float64Array, speed: float | None, bulge: float
  ) -> float:
    """How long the hand takes to cross. A stage that names a speed is timed by
    it; every other stage is timed by its distance, under a speed ceiling."""
    reach = float(np.linalg.norm(goal - start)) + 2.0 * bulge
    if speed is not None:
      return _PEAK * reach / speed
    fixed, per = self.cfg.across
    return max(fixed + per * reach, _PEAK * reach / self.cfg.speed)

  def _gripping(self, stage: _Stage, held: float, opening: float) -> float:
    """The least time the fingers need, measured against the part of the move
    they act over rather than the whole of it, so a stage that grips late still
    grips gently."""
    cfg = self.cfg
    swing = opening - held
    if swing <= 0.0:
      rate = cfg.finger_rate
    else:
      rate = cfg.let_go if stage.loaded else cfg.finger_rate
    low, high = stage.window
    return _PEAK * abs(swing) / rate / max(high - low, 1e-3)

  def _parked(self, hand: int) -> Float64Array:
    """Where a hand stands once it is done: off on its own side of the tape, a
    step back from it, and never twice in quite the same place."""
    spot = home(self.scene, hand)
    side = spot[:2] - self.scene.tape
    turn = self.rng.uniform(-self.cfg.stray, self.cfg.stray)
    spin = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    bearing = side / float(np.linalg.norm(side))
    spot[:2] = self.scene.tape + spin @ bearing * self.rng.uniform(*self.cfg.park)
    return spot

  def _aim(
    self, stage: _Stage, reached: Float64Array, cube: Float64Array
  ) -> tuple[Float64Array, float, float | None]:
    """Where this stage sends the hand, how open it holds it, and the speed it
    is timed by if it names one."""
    cfg = self.cfg
    rotation = self._facing(stage.hand)
    seat = cube - rotation @ self.scene.grasp
    at = stage.hand * PER_HAND
    if stage.name == "poise":
      self._swept(cube)
      return seat - cfg.swing * self._bearing, 0.0, None
    if stage.name == "strike":
      return seat + cfg.swing * self._bearing, 0.0, self._sweep
    if stage.name in ("withdraw", "away"):
      opening = 0.0 if stage.name == "withdraw" else cfg.clear
      return self._parked(stage.hand), opening, None
    if stage.name == "reach":
      return seat, self._grip, None
    if stage.loaded:
      return (
        self._placed(reached[at : at + 3], rotation, cube),
        self._grip + cfg.freed,
        None,
      )
    return reached[at : at + 3], cfg.clear, None

  def _swept(self, cube: Float64Array) -> None:
    """Pick where this kick lands the cube, and how hard to sweep to put it
    there. It goes to the picking hand's side of the table, so that hand can
    reach it, and to a different spot every cycle."""
    cfg = self.cfg
    toward = home(self.scene, PICK)[:2] - cube[:2]
    lie = toward / max(float(np.linalg.norm(toward)), 1e-9)
    scale = np.asarray(cfg.patch, np.float64)
    across, along = np.clip(self.rng.normal(0.0, scale), -2.0 * scale, 2.0 * scale)
    speed, carries = cfg.kick
    spot = (carries + along) * lie + across * np.array([-lie[1], lie[0]])
    far = max(float(np.linalg.norm(spot)), 1e-9)
    self._bearing = np.array([spot[0] / far, spot[1] / far, 0.0])
    self._sweep = min(speed * far / carries, cfg.speed)

  def _placed(
    self, reached: Float64Array, rotation: Float64Array, cube: Float64Array
  ) -> Float64Array:
    """Where to hold the hand for the cube to come down on the tape.

    Across the table the hand moves by the error of the cube it holds, so a cube
    sitting off-centre between the pads still lands on the tape. How high to
    hold it comes from the grasp instead: the cube rocks as the fingers close on
    it, and a height read off that moment would drive the hand into the table.
    """
    rest = np.array([*self.scene.tape, self.scene.cube.rest])
    over: Float64Array = reached + rest - cube
    over[2] = (rest - rotation @ self.scene.grasp)[2]
    return over
