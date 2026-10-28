"""The pick-and-place demonstration: one hand kicks the cube off the tape and
the other picks it up and puts it back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from specialist_sim.arrays import Float64Array
from specialist_sim.gripper import OPEN, SHUT, SLOTS
from specialist_sim.pickplace.env import PickPlaceEnv
from specialist_sim.pickplace.room import Room
from specialist_sim.pickplace.task import HOME_ROTATION, home, on_tape
from specialist_sim.scripted import ScriptedPolicy, ScriptedPolicyCfg, Stage, Waypoint

__all__ = ["KICK", "PICK", "STAGES", "Rules", "Script", "scripted_policy"]

KICK, PICK = 0, 1
"""The hand that sweeps the cube off the tape, and the hand that puts it back."""


SEATED = 0.003
"""How near the table a carried cube must be before it is let go."""

STAGES = (
  Stage(KICK, "poise", window=(0.0, 0.5)),
  Stage(KICK, "strike", turns=False),
  Stage(KICK, "withdraw"),
  Stage(PICK, "reach", window=(0.55, 1.0)),
  Stage(PICK, "carry", loaded=True, window=(0.80, 1.0), turns=False),
  Stage(PICK, "away", window=(0.0, 0.4)),
  Stage(PICK, "wait", turns=False),
)
ARCHED = ("poise", "withdraw", "reach", "carry", "away")
"""The stages that rise over the table on the way; a strike stays low."""
AT_CUBE = ("poise", "reach")
"""The stages that end on the cube, so the wrist must be still by then."""
FROM_TAPE = next(step for step, stage in enumerate(STAGES) if stage.name == "reach")
"""Where a cycle starts when the cube is not on the tape to begin with."""


@dataclass(frozen=True)
class Rules:
  """What a demonstration is like, apart from its pace. Openings are finger
  angles in radians, sized as shares of the travel from shut to open."""

  kick: tuple[float, float] = (0.50, 0.10)
  """The speed a strike sweeps at, and how far that carries the cube."""
  wrist: float = np.radians(10.0)
  """How far a hand turns about the vertical from one stage to the next."""
  squaring: float = 1.05
  """The stretch of a move ending at the cube over which the wrist turns, so
  the pads come at the cube square."""
  patch: tuple[float, float] = (0.035, 0.020)
  """How far a kick scatters where the cube lands, across the line it is
  driven along and along it."""
  reach: tuple[tuple[float, float], tuple[float, float]] = ((0.25, -0.55), (0.85, 0.65))
  """The corners of the patch of table a hand's base is aimed over."""
  rise: float = 0.56
  """How far above the table top a hand's base may be aimed."""
  freed: float = 0.035 * (OPEN - SHUT)
  """How much further than holding the fingers open to let the cube go."""
  bite: float = 0.02 * (OPEN - SHUT)
  """How far inside touching the fingers close on the cube."""
  park: tuple[float, float] = (0.20, 0.25)
  """How far from the tape a hand stands once it is done."""
  stray: float = np.radians(15.0)
  """How far around the tape a parked spot swings."""
  lift: tuple[float, float] = (0.15, 0.010)
  """How high a carried cube rides over its travel: a share of the distance
  plus a fixed part."""
  skim: float = 0.015
  """How high an empty hand rises on its way."""
  arch: float = 0.06
  """The most a hand rises, however far it has to go."""
  swing: float = 0.07
  """How far either side of the cube a kick starts and ends."""
  clear: float = SHUT + 0.40 * (OPEN - SHUT)
  """An opening that clears the cube."""
  linger: float = 0.1
  """How long a hand stands clear after letting go."""


class Script:
  """The pick-and-place rules, read off the room and the cube's position."""

  stages = STAGES

  def __init__(self, room: Room, rng: np.random.Generator, rules: Rules | None = None):
    self.room, self.rng = room, rng
    self.rules = rules or Rules()
    self.reach: tuple[Float64Array, Float64Array] = (
      np.array([*self.rules.reach[0], room.table_top]),
      np.array([*self.rules.reach[1], room.table_top + self.rules.rise]),
    )
    self._side = [SLOTS.index(slot) for slot in room.slots]
    self._grip = max(SHUT, room.pinch - self.rules.bite)
    """The opening a hand holds the cube at."""
    self._wrist = np.zeros(len(room.slots))
    self._sweep = self.rules.kick[0]
    self._bearing = np.array([1.0, 0.0, 0.0])

  def opens(self, hands: Float64Array, cube: Float64Array) -> int:
    """A kick takes the cube off the tape, so a cube already off it has none
    to take."""
    return 0 if on_tape(self.room, cube) else FROM_TAPE

  def finished(self, cube: Float64Array) -> bool:
    return bool(on_tape(self.room, cube))

  def arrived(
    self, stage: Stage, elapsed: float, hands: Float64Array, cube: Float64Array
  ) -> bool | None:
    if stage.loaded:
      return abs(float(cube[2]) - self.room.cube.rest) < SEATED
    if stage.name == "wait":
      return (
        elapsed >= self.rules.linger and float(hands[stage.hand, 7]) > self.room.pinch
      )
    return None

  def waypoint(
    self, stage: Stage, hands: Float64Array, pose: Float64Array, cube: Float64Array
  ) -> Waypoint:
    """Where this stage sends the hand, how open it holds it, and how it
    travels; a hand turns only while it is going somewhere."""
    rules, hand = self.rules, stage.hand
    if stage.turns:
      self._wrist[hand] = self.rng.normal(0.0, rules.wrist)
    facing = self._facing(hand)
    seat = cube - facing @ self.room.grasp
    speed: float | None = None
    if stage.name == "poise":
      self._swept(cube)
      goal, opening = seat - rules.swing * self._bearing, SHUT
    elif stage.name == "strike":
      goal, opening, speed = seat + rules.swing * self._bearing, SHUT, self._sweep
    elif stage.name in ("withdraw", "away"):
      goal = self._parked(hand)
      opening = SHUT if stage.name == "withdraw" else rules.clear
    elif stage.name == "reach":
      goal, opening = seat, self._grip
    elif stage.loaded:
      goal, opening = (
        self._placed(hands[hand, :3], facing, cube),
        self._grip + rules.freed,
      )
    else:
      goal, opening = hands[hand, :3].copy(), rules.clear
    goal = np.clip(goal, *self.reach)
    return Waypoint(
      goal=goal,
      opening=opening,
      facing=facing,
      speed=speed,
      bulge=self._arch(stage, pose[hand, :3], goal),
      landing=self._onward(stage, cube, goal),
      turning=rules.squaring if stage.name in AT_CUBE else 1.0,
    )

  def _facing(self, hand: int) -> Float64Array:
    """The orientation a hand holds this stage: its rest attitude, turned
    about the vertical."""
    turn = float(self._wrist[hand])
    about = np.array(
      [
        [np.cos(turn), -np.sin(turn), 0.0],
        [np.sin(turn), np.cos(turn), 0.0],
        [0.0, 0.0, 1.0],
      ]
    )
    spun: Float64Array = about @ HOME_ROTATION[self._side[hand]]
    return spun

  def _onward(
    self, stage: Stage, cube: Float64Array, goal: Float64Array
  ) -> Float64Array | None:
    """The way a stage is still going as it arrives: toward where the next
    one leads."""
    tape = self.room.tape
    if stage.name == "reach":
      onward = tape - cube[:2]
    elif stage.loaded:
      onward = home(self.room, self._side[stage.hand])[:2] - tape
    elif stage.name == "poise":
      onward = self._bearing[:2]
    elif stage.name == "strike":
      onward = home(self.room, self._side[stage.hand])[:2] - goal[:2]
    else:
      return None
    length = float(np.linalg.norm(onward))
    if length < 1e-9:
      return None
    return np.array([onward[0] / length, onward[1] / length, 0.0])

  def _arch(self, stage: Stage, start: Float64Array, goal: Float64Array) -> float:
    """How high a stage rides over the table between its ends."""
    if stage.name not in ARCHED:
      return 0.0
    if not stage.loaded:
      return self.rules.skim
    share, fixed = self.rules.lift
    flat = float(np.linalg.norm(goal[:2] - start[:2]))
    return float(min(share * flat + fixed, self.rules.arch))

  def _parked(self, hand: int) -> Float64Array:
    """Where a hand stands once it is done: off on its own side of the tape, a
    step back from it, and never twice in quite the same place."""
    rules = self.rules
    spot = home(self.room, self._side[hand])
    side = spot[:2] - self.room.tape
    turn = self.rng.uniform(-rules.stray, rules.stray)
    spin = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    bearing = side / float(np.linalg.norm(side))
    spot[:2] = self.room.tape + spin @ bearing * self.rng.uniform(*rules.park)
    return spot

  def _swept(self, cube: Float64Array) -> None:
    """Pick where this kick lands the cube, on the picking hand's side and at a
    different spot every cycle, and how hard to sweep to put it there."""
    rules = self.rules
    toward = home(self.room, self._side[PICK])[:2] - cube[:2]
    lie = toward / max(float(np.linalg.norm(toward)), 1e-9)
    scale = np.asarray(rules.patch)
    across, along = np.clip(self.rng.normal(0.0, scale), -2.0 * scale, 2.0 * scale)
    speed, carries = rules.kick
    spot = (carries + along) * lie + across * np.array([-lie[1], lie[0]])
    far = max(float(np.linalg.norm(spot)), 1e-9)
    self._bearing = np.array([spot[0] / far, spot[1] / far, 0.0])
    self._sweep = speed * far / carries

  def _placed(
    self, reached: Float64Array, facing: Float64Array, cube: Float64Array
  ) -> Float64Array:
    """Where to hold the hand for the cube to come down on the tape: across
    the table by the error of the cube it holds, at the grasp's height."""
    rest = np.array([*self.room.tape, self.room.cube.rest])
    over: Float64Array = reached + rest - cube
    over[2] = (rest - facing @ self.room.grasp)[2]
    return over


def scripted_policy(
  env: PickPlaceEnv,
  rng: np.random.Generator,
  cfg: ScriptedPolicyCfg | None = None,
  rules: Rules | None = None,
) -> ScriptedPolicy:
  """A scripted policy for the env, drawing from ``rng``; it reads the
  cube's position as its env state."""
  return ScriptedPolicy(
    Script(env.room, rng, rules),
    rng,
    env.cfg.clock.action_dt,
    env.rig.cfg.clock.n_action_steps,
    cfg,
  )
