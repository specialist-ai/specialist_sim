"""What one episode did, accumulated as it runs.

Every number here is a statement about the *environment*, so it is defined once
here rather than in whichever training repository happens to be scoring a
checkpoint this week. Two people reporting a success rate should be reporting the
same quantity, and the way to make that true is for neither of them to own the
definition.

Nothing in this module knows what a policy is. It is handed what a caller already
has -- the action it commanded and the five things ``step`` returned -- and it
holds one accumulator per world, because the environment is batched and an
episode's numbers belong to the world that ran it.

The definitions, in one place:

**Outcome.** ``success`` is the task latching, which is the cube back on the tape
at rest and let go of, having been off it. ``picked`` is a success the hand
*carried*: the cube came at least ``MetricsCfg.lift`` clear of the table and was
between the pads at some point, which a nudge across the tape is not.
``dropped`` is the cube off the table.

**Progress.** A success rate alone throws away most of what an episode showed:
a policy that carried the cube and set it down a centimetre wide scores the same
zero as one that never moved. ``stage`` is how far along :data:`STAGES` the
episode got -- reached, touched, lifted, carried, placed -- and ``progress`` is
that as a fraction. It is the longest achieved *prefix*, so it only ever goes up,
and where it stops is the failure mode: a run that stalls at ``reached`` cannot
grasp, one that stalls at ``carried`` cannot place. Partial credit is also what
makes an evaluation cheap: a graded score separates two policies in far fewer
episodes than a binary one, because most episodes stop being ties.
``closest_approach`` is the nearest a grasp point ever came to the cube, which
grades the episodes that do not even reach.

``stage`` is the *furthest* stage the episode got to, not the longest unbroken
run of them, so a completed episode never scores zero for having skipped a flag.
``stages`` is the separate question of which ones actually happened, and the two
are kept apart because in this task they genuinely differ: a cube can be knocked
into the air by the back of a gripper, so ``lifted`` without ``touched`` is a
real episode and not a bookkeeping error. Reading a touch out of a lift would be
reporting a grasp that did not occur.

**Behaviour.** ``lift`` is the highest the cube ever got above where it rests, in
metres, whether or not the episode succeeded. ``held_steps`` counts steps with a
pad on the cube and ``carried_steps`` those where it was also clear of the table.
``first_contact`` is the first step a pad touched it. ``path_length`` is how far
the hands travelled in total and ``smoothness`` how fluently, as a spectral arc
length -- two policies that succeed equally often can move nothing alike, and on
real hardware that is not the same policy.

**Command quality.** ``clipped`` is the *fraction* of commanded action
components that fell outside the action box: a policy commanding poses the rig
will not accept is a policy whose numbers mean less than they look. A fraction
rather than a count, because a count scales with the horizon and the hand count,
and a single-armed rig would otherwise look half as bad as a two-armed one for
the same behaviour. ``tracking`` and
``tracking_rotation`` are how far the worst hand ended from the last pose the
control step actually executed -- the rig's own fidelity, not the policy's.
**Sensing.** ``image_age`` is the greatest number of rows any delivered frame ran
behind the boundary it arrived at, so a run that quietly saw staler images than
it was configured for says so.

**Physics.** ``penetration`` is the deepest two bodies overlapped at any step, in
metres. It says nothing about the policy and everything about whether the numbers
above are worth reading: a solver being forced through a wall is not a result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from specialist_sim import pose as se3
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.metrics.smoothness import sparc
from specialist_sim.spaces import ACTION_HIGH, ACTION_LOW, F_STEP, GRIPPERS, PER_HAND

if TYPE_CHECKING:
  from specialist_sim.env import PickPlaceEnv

__all__ = ["STAGES", "Episode", "EpisodeReport", "MetricsCfg"]

STAGES = ("reached", "touched", "lifted", "carried", "placed")
"""What an episode has to do, in the order it has to do it.

How far along this an episode got is its partial credit, and where it stopped is
its failure mode. The order is the physical one: a pad cannot touch the cube
before a hand is near it, and the task cannot latch before the cube has been
somewhere else.
"""


@dataclass(frozen=True)
class MetricsCfg:
  lift: float = 0.010
  """How far clear of its resting height the cube must come for a success to
  count as a pick rather than a nudge, in metres."""
  reach: float = 0.05
  """How near a grasp point must come to the cube to count as having reached it,
  in metres. Loose on purpose: it grades a policy that is aiming at the cube
  against one that is not, and is not a claim about grasp quality."""


@dataclass(frozen=True)
class EpisodeReport:
  """One world's episode, scored. Every field is defined in the module docstring."""

  seed: int | None
  steps: int
  ended: str
  """How it finished: ``success``, ``dropped``, ``truncated`` when the
  environment's own horizon ran out, ``stopped`` when the caller ended it, and
  ``running`` if it has not finished at all."""
  success: bool
  picked: bool
  dropped: bool
  stage: int
  """The furthest of :data:`STAGES` the episode got to. 0 to ``len(STAGES)``."""
  progress: float
  """The same, as a fraction: partial credit for an episode that did not win."""
  stages: tuple[str, ...]
  """Which of :data:`STAGES` actually happened, in order. Not implied by
  ``stage``: a cube knocked off the table was lifted without being touched."""
  reward: float
  steps_to_success: int
  """-1 if it never succeeded."""
  lift: float
  closest_approach: float
  """The nearest a grasp point came to the cube, in metres."""
  held_steps: int
  carried_steps: int
  first_contact: int
  """-1 if a pad never touched the cube."""
  path_length: float
  """How far the hands travelled in total, in metres."""
  smoothness: float
  """Spectral arc length of the hands' speed. Negative; less negative is
  smoother. NaN for an episode too short, or one that never moved."""
  penetration: float
  """The deepest two bodies overlapped, in metres. A physics-health number."""
  clipped: float
  tracking: float
  tracking_rotation: float
  image_age: int

  def as_dict(self) -> dict[str, Any]:
    """The report as plain values, for a logger that wants keys."""
    return asdict(self)


@dataclass
class _Running:
  """One world's counters while its episode is still going."""

  seed: int | None = None
  steps: int = 0
  reward: float = 0.0
  succeeded_at: int = -1
  dropped: bool = False
  ended: str = "running"
  lift: float = 0.0
  closest_approach: float = float("inf")
  held_steps: int = 0
  carried_steps: int = 0
  first_contact: int = -1
  clipped: int = 0
  commanded: int = 0
  """Action components commanded in total, which `clipped` is a fraction of."""
  image_age: int = 0
  penetration: float = 0.0
  reached: set[str] = field(default_factory=set)
  """Which of `STAGES` have happened at all, in any order."""
  path: list[Float64Array] = field(default_factory=list)
  """Every hand's position on every row, `(hands, rows, 3)` a step at a time."""
  tracking: list[float] = field(default_factory=list)
  tracking_rotation: list[float] = field(default_factory=list)


class Episode:
  """One accumulator per world, fed the same values a caller already has.

  Args:
    env: The environment being scored. Read for its hand count, its clock and
      its tracker, never stepped.
    cfg: Thresholds the definitions need.
  """

  def __init__(self, env: PickPlaceEnv, cfg: MetricsCfg | None = None):
    self.env, self.cfg = env, cfg or MetricsCfg()
    self.slots = GRIPPERS
    self.hands = len(self.slots)
    self.worlds = env.worlds
    self._low, self._high = ACTION_LOW, ACTION_HIGH
    self._rest = env.scene.cube.rest
    self._grasp = env.scene.grasp
    """Where the pads meet, in a hand's own frame, so a grasp point can be
    walked to from the pose an observation reports."""
    self._running = [_Running() for _ in range(self.worlds)]
    self._done: list[EpisodeReport | None] = [None] * self.worlds

  def begin(
    self, worlds: list[int] | None = None, seeds: list[int] | None = None
  ) -> None:
    """Start scoring a fresh episode in every world, or only the ones named.

    Call it after ``reset``. Whatever those worlds had accumulated is dropped, so
    a caller that wants the finished report takes it with :meth:`report` first.
    """
    chosen = list(range(self.worlds)) if worlds is None else list(worlds)
    if seeds is not None and len(seeds) != len(chosen):
      raise ValueError(f"{len(chosen)} worlds want that many seeds, not {len(seeds)}")
    for index, world in enumerate(chosen):
      self._running[world] = _Running(seed=None if seeds is None else int(seeds[index]))
      self._done[world] = None

  def record(
    self,
    action: Float64Array,
    observation: dict[str, Any],
    reward: Float64Array,
    terminated: BoolArray,
    truncated: BoolArray,
    info: dict[str, Any],
  ) -> None:
    """Take one control step's worth of every world, exactly as ``step`` gave it.

    A world whose episode has already ended is left alone, so a batch in which
    one world finished early keeps that world's report intact while the rest run
    on.
    """
    commanded = self._commanded(action)
    executed = commanded[:, self.env.hands.tracked - 1]
    outside = np.count_nonzero(
      (commanded < self._low) | (commanded > self._high), axis=(1, 2)
    )
    for world in range(self.worlds):
      one = self._running[world]
      if one.ended != "running":
        continue
      one.steps += 1
      one.reward += float(reward[world])
      one.clipped += int(outside[world])
      one.commanded += int(self._low.size)
      held = bool(info["held"][world])
      cube = np.asarray(info["cube_pos"][world], np.float64)
      lift = float(cube[2]) - self._rest
      one.lift = max(one.lift, lift)
      one.held_steps += held
      one.carried_steps += held and lift > self.cfg.lift
      if held and one.first_contact < 0:
        one.first_contact = one.steps
      succeeded = bool(info["is_success"][world])
      if succeeded and one.succeeded_at < 0:
        one.succeeded_at = one.steps
      one.dropped |= bool(info["dropped"][world])
      near, path = self._near(observation, world, cube)
      one.closest_approach = min(one.closest_approach, near)
      one.path.append(path)
      one.penetration = max(one.penetration, self._deepest(world))
      if near <= self.cfg.reach:
        one.reached.add("reached")
      if held:
        one.reached.add("touched")
      if lift > self.cfg.lift:
        one.reached.add("lifted")
      if held and lift > self.cfg.lift:
        one.reached.add("carried")
      if succeeded:
        one.reached.add("placed")
      gap, spin = self._gap(observation, executed, world)
      one.tracking.append(gap)
      one.tracking_rotation.append(spin)
      one.image_age = max(one.image_age, self._age(observation, world))
      if bool(terminated[world]):
        one.ended = "success" if one.succeeded_at >= 0 else "dropped"
      elif bool(truncated[world]):
        one.ended = "truncated"
      if one.ended != "running":
        self._done[world] = self._scored(one)

  def stop(self, worlds: list[int] | None = None, reason: str = "stopped") -> None:
    """End any episode still running, because the caller is done with it.

    An environment ends an episode on the task or on its own horizon. A caller
    often has a rule of its own -- a step budget, a wall clock, an operator
    calling a halt -- and an episode ended that way is finished data, not a
    half-finished episode. Without this it would stay ``running`` and
    :func:`specialist_sim.metrics.summary.summarise` would refuse the whole set.

    Args:
      worlds: Which worlds to close out, or every one still running.
      reason: What to record in ``ended``. Anything but the environment's own
        outcomes, which are not the caller's to claim.
    """
    if reason in ("success", "dropped", "truncated", "running"):
      raise ValueError(f"{reason} is the environment's to say, not the caller's")
    for world in range(self.worlds) if worlds is None else worlds:
      one = self._running[world]
      if one.ended == "running":
        one.ended = reason
        self._done[world] = self._scored(one)

  def report(self, world: int = 0) -> EpisodeReport:
    """One world's episode, finished or as it stands."""
    done = self._done[world]
    return done if done is not None else self._scored(self._running[world])

  def reports(self) -> list[EpisodeReport]:
    """Every world's, in world order."""
    return [self.report(world) for world in range(self.worlds)]

  @property
  def running(self) -> BoolArray:
    """Which worlds are still mid-episode."""
    return np.array([one.ended == "running" for one in self._running], np.bool_)

  def _scored(self, one: _Running) -> EpisodeReport:
    success = one.succeeded_at >= 0
    stage = max(
      (index + 1 for index, name in enumerate(STAGES) if name in one.reached),
      default=0,
    )
    travelled, fluency = self._travelled(one)
    return EpisodeReport(
      seed=one.seed,
      steps=one.steps,
      ended=one.ended,
      success=success,
      picked=success and one.lift > self.cfg.lift and one.held_steps > 0,
      dropped=one.dropped,
      stage=stage,
      progress=stage / len(STAGES),
      stages=tuple(name for name in STAGES if name in one.reached),
      reward=one.reward,
      steps_to_success=one.succeeded_at,
      lift=one.lift,
      closest_approach=(
        one.closest_approach if np.isfinite(one.closest_approach) else float("nan")
      ),
      held_steps=one.held_steps,
      carried_steps=one.carried_steps,
      first_contact=one.first_contact,
      path_length=travelled,
      smoothness=fluency,
      penetration=one.penetration,
      clipped=one.clipped / one.commanded if one.commanded else float("nan"),
      tracking=float(np.median(one.tracking)) if one.tracking else float("nan"),
      tracking_rotation=(
        float(np.median(one.tracking_rotation))
        if one.tracking_rotation
        else float("nan")
      ),
      image_age=one.image_age,
    )

  def _travelled(self, one: _Running) -> tuple[float, float]:
    """How far the hands went, and how fluently, over the whole episode.

    The rows of every step are laid end to end first, so a movement that crosses
    a control-step boundary is one movement. Smoothness is the worst hand's,
    because a rig is as juddery as its juddering half.
    """
    if not one.path:
      return (0.0, float("nan"))
    path = np.concatenate(one.path, axis=1)
    legs = np.linalg.norm(np.diff(path, axis=1), axis=2)
    fluency = [sparc(speed * F_STEP, F_STEP) for speed in legs]
    kept = [value for value in fluency if not np.isnan(value)]
    return (float(legs.sum()), min(kept) if kept else float("nan"))

  def _near(
    self, observation: dict[str, Any], world: int, cube: Float64Array
  ) -> tuple[float, Float64Array]:
    """The nearest a grasp point came to the cube this step, and where the hands went.

    A grasp point is where the pads meet, not where the gripper base is, so the
    distance means the same thing on a hand of any size.
    """
    nearest = float("inf")
    path = np.empty((self.hands, len(observation["step"][world]), 3))
    for hand, slot in enumerate(self.slots):
      rows = np.asarray(observation[slot]["pose"][world], np.float64)
      path[hand] = rows[:, :3]
      for row in rows:
        grasp = row[:3] + se3.rotation(row[3:7]) @ self._grasp
        nearest = min(nearest, float(np.linalg.norm(grasp - cube)))
    return nearest, path

  def _deepest(self, world: int) -> float:
    """How far two bodies overlap, at the boundary this step returned at."""
    data = self.env.datas[world]
    if not data.ncon:
      return 0.0
    return float(max(0.0, -data.contact.dist[: data.ncon].min()))

  def _commanded(self, action: Float64Array) -> Float64Array:
    """The action with its leading world axis, however the caller passed it."""
    trajectory = np.asarray(action, np.float64)
    if self.worlds == 1 and trajectory.shape == self._low.shape:
      trajectory = trajectory[None]
    if trajectory.shape != (self.worlds, *self._low.shape):
      raise ValueError(
        f"an action has shape {(self.worlds, *self._low.shape)}, not {trajectory.shape}"
      )
    return trajectory

  def _gap(
    self, observation: dict[str, Any], executed: Float64Array, world: int
  ) -> tuple[float, float]:
    """How far the worst hand ended from the pose the step actually executed."""
    wanted = executed[world].reshape(self.hands, PER_HAND)
    apart, spun = 0.0, 0.0
    for hand, slot in enumerate(self.slots):
      pose = np.asarray(observation[slot]["pose"][world, -1], np.float64)
      apart = max(apart, float(np.linalg.norm(pose[:3] - wanted[hand, :3])))
      here = se3.rotation(pose[3:7])
      there = se3.rotation(wanted[hand, 3:7] / np.linalg.norm(wanted[hand, 3:7]))
      spun = max(spun, float(np.linalg.norm(se3.log(there @ here.T))))
    return apart, spun

  def _age(self, observation: dict[str, Any], world: int) -> int:
    """How far behind the boundary the oldest delivered frame was captured."""
    boundary = int(observation["step"][world][-1])
    return max(
      boundary - int(observation[slot]["image_step"][world]) for slot in self.slots
    )
