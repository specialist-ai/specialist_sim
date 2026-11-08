"""``num_envs`` copies of a task's room with the hands in it, stepped together
on a persistent ``mujoco.batch.Batch``. Time is counted in rows, the lattice a
dataset records a hand at."""

import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import import_module
from typing import Any, ClassVar

import mujoco
import numpy as np
from mujoco.batch import Batch

from specialist_sim import gripper, rotation
from specialist_sim.arrays import (
  BoolArray,
  Float64Array,
  Int32Array,
  Int64Array,
  UInt8Array,
)
from specialist_sim.camera.lens import Lens, Mei
from specialist_sim.camera.render import Renderer
from specialist_sim.camera.shutter import Shutter
from specialist_sim.embodiment import Embodiment, Floating
from specialist_sim.episode import Rows
from specialist_sim.gripper import HANDS

PHYSICS_DT = 1.0 / 1200.0
IMPRATIO = 100.0
# Every colliding geom's contact. The SOLREF time constant must stay above twice
# PHYSICS_DT. A pad's steady sink scales with (1 - d) / d times that time
# constant squared, d the SOLIMP impedance.
SOLREF = (0.002, 1.0)
SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
# The warnings MuJoCo raises as it resets an env it cannot carry.
UNSTABLE = [
  int(mujoco.mjtWarning.mjWARN_BADQPOS),
  int(mujoco.mjtWarning.mjWARN_BADQVEL),
  int(mujoco.mjtWarning.mjWARN_BADQACC),
]
VEL = mujoco.mjtStage.mjSTAGE_VEL


@dataclass(frozen=True)
class Clock:
  per_row: int = 12
  """Physics steps in a row."""
  per_action: int = 2
  """Rows in one action of an action chunk."""
  per_step: int = 10
  """Rows in a control step, a whole number of actions."""

  @property
  def row_dt(self) -> float:
    return PHYSICS_DT * self.per_row

  @property
  def step_dt(self) -> float:
    return self.row_dt * self.per_step

  @property
  def n_action_steps(self) -> int:
    """Actions of a chunk that fall inside one control step."""
    return self.per_step // self.per_action


# One row a step, its one action the setpoint: for a hand that follows a person live.
ROWWISE = Clock(per_action=1, per_step=1)


@dataclass(frozen=True)
class Cfg:
  """What a room runs under."""

  clock: Clock = Clock()
  embodiment: type[Embodiment] = Floating
  lenses: Mapping[str, Lens] = field(default_factory=dict)
  """Each hand's lens; a hand left out carries ``Lens()``."""
  image_size: int = 360
  supersample: int = 2
  """Rays across each pixel, averaged: the robot's 720 px frame averaged to 360."""
  finger: tuple[float, float] = (116.0, 1.27)
  """The finger servo's stiffness and damping: the robot's."""
  camera_latency: float = 0.12
  """Seconds a frame takes to reach a policy once the shutter has fallen. The
  frame in hand is older still, by the time since the shutter last fell."""
  driven: tuple[str, ...] = HANDS
  """The hands the action names; every other hand holds where it was last sent."""
  image_hands: tuple[str, ...] = HANDS
  """The hands whose images the observation carries; with none, nothing is
  traced."""
  autoreset: bool = True


@dataclass(frozen=True)
class Lighting:
  """A room's key light, which casts shadow, and its fill, which does not, both
  directional. The defaults match the recordings."""

  key: tuple[float, float, float] = (0.282216, 0.188144, -0.940721)
  key_diffuse: float = 0.3120
  fill: tuple[float, float, float] = (-0.646997, -0.539164, -0.539164)
  fill_diffuse: float = 0.3000
  ambient: float = 0.6769

  def stand(self, spec: mujoco.MjSpec) -> None:
    far = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    spec.worldbody.add_light(
      type=far, dir=list(self.key), diffuse=[self.key_diffuse] * 3
    )
    spec.worldbody.add_light(
      type=far,
      dir=list(self.fill),
      diffuse=[self.fill_diffuse] * 3,
      ambient=[self.ambient] * 3,
      castshadow=False,
    )


@dataclass(frozen=True)
class Start:
  """What a caller fixes as an episode opens: one for every chosen env, or one
  each along a leading axis."""

  hands: Float64Array
  """Every hand's state, ``(hands, 8)``, in ``HANDS`` order."""
  objects: Float64Array | None = None
  """Every object's pose, ``(objects, 7)``; ``None`` has the task draw them
  clear of the hands."""
  phases: Float64Array | None = None
  """Where each lens stands in its period, ``(lenses,)`` in ``[0, 1)``;
  ``None`` draws them."""


@dataclass(frozen=True)
class Objects:
  """What a task judges: every env's objects, in body order."""

  poses: Float64Array
  """Every object's pose, ``(num_envs, objects, 7)``."""
  velocity: Float64Array
  """Every object's ``cvel``, ``(num_envs, objects, 6)``, angular then linear."""
  held: BoolArray
  """Whether a gripping pad touches each object, ``(num_envs, objects)``."""
  laid: Float64Array
  """Every object's pose as the episode was laid, ``(num_envs, objects, 7)``."""
  failed: BoolArray
  """Whether an object touches a geom ``Task.fails`` names, ``(num_envs,)``."""


@dataclass(frozen=True)
class Verdict:
  """What a task makes of every env, one entry per env."""

  progress: Float64Array
  """From 0 to 1; the goal holds right now where it is 1."""
  failed: BoolArray
  """Where the task is lost, which ends the episode."""
  labels: dict[str, BoolArray] = field(default_factory=dict)
  """Whatever else the task tells about each env, by its own names."""
  counts: dict[str, Int64Array] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
  """What a policy sees, one entry per env."""

  row: Int64Array
  """The rows the step spanned, ``(num_envs, rows)``."""
  state: Float64Array
  """The driven hands at each row, ``(num_envs, rows, driven, 8)``, in
  ``Cfg.driven`` order."""
  image: UInt8Array
  """Each lens's newest frame, ``(num_envs, lenses, size, size, 3)``, black
  outside the disc. The lenses are in ``Cfg.image_hands`` order, as in
  ``body`` and ``capture_row``."""
  body: Int32Array
  """The highest-numbered body each pixel's rays struck, ``(num_envs, lenses,
  size, size)``, ``-1`` for none."""
  capture_row: Int64Array
  """The row each frame is filed under, ``(num_envs, lenses)``."""


@dataclass(frozen=True)
class Task:
  """The lit empty room. A task is a frozen dataclass whose fields an episode
  records."""

  lighting: ClassVar[Lighting] = Lighting()
  dwell: ClassVar[float] = 0.1
  """Seconds the goal must hold together before success latches. The env
  counts it in whole control steps, at least one."""
  latched: ClassVar[tuple[str, ...]] = ()
  """Labels that mean "this happened", which the env holds once they do."""
  fails: ClassVar[Mapping[str, tuple[str, ...]]] = {}
  """By object body, the geom name prefixes ``Objects.failed`` reports it touching."""
  focus: ClassVar[Float64Array] = np.zeros(3)
  """The room point an overview looks at."""

  @property
  def settle(self) -> float:
    """Seconds the room runs before an episode opens, so what was laid comes
    to rest."""
    return 0.0

  def spec(self) -> mujoco.MjSpec:
    """The room, unlit, that the hands are attached into."""
    return mujoco.MjSpec()

  def draw(
    self, rng: np.random.Generator, driven: tuple[str, ...]
  ) -> tuple[Float64Array, Float64Array]:
    """One episode's start: every hand's state, ``(hands, 8)`` in ``HANDS``
    order, and every object's pose, ``(objects, 7)``."""
    raise NotImplementedError

  def judge(self, objects: Objects) -> Verdict:
    """Where every env stands. A reset judges every env, including ones
    mid-episode, so this reads the objects and advances nothing."""
    num_envs = len(objects.poses)
    return Verdict(np.zeros(num_envs), np.zeros(num_envs, np.bool_))


class Env[T: Task]:
  """``num_envs`` copies of a task's room, reset and stepped together. The
  objects are every body on a free joint that is not part of a hand, in body
  order."""

  def __init__(self, task: T, cfg: Cfg = Cfg(), num_envs: int = 1) -> None:
    self.task = task
    self.cfg = cfg
    self.num_envs = num_envs
    clock = cfg.clock
    spec = task.spec()
    task.lighting.stand(spec)
    lenses = {hand: cfg.lenses.get(hand, Lens()) for hand in HANDS}
    for hand in HANDS:
      gripper.attach(
        spec, cfg.embodiment.attach(spec, hand), hand, lenses[hand], cfg.finger
      )
    # The batch copies the model, so a later write to it never reaches the physics.
    model = spec.compile()
    self.model = model
    self.hands = tuple(gripper.bind(model, hand) for hand in HANDS)
    self._driven = [HANDS.index(hand) for hand in cfg.driven]
    self._servos = np.array([hand.servo for hand in self.hands])
    roots = model.body_rootid[[hand.base for hand in self.hands]]
    free = model.jnt_type[model.body_jntadr] == mujoco.mjtJoint.mjJNT_FREE
    self._object_bodies = np.flatnonzero(free & ~np.isin(model.body_rootid, roots))
    at = model.jnt_qposadr[model.body_jntadr[self._object_bodies]]
    self._object_q = at[:, None] + np.arange(7)
    of = np.full(model.nbody, -1)
    of[self._object_bodies] = np.arange(len(self._object_bodies))
    geom_root = model.body_rootid[model.geom_bodyid]
    # The object each geom belongs to, -1 for none.
    self._object_of = of[geom_root]
    self._hand_geom = np.isin(geom_root, roots)
    self._pad = np.zeros(model.ngeom, np.bool_)
    self._pad[[pad for hand in self.hands for pad in hand.pads]] = True
    # Whether a contact between two geoms counts in Objects.failed.
    self._ends = np.zeros((model.ngeom, model.ngeom), np.bool_)
    for body, fails in task.fails.items():
      ends = [model.geom(g).name.startswith(fails) for g in range(model.ngeom)]
      self._ends[np.ix_(model.geom_bodyid == model.body(body).id, ends)] = True
    _physics(model)
    self.embodiment = cfg.embodiment(model)
    seen = [self.hands[HANDS.index(hand)] for hand in cfg.image_hands]
    views = [(hand.lens_site, lenses[hand.name], hand.housing) for hand in seen]
    self._renderer = (
      Renderer(model, views, cfg.image_size, cfg.supersample) if seen else None
    )
    self._shutter = Shutter(
      num_envs,
      len(seen),
      PHYSICS_DT,
      clock.per_row,
      clock.per_step,
      cfg.camera_latency,
    )
    self.sim = Batch(
      model, num_envs, nthread=min(num_envs, os.cpu_count() or 1), persistent=True
    )
    rows = clock.per_step
    hands = len(HANDS)
    self._qpos_rows = np.zeros((num_envs, rows, model.nq))
    # Every env's hands at each row of the last step, as they stood and their
    # setpoint at the row's end.
    self._state_rows = np.zeros((num_envs, rows, hands, 8))
    self._setpoint_rows = np.zeros((num_envs, rows, hands, 8))
    # Where every hand was last sent.
    self._last = np.zeros((num_envs, hands, 8))
    self._laid = np.zeros((num_envs, len(self._object_bodies), 7))
    # The seed of the layout each env laid.
    self._seed = np.zeros(num_envs, np.int64)
    self._row = np.zeros(num_envs, np.int64)
    self._unstable = np.zeros(num_envs, np.bool_)
    self._tracking = np.zeros(num_envs)
    # Control steps the goal must hold together, and each env's count of them.
    self._dwell = max(1, round(task.dwell / clock.step_dt))
    self._streak = np.zeros(num_envs, np.int64)
    self._success = np.zeros(num_envs, np.bool_)
    self._done = np.zeros(num_envs, np.bool_)
    self._marks = {name: np.zeros(num_envs, np.bool_) for name in task.latched}
    self._scratch = mujoco.MjData(model)

  def reset(
    self,
    seed: Sequence[int],
    env_ids: Sequence[int] | None = None,
    start: Start | None = None,
  ) -> tuple[Observation, dict[str, Any]]:
    """Open an episode in every env, or in the named ones, ascending, each
    laid from the seed beside it or where ``start`` says. Hands given without
    objects pass over each seed whose objects would start in a hand."""
    chosen = list(range(self.num_envs)) if env_ids is None else list(env_ids)
    n = len(chosen)
    self.sim.reset(chosen, stage=mujoco.mjtStage.mjSTAGE_NONE)
    phases = np.zeros((n, len(self.cfg.image_hands)))
    for k, (env, first) in enumerate(zip(chosen, seed, strict=True)):
      laid = first
      rng = np.random.default_rng(laid)
      if start is None:
        hands, objects = self.task.draw(rng, self.cfg.driven)
      else:
        hands = np.broadcast_to(start.hands, (n, len(HANDS), 8))[k]
        if start.objects is not None:
          objects = np.broadcast_to(start.objects, (n, *self._laid.shape[1:]))[k]
        else:
          objects = self.task.draw(rng, self.cfg.driven)[1]
          # Up to 49 more seeds before the last is kept.
          for tries in range(1, 50):
            self._lay(env, hands, objects)
            self.sim.forward([env], stage=mujoco.mjtStage.mjSTAGE_POS)
            if not self._touching(env):
              break
            laid = first + tries
            rng = np.random.default_rng(laid)
            objects = self.task.draw(rng, self.cfg.driven)[1]
      self._lay(env, hands, objects)
      self._laid[env] = objects
      self._seed[env] = laid
      if start is not None and start.phases is not None:
        phases[k] = np.broadcast_to(start.phases, phases.shape)[k]
      else:
        drawn = rng.uniform(size=len(HANDS))
        phases[k] = [drawn[HANDS.index(hand)] for hand in self.cfg.image_hands]
    self.sim.forward(chosen, stage=VEL)
    if self.task.settle:
      self._settle(chosen, round(self.task.settle / self.cfg.clock.row_dt))
    states = gripper.states(self.hands, self.sim.qpos, self.sim.xpos, self.sim.xquat)[
      chosen
    ]
    self._last[chosen] = states
    self._state_rows[chosen] = states[:, None]
    self._setpoint_rows[chosen] = states[:, None]
    self._qpos_rows[chosen] = self.sim.qpos[chosen, None]
    for kept in (self._row, self._streak, self._tracking, self._success):
      kept[chosen] = 0
    self._unstable[chosen] = self.sim.warning[chosen][:, UNSTABLE, 1].sum(-1) > 0
    for mark in self._marks.values():
      mark[chosen] = False
    if self._renderer is not None:
      self._shutter.open(
        chosen, phases, *self._render(chosen, range(len(self.cfg.image_hands)))
      )
    return self._observe(), self._info(self.task.judge(self.objects()))

  def step(
    self, action: Float64Array | None = None
  ) -> tuple[Observation, dict[str, Any]]:
    """Track the leading actions of every env's action chunk, ``(num_envs,
    chunk_size, driven, 8)``, and judge where that leaves each env. With no
    action every hand holds where it was last sent. Under ``Cfg.autoreset``,
    an env whose episode ended first opens seed + 1 under its standing hands."""
    if ended := np.flatnonzero(self._done & self.cfg.autoreset).tolist():
      self.reset([*self._seed[ended] + 1], ended, Start(self._state_rows[ended, -1]))
    clock = self.cfg.clock
    full = hold(self._last, clock.n_action_steps)
    if action is not None:
      full[:, :, self._driven] = action[:, : clock.n_action_steps]
    setpoint, twist = interpolate(self._last, full, clock)
    for row in range(clock.per_step):
      self._command(setpoint[:, row], twist[:, row])
      self._advance((self._row + row) * clock.per_row)
      self._qpos_rows[:, row] = self.sim.qpos
      self._state_rows[:, row] = gripper.states(
        self.hands, self.sim.qpos, self.sim.xpos, self.sim.xquat
      )
    self._setpoint_rows = setpoint
    self._last = setpoint[:, -1].copy()
    self._row += clock.per_step
    self._unstable = self.sim.warning[:, UNSTABLE, 1].sum(-1) > 0
    apart = self._state_rows[:, -1, self._driven, :3] - self._last[:, self._driven, :3]
    self._tracking = np.linalg.norm(apart, axis=-1).max(axis=1)
    verdict = self.task.judge(self.objects())
    self._streak = np.where(
      (verdict.progress >= 1.0) & ~self._unstable, self._streak + 1, 0
    )
    self._success |= self._streak >= self._dwell
    return self._observe(), self._info(verdict)

  def objects(self) -> Objects:
    held = np.zeros((self.num_envs, len(self._object_bodies)), np.bool_)
    failed = np.zeros(self.num_envs, np.bool_)
    for env in np.flatnonzero(self.sim.ncon).tolist():
      one, other = self.sim.get_data(env, self._scratch).contact.geom.T
      touched = np.concatenate(
        [self._object_of[other][self._pad[one]], self._object_of[one][self._pad[other]]]
      )
      held[env, touched[touched >= 0]] = True
      failed[env] = self._ends[one, other].any() | self._ends[other, one].any()
    return Objects(
      poses=np.concatenate(
        [self.sim.xpos[:, self._object_bodies], self.sim.xquat[:, self._object_bodies]],
        axis=-1,
      ),
      velocity=self.sim.cvel[:, self._object_bodies].copy(),
      held=held,
      laid=self._laid.copy(),
      failed=failed,
    )

  def rows(self, env_idx: int) -> Rows:
    """One env's rows of the last step."""
    clock = self.cfg.clock
    return Rows(
      timestamp=(self._row[env_idx] + np.arange(1 - clock.per_step, 1)) * clock.row_dt,
      state=self._state_rows[env_idx].copy(),
      setpoint=self._setpoint_rows[env_idx].copy(),
      qpos=self._qpos_rows[env_idx].copy(),
    )

  def trace(self, qpos: Float64Array) -> tuple[UInt8Array, Int32Array]:
    """Every lens's view of the room at each of ``qpos``, ``(m, nq)``: images
    ``(m, lenses, size, size, 3)`` and the highest-numbered body each pixel's rays
    struck."""
    scratch = self._scratch
    m = len(qpos)
    geom_xpos = np.empty((m, self.model.ngeom, 3))
    geom_xmat = np.empty((m, self.model.ngeom, 9))
    site_xpos = np.empty((m, self.model.nsite, 3))
    site_xmat = np.empty((m, self.model.nsite, 9))
    for k, q in enumerate(qpos):
      scratch.qpos[:] = q
      mujoco.mj_kinematics(self.model, scratch)
      geom_xpos[k] = scratch.geom_xpos
      geom_xmat[k] = scratch.geom_xmat
      site_xpos[k] = scratch.site_xpos
      site_xmat[k] = scratch.site_xmat
    assert self._renderer is not None
    out: tuple[UInt8Array, Int32Array] = self._renderer.render(
      geom_xpos, geom_xmat, site_xpos, site_xmat, range(len(self.cfg.image_hands))
    )
    return out

  def describe(self) -> dict[str, Any]:
    """What this env is, as JSON, which ``make`` stands again."""
    return {
      "task": _named(type(self.task)),
      "fields": asdict(self.task),
      "cfg": {**asdict(self.cfg), "embodiment": _named(self.cfg.embodiment)},
    }

  def _lay(self, env: int, hands: Float64Array, objects: Float64Array) -> None:
    """Write one env's hands, ``(hands, 8)``, and objects, ``(objects, 7)``."""
    qpos = self.sim.qpos[env]
    for index, (hand, state) in enumerate(zip(self.hands, hands, strict=True)):
      self.embodiment.place(qpos, index, state[:7])
      qpos[list(hand.fingers)] = state[7], -state[7]
    qpos[self._object_q] = objects

  def _touching(self, env: int) -> bool:
    """Whether a hand touches an object in one env, as its last call left it."""
    one, other = self.sim.get_data(env, self._scratch).contact.geom.T
    objects = self._object_of >= 0
    touching = (
      objects[one] & self._hand_geom[other] | objects[other] & self._hand_geom[one]
    )
    return bool(touching.any())

  def _settle(self, env_ids: Sequence[int], rows: int) -> None:
    """Run the named envs alone for ``rows`` rows, every hand held where it
    stands. The other envs stand still."""
    held = gripper.states(self.hands, self.sim.qpos, self.sim.xpos, self.sim.xquat)
    still = np.zeros((self.num_envs, len(HANDS), 6))
    for _ in range(rows):
      self._command(held, still)
      self.sim.step(env_ids, nstep=self.cfg.clock.per_row, stage=VEL)

  def _command(self, setpoint: Float64Array, twist: Float64Array) -> None:
    """Send every env's hands toward ``setpoint``, ``(num_envs, hands, 8)``, at
    ``twist``, ``(num_envs, hands, 6)``, from where each env stands."""
    self.embodiment.command(setpoint[..., :7], twist, self.cfg.clock.row_dt, self.sim)
    gripper.grip(self.sim.ctrl, self._servos, setpoint[..., 7])

  def _advance(self, elapsed: Int64Array) -> None:
    """Run every env one row, which starts ``elapsed``, ``(num_envs,)``,
    physics steps into its episode, stopping where a lens exposes to trace it."""
    done = 0
    for offset, firing in self._shutter.shots(elapsed):
      self.sim.step(nstep=offset - done, stage=VEL)
      done = offset
      for lens, env_ids in firing:
        image, body = self._render(env_ids, [lens])
        self._shutter.keep(
          lens, env_ids, elapsed[env_ids] + offset, image[:, 0], body[:, 0]
        )
    if done < self.cfg.clock.per_row:
      self.sim.step(nstep=self.cfg.clock.per_row - done, stage=VEL)

  def _render(
    self, env_ids: Sequence[int], which: Sequence[int]
  ) -> tuple[UInt8Array, Int32Array]:
    """What the lenses ``which`` indexes see in the named envs as they stand."""
    sim = self.sim
    assert self._renderer is not None
    out: tuple[UInt8Array, Int32Array] = self._renderer.render(
      sim.geom_xpos[env_ids],
      sim.geom_xmat[env_ids],
      sim.site_xpos[env_ids],
      sim.site_xmat[env_ids],
      which,
    )
    return out

  def _observe(self) -> Observation:
    cfg = self.cfg
    image, body, capture_row = self._shutter.deliver(self._row)
    framed = (self.num_envs, len(cfg.image_hands), cfg.image_size, cfg.image_size)
    return Observation(
      row=self._row[:, None] + np.arange(1 - cfg.clock.per_step, 1),
      state=self._state_rows[:, :, self._driven].copy(),
      image=image.reshape(*framed, 3),
      body=body.reshape(framed),
      capture_row=capture_row,
    )

  def _info(self, verdict: Verdict) -> dict[str, Any]:
    """How every env's episode stands, one ``(num_envs,)`` array a key."""
    labels = dict(verdict.labels)
    for name, mark in self._marks.items():
      mark |= labels[name]
      labels[name] = mark.copy()
    self._done = self._success | verdict.failed | self._unstable
    return {
      "labels": labels,
      "counts": dict(verdict.counts),
      "progress": verdict.progress,
      "success": self._success.copy(),
      "unstable": self._unstable.copy(),
      "tracking_error": self._tracking.copy(),
      "done": self._done.copy(),
      "seed": self._seed.copy(),
    }


def make(meta: dict[str, Any], **cfg: Any) -> Env[Task]:
  """The env ``meta``, an ``Env.describe()``, names, stood again, each keyword
  replacing a recorded ``Cfg`` field."""
  recorded = meta["cfg"]
  decoded = {
    **_tupled(recorded),
    "clock": Clock(**recorded["clock"]),
    "embodiment": _resolve(recorded["embodiment"]),
    "lenses": {
      hand: Lens(**{**_tupled(lens), "params": Mei(**lens["params"])})
      for hand, lens in recorded["lenses"].items()
    },
  }
  task = _resolve(meta["task"])(**_tupled(meta["fields"]))
  return Env(task, Cfg(**{**decoded, **cfg}))


def interpolate(
  last: Float64Array, action: Float64Array, clock: Clock
) -> tuple[Float64Array, Float64Array]:
  """The executed actions, ``(num_envs, n_action_steps, hands, 8)``, laid
  from ``last``, ``(num_envs, hands, 8)``, onto the rows of a control step and
  sampled at each row's end: every row's setpoint, ``(num_envs, rows, hands,
  8)``, and its twist, ``(num_envs, rows, hands, 6)``, linear then angular in
  the room frame."""
  num_envs, actions, hands, _ = action.shape
  knots = np.concatenate([last[:, None], action], axis=1)
  ends = knots[..., [0, 1, 2, 7]]
  quaternion = knots[..., 3:7] / np.linalg.norm(knots[..., 3:7], axis=-1, keepdims=True)
  step = ends[:, 1:] - ends[:, :-1]
  spin = rotation.log(
    rotation.multiply(rotation.conjugate(quaternion[:, :-1]), quaternion[:, 1:])
  )
  blend = (np.arange(1, clock.per_action + 1) / clock.per_action)[:, None, None]
  moved = ends[:, :-1, None] + blend * step[:, :, None]
  turned = rotation.multiply(
    quaternion[:, :-1, None], rotation.exp(blend * spin[:, :, None])
  )
  rows = (num_envs, actions * clock.per_action, hands)
  setpoint = np.concatenate([moved[..., :3], turned, moved[..., 3:]], axis=-1)
  action_dt = clock.row_dt * clock.per_action
  velocity = np.concatenate([step[..., :3], spin], axis=-1)[:, :, None] / action_dt
  twist = np.broadcast_to(velocity, (*moved.shape[:-1], 6)).reshape(*rows, 6)
  setpoint = setpoint.reshape(*rows, 8)
  angular = rotation.rotate(setpoint[..., 3:7], twist[..., 3:])
  return setpoint, np.concatenate([twist[..., :3], angular], axis=-1)


def hold(state: Float64Array, chunk_size: int) -> Float64Array:
  """An action chunk that asks for one state, ``(..., hands, 8)``, at every
  action: ``(..., chunk_size, hands, 8)``."""
  return np.repeat(np.expand_dims(state, -3), chunk_size, axis=-3)


def _tupled(knobs: dict[str, Any]) -> dict[str, Any]:
  return {k: tuple(v) if isinstance(v, list) else v for k, v in knobs.items()}


def _named(kind: type) -> str:
  return f"{kind.__module__}:{kind.__qualname__}"


def _resolve(name: str) -> Any:
  module, _, attribute = name.partition(":")
  return getattr(import_module(module), attribute)


def _physics(model: mujoco.MjModel) -> None:
  """Set ``model``'s physics in place. A gripping pad takes priority, so a pad
  against anything is the pad's contact."""
  model.opt.timestep = PHYSICS_DT
  model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  model.opt.impratio = IMPRATIO
  colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
  model.geom_solref[colliding] = SOLREF
  model.geom_solimp[colliding] = SOLIMP
  model.geom_priority[model.geom_condim == gripper.GRIPPING] = 1
