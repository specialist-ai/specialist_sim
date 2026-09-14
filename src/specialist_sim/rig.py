"""The rig: hands on their carriers standing in a world, stepped together
across many worlds. Time is counted in rows, the lattice a dataset records a
hand at.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import mujoco
import numpy as np
from mujoco import rollout

from specialist_sim import gripper
from specialist_sim.arrays import (
  Float64Array,
  Int32Array,
  Int64Array,
  UInt8Array,
)
from specialist_sim.camera import NOMINAL, Camera
from specialist_sim.embodiment import Embodiment, Floating
from specialist_sim.gripper import SLOTS
from specialist_sim.render import Renderer
from specialist_sim.trajectory import Ramp, hold

__all__ = ["Cfg", "Clock", "Frame", "Observation", "Rig"]


@dataclass(frozen=True)
class Clock:
  physics_dt: float = 1.0 / 1200.0
  per_row: int = 12
  """Physics steps in a row."""
  per_pose: int = 2
  """Rows in one pose of a trajectory."""
  per_step: int = 10
  """Rows in a control step, a whole number of poses."""

  @property
  def row_dt(self) -> float:
    return self.physics_dt * self.per_row

  @property
  def pose_dt(self) -> float:
    return self.row_dt * self.per_pose

  @property
  def step_dt(self) -> float:
    return self.row_dt * self.per_step

  @property
  def tracked(self) -> int:
    """Poses of a trajectory that fall inside one control step."""
    return self.per_step // self.per_pose


@dataclass(frozen=True)
class Cfg:
  world: Callable[[], mujoco.MjSpec]
  """What the hands stand in."""
  embodiment: Embodiment = field(default_factory=Floating)
  clock: Clock = Clock()
  cameras: dict[str, Camera] = field(default_factory=dict)
  """The lens each hand carries, by slot; a slot left out carries the nominal one."""
  image_size: int = 240
  driven: tuple[str, ...] = SLOTS
  """The hands the action names; every other hand holds where it was last sent."""
  seen: tuple[str, ...] = ()
  """Slots whose lens the observation carries beyond the driven hands' own."""
  image_lag: float = 0.0
  """Seconds a frame takes to reach a policy once the shutter has fallen. The
  frame in hand is older still, by however long ago the shutter last fell."""


@dataclass(frozen=True)
class Observation:
  """What a policy sees, one entry per world."""

  row: Int64Array
  """The rows the step spanned, ``(worlds, rows)``."""
  state: Float64Array
  """The driven hands at each row, ``(worlds, rows, driven, 8)``."""
  image: UInt8Array
  """What each lens saw, ``(worlds, lenses, size, size, 3)``, black outside the disc."""
  body: Int32Array
  """The body each pixel struck, ``(worlds, lenses, size, size)``, ``-1`` for none."""
  image_row: Int64Array
  """The row each lens's frame was taken at, ``(worlds, lenses)``."""


@dataclass(frozen=True)
class Frame:
  """One lens's exposure: what it saw, and the instant its shutter fell,
  counted in rows."""

  at: float
  image: UInt8Array
  body: Int32Array

  @property
  def row(self) -> int:
    """The row it is filed under, as a capture files one: the nearest."""
    return int(round(self.at))


STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS
DRIVES = mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_XFRC_APPLIED


class Rig:
  """``worlds`` copies of the world with the hands standing in it. The
  worlds' data are the truth between steps."""

  def __init__(self, cfg: Cfg, worlds: int = 1):
    self.cfg = cfg
    self.worlds = worlds
    clock = cfg.clock
    spec = cfg.world()
    cfg.embodiment.attach(spec)
    self.model = model = spec.compile()
    model.opt.timestep = clock.physics_dt
    self.hands = cfg.embodiment.bind(model)
    self.slots = tuple(hand.slot for hand in self.hands)
    self._driven = [self.slots.index(slot) for slot in cfg.driven]
    """Which hands the action names."""
    self.lenses = [s for s in self.slots if s in cfg.driven or s in cfg.seen]
    """The slots the observation carries frames for."""
    self.datas = [mujoco.MjData(model) for _ in range(worlds)]
    self._scratch = mujoco.MjData(model)
    self._renderer = Renderer(
      model,
      {
        slot: (self.hands[self.slots.index(slot)].base, cfg.cameras.get(slot, NOMINAL))
        for slot in self.lenses
      },
      cfg.image_size,
    )
    self._ramp = Ramp(
      worlds,
      len(self.slots),
      clock.tracked,
      clock.per_row * clock.per_pose,
      clock.pose_dt,
    )
    threads = min(worlds, os.cpu_count() or 1)
    self._roll = rollout.Rollout(nthread=threads)
    self._pool = [mujoco.MjData(model) for _ in range(threads)]
    self._state = np.zeros((worlds, mujoco.mj_stateSize(model, STATE)))
    self._stepped = np.zeros((worlds, 1, self._state.shape[1]))
    self._warmstart = np.zeros((worlds, model.nv))
    self._control = np.zeros((worlds, 1, mujoco.mj_stateSize(model, DRIVES)))
    self._ctrl = self._control[:, 0, : model.nu]
    self._xfrc = self._control[:, 0, model.nu :].reshape(worlds, model.nbody, 6)
    self._qpos = self._state[:, 1 : 1 + model.nq]
    self._qvel = self._state[:, 1 + model.nq : 1 + model.nq + model.nv]
    rows, hands = clock.per_step, len(self.slots)
    self.qpos_rows = np.zeros((worlds, rows, model.nq))
    self.qvel_rows = np.zeros((worlds, rows, model.nv))
    """Every world's joint coordinates and velocities at each row of the last step."""
    self.state_rows = np.zeros((worlds, rows, hands, 8))
    """Every world's hands at each row of the last step, ``(worlds, rows, hands, 8)``."""
    self._period = np.array(
      [
        1.0 / (cfg.cameras.get(slot, NOMINAL).hz * clock.physics_dt)
        for slot in self.lenses
      ]
    )
    """Physics steps between one exposure of a lens and its next."""
    self._phase = np.zeros((worlds, len(self.lenses)))
    self._shot = np.zeros((worlds, len(self.lenses)), np.int64)
    self._next = np.zeros((worlds, len(self.lenses)), np.int64)
    """The physics step each lens's next exposure falls on, from its episode."""
    self._frames: list[dict[str, list[Frame]]] = [{} for _ in range(worlds)]
    self.exposed: list[dict[str, list[Frame]]] = [{} for _ in range(worlds)]
    """What each lens took during the last step, by slot, oldest first."""
    self.row = np.zeros(worlds, np.int64)
    """The row each world stands on."""
    self.broke = np.zeros(worlds, np.bool_)
    """Whether MuJoCo could not carry the world since it began."""
    self.tracking = np.zeros(worlds)
    """The largest gap between a driven hand and its setpoint at the last
    step's end, in metres."""

  def place(self, world: int, hands: Float64Array) -> None:
    """Start a world from rest with every hand where a state, ``(hands, 8)``,
    says, or raise ``OutOfReach``. A scene puts its own things on
    ``datas[world]`` before ``begin``."""
    data = self.datas[world]
    mujoco.mj_resetData(self.model, data)
    for index, (hand, state) in enumerate(zip(self.hands, hands, strict=True)):
      self.cfg.embodiment.place(data, index, state[:7])
      data.qpos[list(hand.fingers)] = state[7], -state[7]

  def begin(
    self, worlds: Sequence[int], phases: Float64Array | None = None
  ) -> Observation:
    """Open an episode in the named worlds where their data stand; every
    other carries on. ``phases``, ``(worlds, lenses)`` in ``[0, 1)``, is where
    each lens stands in its own period."""
    chosen = list(worlds)
    for world in chosen:
      mujoco.mj_forward(self.model, self.datas[world])
    self.row[chosen] = 0
    self.broke[chosen] = False
    self.tracking[chosen] = 0.0
    self._gather()
    states = self.states(self._qpos[chosen])
    self.state_rows[chosen] = states[:, None]
    self.qpos_rows[chosen] = self._qpos[chosen, None]
    self.qvel_rows[chosen] = self._qvel[chosen, None]
    self._ramp.hold(chosen, states)
    self.cfg.embodiment.reset(chosen, self._qpos)
    self._phase[chosen] = 0.0 if phases is None else phases
    for world in chosen:
      self._frames[world] = {
        slot: self._running(world, lens) for lens, slot in enumerate(self.lenses)
      }
      self.exposed[world] = {slot: [] for slot in self.lenses}
    return self._observe()

  def _running(self, world: int, lens: int) -> list[Frame]:
    """What a lens already running would have in hand as an episode opens:
    the exposures reaching back past its delay, all of the world as it stands."""
    clock = self.cfg.clock
    per_frame = self._period[lens] / clock.per_row
    behind = self.cfg.image_lag / clock.row_dt
    last = math.floor(0.5 / self._period[lens] - self._phase[world, lens])
    first = last - math.ceil(behind / per_frame) - 1
    image, body = self._shoot(world, lens)
    self._shot[world, lens] = last + 1
    self._next[world, lens] = self._shutter(last + 1, world, lens)
    return [
      Frame(self._shutter(shot, world, lens) / clock.per_row, image, body)
      for shot in range(first, last + 1)
    ]

  def step(self, action: Float64Array) -> Observation:
    """Track the leading poses of every world's trajectory, ``(worlds, poses,
    driven, 8)``. A world MuJoCo could not carry is caught by ``broke``."""
    clock, model = self.cfg.clock, self.model
    full = hold(self._ramp.last, action.shape[1])
    full[:, :, self._driven] = action
    self._ramp.process(full)
    self._gather()
    began = self._state[:, 0].copy()
    self.exposed = [{slot: [] for slot in self.lenses} for _ in range(self.worlds)]
    for substep in range(clock.per_row * clock.per_step):
      self.cfg.embodiment.command(
        self._ramp, substep, self._qpos, self._ctrl, self._xfrc
      )
      self._roll.rollout(
        model,
        self._pool,
        self._state,
        self._control,
        control_spec=DRIVES,
        nstep=1,
        state=self._stepped,
        initial_warmstart=self._warmstart,
      )
      self._state[:] = self._stepped[:, 0]
      elapsed = self.row * clock.per_row + substep + 1
      firing = np.argwhere(self._next == elapsed[:, None])
      if firing.size:
        self._land(sorted({int(world) for world, _ in firing}))
        for world, lens in firing:
          self._expose(int(world), int(lens), int(elapsed[world]))
      if (substep + 1) % clock.per_row:
        continue
      row = (substep + 1) // clock.per_row
      self.qpos_rows[:, row - 1] = self._qpos
      self.qvel_rows[:, row - 1] = self._qvel
    self._land()
    self.row += clock.per_step
    self.state_rows = self.states(self.qpos_rows.reshape(-1, model.nq)).reshape(
      self.state_rows.shape
    )
    self.broke |= self._state[:, 0] < began + clock.step_dt - 0.5 * clock.physics_dt
    apart = (
      self.state_rows[:, -1, self._driven, :3]
      - self._ramp.position[:, -1, self._driven]
    )
    self.tracking = np.linalg.norm(apart, axis=-1).max(axis=1)
    return self._observe()

  def states(self, qpos: Float64Array) -> Float64Array:
    """Every hand read out of ``qpos``, ``(n, nq)``, ``(n, hands, 8)``."""
    return gripper.states(self.model, self._scratch, self.hands, qpos)

  def twists(self, qpos: Float64Array, qvel: Float64Array) -> Float64Array:
    """Every hand's base velocity in the room frame, ``(n, hands, 6)``."""
    return gripper.twists(self.model, self._scratch, self.hands, qpos, qvel)

  def _shutter(self, shot: int, world: int, lens: int) -> int:
    """The physics step an exposure of one lens falls on, counted from the
    step its episode opened at."""
    return int(round((shot + self._phase[world, lens]) * self._period[lens]))

  def _expose(self, world: int, lens: int, elapsed: int) -> None:
    """Fire one lens, for the observation to hand over and a dataset to
    record, and wind it on to its next exposure."""
    slot = self.lenses[lens]
    frame = Frame(elapsed / self.cfg.clock.per_row, *self._shoot(world, lens))
    self._frames[world][slot].append(frame)
    self.exposed[world][slot].append(frame)
    self._shot[world, lens] += 1
    self._next[world, lens] = self._shutter(int(self._shot[world, lens]), world, lens)

  def _deliver(self, world: int, lens: int) -> Frame:
    """The newest exposure of a lens that has reached the policy, and the
    oldest of its frames still worth keeping."""
    held = self._frames[world][self.lenses[lens]]
    cutoff = self.row[world] - self.cfg.image_lag / self.cfg.clock.row_dt
    at = max(index for index, frame in enumerate(held) if frame.at <= cutoff)
    self._frames[world][self.lenses[lens]] = held[at:]
    return held[at]

  def _gather(self) -> None:
    for world, data in enumerate(self.datas):
      mujoco.mj_getState(self.model, data, self._state[world], STATE)

  def _land(self, worlds: Sequence[int] | None = None) -> None:
    for world in range(self.worlds) if worlds is None else worlds:
      data = self.datas[world]
      mujoco.mj_setState(self.model, data, self._state[world], STATE)
      mujoco.mj_forward(self.model, data)

  def _shoot(self, world: int, lens: int) -> tuple[UInt8Array, Int32Array]:
    slot = self.lenses[lens]
    view = self._renderer.render(self.datas[world], (slot,))[slot]
    return view.image, view.body

  def _observe(self) -> Observation:
    rows = self.cfg.clock.per_step
    frames = [
      [self._deliver(world, lens) for lens in range(len(self.lenses))]
      for world in range(self.worlds)
    ]
    return Observation(
      row=self.row[:, None] + np.arange(1 - rows, 1),
      state=self.state_rows[:, :, self._driven].copy(),
      image=np.stack([np.stack([frame.image for frame in world]) for world in frames]),
      body=np.stack([np.stack([frame.body for frame in world]) for world in frames]),
      image_row=np.array([[frame.row for frame in world] for world in frames]),
    )
