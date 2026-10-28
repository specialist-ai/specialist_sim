"""The rig: hands on their embodiment standing in a scene, stepped together
across many envs. Time is counted in rows, the lattice a dataset records a
hand at.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import mujoco
import numpy as np

from specialist_sim import gripper
from specialist_sim.action import Interpolator, hold
from specialist_sim.arrays import (
  Float64Array,
  Int32Array,
  Int64Array,
  UInt8Array,
)
from specialist_sim.camera import NOMINAL, Camera
from specialist_sim.embodiment import Embodiment, Floating
from specialist_sim.gripper import SLOTS
from specialist_sim.physics import Physics
from specialist_sim.render import Renderer

__all__ = ["Cfg", "Clock", "Frame", "Observation", "Rig"]


@dataclass(frozen=True)
class Clock:
  physics_dt: float = 1.0 / 1200.0
  per_row: int = 12
  """Physics steps in a row."""
  per_action: int = 2
  """Rows in one action of an action chunk."""
  per_step: int = 10
  """Rows in a control step, a whole number of actions."""

  @property
  def row_dt(self) -> float:
    return self.physics_dt * self.per_row

  @property
  def action_dt(self) -> float:
    return self.row_dt * self.per_action

  @property
  def step_dt(self) -> float:
    return self.row_dt * self.per_step

  @property
  def n_action_steps(self) -> int:
    """Actions of a chunk that fall inside one control step."""
    return self.per_step // self.per_action


@dataclass(frozen=True)
class Cfg:
  scene: Callable[[], mujoco.MjSpec]
  """What the hands stand in."""
  embodiment: Embodiment = field(default_factory=Floating)
  clock: Clock = Clock()
  physics: Physics = Physics()
  cameras: dict[str, Camera] = field(default_factory=dict)
  """The lens each hand carries, by slot; a slot left out carries the nominal one."""
  image_size: int = 240
  driven: tuple[str, ...] = SLOTS
  """The hands the action names; every other hand holds where it was last sent."""
  seen: tuple[str, ...] = ()
  """Slots whose lens the observation carries beyond the driven hands' own."""
  camera_latency: float = 0.0
  """Seconds a frame takes to reach a policy once the shutter has fallen. The
  frame in hand is older still, by however long ago the shutter last fell."""


@dataclass(frozen=True)
class Observation:
  """What a policy sees, one entry per env."""

  row: Int64Array
  """The rows the step spanned, ``(num_envs, rows)``."""
  state: Float64Array
  """The driven hands at each row, ``(num_envs, rows, driven, 8)``."""
  image: UInt8Array
  """What each lens saw, ``(num_envs, lenses, size, size, 3)``, black outside
  the disc."""
  body: Int32Array
  """The body each pixel struck, ``(num_envs, lenses, size, size)``, ``-1`` for none."""
  image_row: Int64Array
  """The row each lens's frame was taken at, ``(num_envs, lenses)``."""


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


UNSTABLE = (
  mujoco.mjtWarning.mjWARN_BADQPOS,
  mujoco.mjtWarning.mjWARN_BADQVEL,
  mujoco.mjtWarning.mjWARN_BADQACC,
)
"""The warnings MuJoCo raises as it resets an env it cannot carry."""


class Rig:
  """``num_envs`` copies of the scene with the hands standing in it. The
  envs' data are the truth between and during steps.

  Each env steps its own ``MjData`` with ``mj_step`` on a thread pool, which
  MuJoCo runs without the GIL, so an env lands bitwise where it would alone and
  keeps its solver's warm start from one physics step to the next."""

  def __init__(self, cfg: Cfg, num_envs: int = 1):
    self.cfg = cfg
    self.num_envs = num_envs
    clock = cfg.clock
    spec = cfg.scene()
    cfg.embodiment.attach(spec)
    self.model = model = spec.compile()
    cfg.physics.apply(model)
    model.opt.timestep = clock.physics_dt
    self.hands = cfg.embodiment.bind(model)
    self.slots = tuple(hand.slot for hand in self.hands)
    self._driven = [self.slots.index(slot) for slot in cfg.driven]
    """Which hands the action names."""
    self.lenses = [s for s in self.slots if s in cfg.driven or s in cfg.seen]
    """The slots the observation carries frames for."""
    self.datas = [mujoco.MjData(model) for _ in range(num_envs)]
    self._scratch = mujoco.MjData(model)
    self._renderer = Renderer(
      model,
      {
        slot: (self.hands[self.slots.index(slot)].base, cfg.cameras.get(slot, NOMINAL))
        for slot in self.lenses
      },
      cfg.image_size,
    )
    self._interpolator = Interpolator(
      num_envs,
      len(self.slots),
      clock.n_action_steps,
      clock.per_row * clock.per_action,
      clock.action_dt,
    )
    threads = min(num_envs, os.cpu_count() or 1)
    self._pool = ThreadPoolExecutor(threads) if threads > 1 else None
    self._shares = [range(part, num_envs, threads) for part in range(threads)]
    """The envs each thread steps."""
    self._qpos = np.zeros((num_envs, model.nq))
    self._ctrl = np.zeros((num_envs, model.nu))
    self._xfrc = np.zeros((num_envs, model.nbody, 6))
    rows, hands = clock.per_step, len(self.slots)
    self.qpos_rows = np.zeros((num_envs, rows, model.nq))
    self.qvel_rows = np.zeros((num_envs, rows, model.nv))
    """Every env's joint coordinates and velocities at each row of the last step."""
    self.state_rows = np.zeros((num_envs, rows, hands, 8))
    """Every env's hands at each row of the last step, ``(num_envs, rows,
    hands, 8)``."""
    self._period = np.array(
      [
        1.0 / (cfg.cameras.get(slot, NOMINAL).hz * clock.physics_dt)
        for slot in self.lenses
      ]
    )
    """Physics steps between one exposure of a lens and its next."""
    self._phase = np.zeros((num_envs, len(self.lenses)))
    self._shot = np.zeros((num_envs, len(self.lenses)), np.int64)
    self._next = np.zeros((num_envs, len(self.lenses)), np.int64)
    """The physics step each lens's next exposure falls on, from its episode."""
    self._frames: list[dict[str, list[Frame]]] = [{} for _ in range(num_envs)]
    self.exposed: list[dict[str, list[Frame]]] = [{} for _ in range(num_envs)]
    """What each lens took during the last step, by slot, oldest first."""
    self.row = np.zeros(num_envs, np.int64)
    """The row each env stands on."""
    self.unstable = np.zeros(num_envs, np.bool_)
    """Whether MuJoCo reset the env for want of a finite state since it began."""
    self.tracking_error = np.zeros(num_envs)
    """The largest gap between a driven hand and its setpoint at the last
    step's end, in meters."""

  def place(self, env_idx: int, hands: Float64Array) -> None:
    """Start an env from rest with every hand where a state, ``(hands, 8)``,
    says, or raise ``OutOfReach``. An env puts its own things on
    ``datas[env_idx]`` before ``begin``."""
    data = self.datas[env_idx]
    mujoco.mj_resetData(self.model, data)
    for index, (hand, state) in enumerate(zip(self.hands, hands, strict=True)):
      self.cfg.embodiment.place(data, index, state[:7])
      data.qpos[list(hand.fingers)] = state[7], -state[7]

  def begin(
    self, env_ids: Sequence[int], phases: Float64Array | None = None
  ) -> Observation:
    """Open an episode in the named envs where their data stand; every
    other carries on. ``phases``, ``(len(env_ids), lenses)`` in ``[0, 1)``, is where
    each lens stands in its own period."""
    chosen = list(env_ids)
    for env_idx in chosen:
      mujoco.mj_forward(self.model, self.datas[env_idx])
    self.row[chosen] = 0
    self.unstable[chosen] = False
    self.tracking_error[chosen] = 0.0
    self._gather()
    states = self.states(self._qpos[chosen])
    self.state_rows[chosen] = states[:, None]
    self.qpos_rows[chosen] = self._qpos[chosen, None]
    self.qvel_rows[chosen] = np.stack([self.datas[env_idx].qvel for env_idx in chosen])[
      :, None
    ]
    self._interpolator.hold(chosen, states)
    self.cfg.embodiment.reset(chosen, self._qpos)
    self._phase[chosen] = 0.0 if phases is None else phases
    for env_idx in chosen:
      self._frames[env_idx] = {
        slot: self._running(env_idx, lens) for lens, slot in enumerate(self.lenses)
      }
      self.exposed[env_idx] = {slot: [] for slot in self.lenses}
    return self._observe()

  def _running(self, env_idx: int, lens: int) -> list[Frame]:
    """What a lens already running would have in hand as an episode opens:
    the exposures reaching back past its delay, all of the scene as it stands."""
    clock = self.cfg.clock
    per_frame = self._period[lens] / clock.per_row
    behind = self.cfg.camera_latency / clock.row_dt
    last = math.floor(0.5 / self._period[lens] - self._phase[env_idx, lens])
    first = last - math.ceil(behind / per_frame) - 1
    image, body = self._shoot(env_idx, lens)
    self._shot[env_idx, lens] = last + 1
    self._next[env_idx, lens] = self._shutter(last + 1, env_idx, lens)
    return [
      Frame(self._shutter(shot, env_idx, lens) / clock.per_row, image, body)
      for shot in range(first, last + 1)
    ]

  def step(self, action: Float64Array) -> Observation:
    """Track the leading actions of every env's action chunk, ``(num_envs,
    chunk_size, driven, 8)``. An env that goes unstable is caught by ``unstable``."""
    clock, model = self.cfg.clock, self.model
    full = hold(self._interpolator.last, action.shape[1])
    full[:, :, self._driven] = action
    self._interpolator.process(full)
    warned = np.array([_warnings(data) for data in self.datas])
    self.exposed = [{slot: [] for slot in self.lenses} for _ in range(self.num_envs)]
    for substep in range(clock.per_row * clock.per_step):
      self._gather()
      self.cfg.embodiment.command(
        self._interpolator, substep, self._qpos, self._ctrl, self._xfrc
      )
      if self._pool is None:
        self._push(self._shares[0])
      else:
        list(self._pool.map(self._push, self._shares))
      elapsed = self.row * clock.per_row + substep + 1
      firing = np.argwhere(self._next == elapsed[:, None])
      for env_idx in sorted({int(env_idx) for env_idx, _ in firing}):
        mujoco.mj_kinematics(model, self.datas[env_idx])
      for env_idx, lens in firing:
        self._expose(int(env_idx), int(lens), int(elapsed[env_idx]))
      if (substep + 1) % clock.per_row:
        continue
      row = (substep + 1) // clock.per_row
      for env_idx, data in enumerate(self.datas):
        self.qpos_rows[env_idx, row - 1] = data.qpos
        self.qvel_rows[env_idx, row - 1] = data.qvel
    for data in self.datas:
      mujoco.mj_forward(model, data)
    self.row += clock.per_step
    self.state_rows = self.states(self.qpos_rows.reshape(-1, model.nq)).reshape(
      self.state_rows.shape
    )
    self.unstable |= np.array([_warnings(data) for data in self.datas]) > warned
    apart = (
      self.state_rows[:, -1, self._driven, :3]
      - self._interpolator.position[:, -1, self._driven]
    )
    self.tracking_error = np.linalg.norm(apart, axis=-1).max(axis=1)
    return self._observe()

  def states(self, qpos: Float64Array) -> Float64Array:
    """Every hand read out of ``qpos``, ``(n, nq)``, ``(n, hands, 8)``."""
    return gripper.states(self.model, self._scratch, self.hands, qpos)

  def twists(self, qpos: Float64Array, qvel: Float64Array) -> Float64Array:
    """Every hand's base tag velocity in the room frame, ``(n, hands, 6)``."""
    return gripper.twists(self.model, self._scratch, self.hands, qpos, qvel)

  def _shutter(self, shot: int, env_idx: int, lens: int) -> int:
    """The physics step an exposure of one lens falls on, counted from the
    step its episode opened at."""
    return int(round((shot + self._phase[env_idx, lens]) * self._period[lens]))

  def _expose(self, env_idx: int, lens: int, elapsed: int) -> None:
    """Fire one lens, for the observation to hand over and a dataset to
    record, and wind it on to its next exposure."""
    slot = self.lenses[lens]
    frame = Frame(elapsed / self.cfg.clock.per_row, *self._shoot(env_idx, lens))
    self._frames[env_idx][slot].append(frame)
    self.exposed[env_idx][slot].append(frame)
    self._shot[env_idx, lens] += 1
    self._next[env_idx, lens] = self._shutter(
      int(self._shot[env_idx, lens]), env_idx, lens
    )

  def _deliver(self, env_idx: int, lens: int) -> Frame:
    """The newest exposure of a lens that has reached the policy, and the
    oldest of its frames still worth keeping."""
    held = self._frames[env_idx][self.lenses[lens]]
    cutoff = self.row[env_idx] - self.cfg.camera_latency / self.cfg.clock.row_dt
    at = max(index for index, frame in enumerate(held) if frame.at <= cutoff)
    self._frames[env_idx][self.lenses[lens]] = held[at:]
    return held[at]

  def _gather(self) -> None:
    for env_idx, data in enumerate(self.datas):
      self._qpos[env_idx] = data.qpos

  def _push(self, share: Sequence[int]) -> None:
    """One physics step of the envs in ``share``, under the commanded drives."""
    for env_idx in share:
      data = self.datas[env_idx]
      data.ctrl[:] = self._ctrl[env_idx]
      data.xfrc_applied[:] = self._xfrc[env_idx]
      mujoco.mj_step(self.model, data)

  def _shoot(self, env_idx: int, lens: int) -> tuple[UInt8Array, Int32Array]:
    slot = self.lenses[lens]
    view = self._renderer.render(self.datas[env_idx], (slot,))[slot]
    return view.image, view.body

  def _observe(self) -> Observation:
    rows = self.cfg.clock.per_step
    frames = [
      [self._deliver(env_idx, lens) for lens in range(len(self.lenses))]
      for env_idx in range(self.num_envs)
    ]
    return Observation(
      row=self.row[:, None] + np.arange(1 - rows, 1),
      state=self.state_rows[:, :, self._driven].copy(),
      image=np.stack([np.stack([frame.image for frame in taken]) for taken in frames]),
      body=np.stack([np.stack([frame.body for frame in taken]) for taken in frames]),
      image_row=np.array([[frame.row for frame in taken] for taken in frames]),
    )


def _warnings(data: mujoco.MjData) -> int:
  """How many times MuJoCo has reset ``data`` for want of a finite state."""
  return sum(int(data.warning[kind].number) for kind in UNSTABLE)
