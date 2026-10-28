"""The rig: hands on their embodiment standing in a scene, stepped together
across many envs. Time is counted in rows, the lattice a dataset records a
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


STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS
DRIVES = mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_XFRC_APPLIED


class Rig:
  """``num_envs`` copies of the scene with the hands standing in it. The
  envs' data are the truth between steps."""

  def __init__(self, cfg: Cfg, num_envs: int = 1):
    self.cfg = cfg
    self.num_envs = num_envs
    clock = cfg.clock
    spec = cfg.scene()
    cfg.embodiment.attach(spec)
    self.model = model = spec.compile()
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
    self._roll = rollout.Rollout(nthread=threads)
    self._pool = [mujoco.MjData(model) for _ in range(threads)]
    self._state = np.zeros((num_envs, mujoco.mj_stateSize(model, STATE)))
    self._stepped = np.zeros((num_envs, 1, self._state.shape[1]))
    self._warmstart = np.zeros((num_envs, model.nv))
    self._control = np.zeros((num_envs, 1, mujoco.mj_stateSize(model, DRIVES)))
    self._ctrl = self._control[:, 0, : model.nu]
    self._xfrc = self._control[:, 0, model.nu :].reshape(num_envs, model.nbody, 6)
    self._qpos = self._state[:, 1 : 1 + model.nq]
    self._qvel = self._state[:, 1 + model.nq : 1 + model.nq + model.nv]
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
    """Whether MuJoCo could not carry the env since it began."""
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
    self.qvel_rows[chosen] = self._qvel[chosen, None]
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
    self._gather()
    began = self._state[:, 0].copy()
    self.exposed = [{slot: [] for slot in self.lenses} for _ in range(self.num_envs)]
    for substep in range(clock.per_row * clock.per_step):
      self.cfg.embodiment.command(
        self._interpolator, substep, self._qpos, self._ctrl, self._xfrc
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
        self._land(sorted({int(env_idx) for env_idx, _ in firing}))
        for env_idx, lens in firing:
          self._expose(int(env_idx), int(lens), int(elapsed[env_idx]))
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
    self.unstable |= self._state[:, 0] < began + clock.step_dt - 0.5 * clock.physics_dt
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
    """Every hand's base velocity in the room frame, ``(n, hands, 6)``."""
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
      mujoco.mj_getState(self.model, data, self._state[env_idx], STATE)

  def _land(self, env_ids: Sequence[int] | None = None) -> None:
    for env_idx in range(self.num_envs) if env_ids is None else env_ids:
      data = self.datas[env_idx]
      mujoco.mj_setState(self.model, data, self._state[env_idx], STATE)
      mujoco.mj_forward(self.model, data)

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
