"""A pick-and-place environment for evaluating policies.

Follows the Gymnasium interface without importing it.

The environment is a stack of sensors on one clock. A row is one tick of the
step timeline; a control step spans several and returns at the last of them.
Every value an observation carries is labelled with the row it happened at, and
nothing here reports an age or a delay: a reader subtracts.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import specialist_description
from mujoco import rollout

from specialist_sim import randomise, spaces
from specialist_sim.action import ActionCfg, Ramp
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.calibration import SCENE_VIEW, CameraCfg
from specialist_sim.control import XFRC, ControllerCfg
from specialist_sim.embodiment import (
  ASSETS,
  VARIANT,
  EmbodimentCfg,
  FloatingCfg,
  OutOfReach,
)
from specialist_sim.render import Renderer, seat
from specialist_sim.scene import Scene, SceneCfg
from specialist_sim.spaces import F_STEP, HORIZON, PER_HAND
from specialist_sim.task import BIMANUAL, Judge, Task

__all__ = ["Contract", "PickPlaceCfg", "PickPlaceEnv"]


MAX_LAG = 20
"""How many rows a camera may run behind the rest of the stack."""

STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS
"""What carries from one substep to the next. A rollout names no other state, so
the solver starts every substep from a zero warmstart, and a run of single
substeps drifts from an unbroken one."""


@dataclass(frozen=True)
class PickPlaceCfg:
  control_hz: float = 10.0
  camera_lag: int = 0
  """How many rows behind the boundary the delivered frame was captured."""
  max_steps: int | None = None
  """Steps after which a world truncates; ``None`` leaves the ending to the task."""
  image_size: int = 240
  render_size: int = 480
  reset_attempts: int = 20
  penetration: float = 1e-4
  cameras: dict[str, CameraCfg] | None = None
  """The lens each hand carries, by slot; ``None`` seats a nominal camera on
  every hand the embodiment stands up."""
  embodiment: EmbodimentCfg = field(default_factory=FloatingCfg)
  """What carries the hands, and so which slots the room holds."""
  scene: SceneCfg = field(default_factory=SceneCfg)
  action: ActionCfg = field(default_factory=ActionCfg)
  """How fast a ramp may move a hand, and the impedance that chases it on a free rig."""
  controller: ControllerCfg | None = None
  """What drives the hands along the ramp; ``None`` leaves it to the embodiment."""
  task: Task = BIMANUAL
  """Which hands a policy drives, where the episode opens, and what ends it."""


@dataclass(frozen=True)
class Contract:
  """The environment's side of the seam a policy is checked against."""

  step_hz: float
  action_hz: float
  control_hz: float
  horizon: int
  image_size: int
  mask_radius: float | None
  driven: tuple[str, ...]
  lenses: tuple[str, ...]


class _Plant:
  """Every world's physics, stepped together through mujoco's thread pool.

  The state rows are the truth while a control step runs; the worlds' MjData
  are the truth between steps, gathered at the start of one and landed at its
  boundaries. Stepping goes one substep at a time: a controller answers to
  where each plant is, so the pool cannot be handed a whole control step.
  ``drives`` names the state components a substep is driven by.
  """

  def __init__(self, model: mujoco.MjModel, datas: list[mujoco.MjData], drives: int):
    self.model, self.datas, self.drives = model, datas, drives
    worlds = len(datas)
    threads = min(worlds, os.cpu_count() or 1)
    self._roll = rollout.Rollout(nthread=threads)
    self._scratch = [mujoco.MjData(model) for _ in range(threads)]
    self._state = np.zeros((worlds, mujoco.mj_stateSize(model, STATE)))
    self._stepped = np.zeros((worlds, 1, self._state.shape[1]))
    self._warmstart = np.zeros((worlds, model.nv))
    """Zeros, always: rollout otherwise reuses whatever its scratch data held,
    which would make a world's step depend on which thread ran it last."""
    self._control = np.zeros((worlds, 1, mujoco.mj_stateSize(model, drives)))
    self.ctrl = self._control[:, 0, : model.nu]
    self.xfrc = (
      self._control[:, 0, model.nu :].reshape(worlds, model.nbody, 6)
      if drives & XFRC
      else np.zeros((worlds, model.nbody, 6))
    )
    """Views a command is written straight into, one row per world; the wrench
    is a view only when the plant is driven by one."""
    self.qpos = self._state[:, 1 : 1 + model.nq]
    self.qvel = self._state[:, 1 + model.nq : 1 + model.nq + model.nv]
    """Views of every world's joint coordinates and velocities, current while
    a control step runs."""

  def gather(self) -> None:
    """Take the state the worlds hold, which stepping runs from."""
    for world, data in enumerate(self.datas):
      mujoco.mj_getState(self.model, data, self._state[world], STATE)

  def substep(self) -> None:
    """Advance every world one physics step under the written command."""
    self._roll.rollout(
      self.model,
      self._scratch,
      self._state,
      self._control,
      control_spec=self.drives,
      nstep=1,
      state=self._stepped,
      initial_warmstart=self._warmstart,
    )
    self._state[:] = self._stepped[:, 0]

  def land(self) -> None:
    """Put the stepped state back on the worlds, so a reader sees what ran."""
    for world, data in enumerate(self.datas):
      mujoco.mj_setState(self.model, data, self._state[world], STATE)
      mujoco.mj_forward(self.model, data)


class PickPlaceEnv:
  metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 10}

  def __init__(
    self,
    cfg: PickPlaceCfg | None = None,
    render_mode: str | None = None,
    worlds: int = 1,
  ):
    self.cfg = cfg or PickPlaceCfg()
    if worlds < 1:
      raise ValueError(f"an environment holds one world or more, not {worlds}")
    self.worlds = worlds
    modes = PickPlaceEnv.metadata["render_modes"]
    if render_mode is not None and render_mode not in modes:
      raise ValueError(f"{render_mode} is not one of {modes}")
    if not 0 <= self.cfg.camera_lag <= MAX_LAG:
      raise ValueError(
        f"a camera runs 0 to {MAX_LAG} rows behind, not {self.cfg.camera_lag}"
      )
    self.scene = Scene(self.cfg.scene, self.cfg.embodiment.build())
    slots = self.scene.slots
    self.cameras = self.cfg.cameras or {slot: CameraCfg() for slot in slots}
    """The lens each hand carries."""
    if set(self.cameras) != set(slots):
      raise ValueError(f"the cameras are {sorted(slots)}, not {sorted(self.cameras)}")
    self.driven = self.cfg.task.driven
    """The hands the action commands and the observation carries rows for."""
    self.lenses = self.cfg.task.lenses
    """The lenses the observation carries images for."""
    if not set(self.driven) | set(self.lenses) <= set(slots):
      raise ValueError(
        f"the task drives {self.driven} and sees {self.lenses}, and the room holds {slots}"
      )
    self._driven = [slots.index(slot) for slot in self.driven]
    self.render_mode = render_mode
    self.metadata = {**PickPlaceEnv.metadata, "render_fps": round(self.cfg.control_hz)}
    self._stand()
    self._pace()
    self._furnish()

  def _stand(self) -> None:
    """Seat the cameras, and stand one data and one judge per world.

    Every camera is seated before any world computes kinematics, so the first
    frames of every world already look through a mounted lens.
    """
    model = self.scene.model
    for name, camera in {**self.cameras, "scene_view": SCENE_VIEW}.items():
      camera.mount(model, seat(model, name))
    self.datas = [self.scene.data] + [
      mujoco.MjData(model) for _ in range(self.worlds - 1)
    ]
    self.judges = [Judge(self.scene, data, self.cfg.task) for data in self.datas]
    self.judge = self.judges[0]
    """The first world's judge, which is the whole of it when there is one."""

  def _pace(self) -> None:
    """Fit the three clocks together: physics steps, rows, and control steps;
    then lay the ramp on them, and stand the controller and the plant it drives."""
    control_dt = 1.0 / self.cfg.control_hz
    substeps = round(control_dt / self.cfg.scene.physics_dt)
    if not np.isclose(substeps * self.cfg.scene.physics_dt, control_dt):
      raise ValueError("the control step must be a whole number of physics steps")
    rows = F_STEP / self.cfg.control_hz
    if not np.isclose(rows, round(rows)):
      raise ValueError("a control step must span a whole number of rows")
    self.rows = round(rows)
    """How many rows of the step timeline one control step spans."""
    self._capture = self.rows - self.cfg.camera_lag % self.rows
    """The row within a step the camera's frame is taken on. A whole number of
    steps of lag lands on the boundary, which is the last row."""
    if substeps % self.rows:
      raise ValueError("a row must span a whole number of physics steps")
    self._per_row = substeps // self.rows
    action = self.cfg.action
    self.ramp = Ramp(
      len(self.scene.hands),
      substeps,
      control_dt,
      action.max_speed,
      action.max_turn,
      self.worlds,
    )
    """The commanded trajectory, resampled onto this step's substeps."""
    settings = self.cfg.controller or self.scene.embodiment.default_controller(action)
    self.controller = settings.build(self.scene, self.ramp)
    self.plant = _Plant(self.scene.model, self.datas, self.controller.drives)

  def _furnish(self) -> None:
    """Allocate what an episode fills: seeds, clocks, samples and frames."""
    self._draws = [np.random.default_rng() for _ in range(self.worlds)]
    """One stream per world, so a world's start answers to its own seed."""
    self.np_random = self._draws[0]
    self._steps = np.zeros(self.worlds, np.int64)
    """How many control steps each world has run since it was last reset."""
    self._at = np.zeros(self.worlds, np.int64)
    """The row of the step timeline each world stands on."""
    hands = len(self.scene.hands)
    self._states = np.zeros((self.worlds, hands, self.rows, PER_HAND), np.float32)
    """Every world's hands at each row of the step, laid out like a state."""
    self._frames: list[dict[int, dict[str, Any]]] = [{} for _ in range(self.worlds)]
    """What each world's lenses saw, by the row it was taken on."""
    self.renderer: Renderer | None = None
    self.watcher: Renderer | None = None
    self.views: dict[str, Any] = {}

  @property
  def observation_space(self) -> Any:
    return spaces.gym_spaces(self.image_shape, self.rows, self.driven, self.lenses)[0]

  @property
  def action_space(self) -> Any:
    return spaces.gym_spaces(self.image_shape, self.rows, self.driven, self.lenses)[1]

  @property
  def contract(self) -> Contract:
    """What a policy must agree with this environment on."""
    lens = self.cameras[self.lenses[0]]
    return Contract(
      step_hz=F_STEP,
      action_hz=spaces.ACTION_HZ,
      control_hz=self.cfg.control_hz,
      horizon=HORIZON,
      image_size=self.cfg.image_size,
      mask_radius=lens.mask_radius,
      driven=self.driven,
      lenses=self.lenses,
    )

  @property
  def revision(self) -> str:
    """A hash of everything that decides what this environment does with a
    seed: the room, the gripper it is described with, the configuration and the
    physics engine."""
    digest = hashlib.sha256()
    for root in (ASSETS, specialist_description.variant_dir(VARIANT)):
      for path in sorted(root.rglob("*")):
        if path.is_file():
          digest.update(path.relative_to(root).as_posix().encode())
          digest.update(path.read_bytes())
    digest.update(repr(self.cfg).encode())
    digest.update(mujoco.__version__.encode())
    return digest.hexdigest()[:12]

  @property
  def image_shape(self) -> tuple[int, int, int]:
    return (self.cfg.image_size, self.cfg.image_size, 3)

  def reset(
    self,
    *,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    options: dict[str, Any] | Sequence[dict[str, Any] | None] | None = None,
    worlds: Sequence[int] | None = None,
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draw a start, for every world or only for the ones named.

    A world keeps its own clock, so one restarted partway through a batch counts
    its rows and its steps from zero while the rest carry on.
    """
    if seed is not None and seeds is not None:
      raise ValueError("a reset names one seed or a seed per world, not both")
    chosen = list(range(self.worlds)) if worlds is None else list(worlds)
    if not chosen or len(set(chosen)) != len(chosen):
      raise ValueError(f"a reset names each world at most once, not {worlds}")
    if any(not 0 <= world < self.worlds for world in chosen):
      raise ValueError(f"{self.worlds} worlds are numbered 0 to {self.worlds - 1}")
    if seeds is not None:
      if len(seeds) != len(chosen):
        raise ValueError(f"{len(chosen)} worlds want that many seeds, not {len(seeds)}")
      for world, one in zip(chosen, seeds, strict=True):
        self._draws[world] = np.random.default_rng(int(one))
    elif seed is not None:
      for index, world in enumerate(chosen):
        self._draws[world] = np.random.default_rng(seed + index)
    self.np_random = self._draws[0]
    starts = self._starts(options, chosen)
    for world, start in zip(chosen, starts, strict=True):
      self._begin(world, self.datas[world], start)
      self.judges[world].reset()
    self._steps[chosen] = 0
    self._at[chosen] = 0
    self.plant.gather()
    states = self.scene.observe(self.plant.qpos[chosen])
    self._states[chosen] = states[:, :, None]
    self.ramp.hold(chosen, states)
    self.controller.reset(self.datas, chosen)
    # The rows before an episode are the state it settled into, and so is every
    # frame the first boundaries ask for.
    seen = self._render(chosen)
    for index, world in enumerate(chosen):
      self._frames[world] = dict.fromkeys(
        range(-self.cfg.camera_lag, 1, self.rows), seen[index]
      )
    report = self._stacked([judge.info() for judge in self.judges])
    report["tracking"] = np.zeros(self.worlds)
    return self._observe(), report

  def _starts(
    self,
    options: dict[str, Any] | Sequence[dict[str, Any] | None] | None,
    chosen: Sequence[int],
  ) -> list[dict[str, Any]]:
    """One named start per world being reset, however the caller gave it."""
    if options is None or isinstance(options, dict):
      given: list[dict[str, Any] | None] = [options] * len(chosen)
    else:
      given = list(options)
      if len(given) != len(chosen):
        raise ValueError(
          f"{len(chosen)} worlds want that many starts, not {len(given)}"
        )
    out = []
    for one in given:
      named = one or {}
      if not set(named) <= {"state", "cube"}:
        raise ValueError(f"a start names state and cube, not {sorted(named)}")
      out.append(named)
    return out

  def _begin(self, world: int, data: mujoco.MjData, start: dict[str, Any]) -> None:
    """Settle one world onto a start it can hold, redrawing what the caller
    left free; a fully named start gets one attempt."""
    fixed = {"state", "cube"} <= set(start)
    for _ in range(1 if fixed else self.cfg.reset_attempts):
      try:
        self._place(world, data, start)
      except OutOfReach:
        if "state" in start:
          raise
        continue
      if not self._penetrating(data):
        return
    raise ValueError(
      f"world {world} found no start it can hold in {self.cfg.reset_attempts} draws"
    )

  def _place(self, world: int, data: mujoco.MjData, options: dict[str, Any]) -> None:
    """Draw the task's start, then put whatever the caller named where it says."""
    mujoco.mj_resetData(self.scene.model, data)
    self.cfg.task.start.place(self.scene, data, self._draws[world])
    if (state := options.get("state")) is not None:
      randomise.place_hands(self.scene, data, np.asarray(state, np.float64))
    if (cube := options.get("cube")) is not None:
      randomise.place_cube(self.scene, data, np.asarray(cube, np.float64))
    mujoco.mj_forward(self.scene.model, data)

  def step(
    self,
    action: Float64Array,
    watch: Callable[[int, Float64Array, Float64Array], None] | None = None,
  ) -> tuple[dict[str, Any], Float64Array, BoolArray, BoolArray, dict[str, Any]]:
    """Track the leading poses of a trajectory, and report where that leaves things.

    Every world is answered for along a leading axis, in reward, in what ended,
    and in everything an observation and its report carry, however many there
    are: a shape does not turn on how the environment was built.

    ``watch`` sees each physics step by its index, with every world's joint
    coordinates and velocities as that step left them, for a caller recording
    the plant faster than a control step reports it.
    """
    self.ramp.process(action, self._driven)
    plant = self.plant
    # A world is the truth between steps, so anything moved on one since the
    # last step is what this one runs from.
    plant.gather()
    for substep in range(self.ramp.substeps):
      self.controller.command(
        self.ramp, substep, plant.qpos, plant.qvel, plant.ctrl, plant.xfrc
      )
      plant.substep()
      if watch is not None:
        watch(substep, plant.qpos, plant.qvel)
      if (substep + 1) % self._per_row:
        continue
      row = (substep + 1) // self._per_row
      self._sample(row - 1)
      if row == self._capture:
        plant.land()
        self._keep(self._capture)
    if self._capture != self.rows:
      plant.land()
    self._at += self.rows
    self._steps += 1
    judged = [judge.evaluate() for judge in self.judges]
    reward = np.array([one[0] for one in judged], np.float64)
    terminated = np.array([one[1] for one in judged], np.bool_)
    limit = self.cfg.max_steps
    truncated = (
      (self._steps >= limit) & ~terminated if limit else np.zeros_like(terminated)
    )
    report = self._stacked([one[2] for one in judged])
    report["tracking"] = self._tracking()
    return self._observe(), reward, terminated, truncated, report

  def _keep(self, offset: int) -> None:
    """Hold what every world's lenses see, on the row each of them is taking,
    and let go of every row no future observation can ask for."""
    seen = self._render()
    for world in range(self.worlds):
      kept = int(self._at[world]) + offset
      self._frames[world][kept] = seen[world]
      floor = kept - self.cfg.camera_lag
      for older in [row for row in self._frames[world] if row < floor]:
        del self._frames[world][older]

  def _sample(self, row: int) -> None:
    """Keep every world's hands at one row of the step timeline."""
    self._states[:, :, row] = self.scene.observe(self.plant.qpos)

  def _tracking(self) -> Float64Array:
    """Each world's largest gap between a driven hand and its setpoint at the
    step's end, in metres."""
    apart = (
      self._states[:, self._driven, -1, :3] - self.ramp.position[:, -1, self._driven]
    )
    out: Float64Array = np.linalg.norm(apart, axis=-1).max(axis=1)
    return out

  def _stacked(self, reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Every world's account of the cube, as one report."""
    return {key: np.stack([one[key] for one in reports]) for key in reports[0]}

  def render(self) -> np.ndarray | None:
    """The room, watched from across the table."""
    if self.render_mode is None:
      return None
    if self.watcher is None:
      self.watcher = Renderer(
        self.scene.model, {"scene_view": SCENE_VIEW}, self.cfg.render_size
      )
    watched: np.ndarray = self.watcher.render(self.scene.data)["scene_view"].image
    return watched

  def close(self) -> None:
    self.renderer = None
    self.watcher = None

  def _observe(self) -> dict[str, Any]:
    """The rows this step spanned, and the frame that has caught up with it."""
    taken = self._at - self.cfg.camera_lag
    frames = [self._frames[world][int(when)] for world, when in enumerate(taken)]
    out: dict[str, Any] = {
      "step": np.stack(
        [np.arange(one - self.rows + 1, one + 1, dtype=np.int64) for one in self._at]
      )
    }
    for hand, slot in enumerate(self.scene.slots):
      fields: dict[str, Any] = {}
      if slot in self.driven:
        fields["pose"] = self._states[:, hand, :, :7].copy()
        fields["opening"] = self._states[:, hand, :, 7].copy()
      if slot in self.lenses:
        fields["image"] = np.stack([frame[slot] for frame in frames])
        fields["image_step"] = taken.copy()
      if fields:
        out[slot] = fields
    return out

  def _render(self, worlds: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """What the named worlds' lenses see, every world's when none are named."""
    if self.renderer is None:
      self.renderer = Renderer(
        self.scene.model,
        {slot: self.cameras[slot] for slot in self.lenses},
        self.cfg.image_size,
      )
    out = []
    for world in range(self.worlds) if worlds is None else worlds:
      views = self.renderer.render(self.datas[world])
      if world == 0:
        self.views = views
      out.append({slot: view.image for slot, view in views.items()})
    return out

  def _penetrating(self, data: mujoco.MjData) -> bool:
    touching: Float64Array = data.contact.dist[: data.ncon]
    return bool((touching < -self.cfg.penetration).any())
