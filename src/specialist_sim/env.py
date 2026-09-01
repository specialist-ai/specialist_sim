"""A pick-and-place environment for evaluating policies.

Follows the Gymnasium interface without importing it.

The environment is a stack of sensors on one clock. A row is one tick of the
step timeline; a control step spans several and returns at the last of them.
Every value an observation carries is labelled with the row it happened at, and
nothing here reports an age or a delay: a reader subtracts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from specialist_sim import randomise, spaces
from specialist_sim.action import ActionCfg, Hands
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.calibration import SCENE_VIEW, CameraCfg
from specialist_sim.plant import Plant
from specialist_sim.randomise import RandomiseCfg
from specialist_sim.render import Renderer, seat
from specialist_sim.scene import Scene, SceneCfg
from specialist_sim.spaces import F_STEP, GRIPPERS
from specialist_sim.task import PickPlace, TaskCfg
from specialist_sim.warp_plant import WarpPlant

__all__ = ["PickPlaceCfg", "PickPlaceEnv"]


MAX_LAG = 20
"""How many rows a camera may run behind the rest of the stack."""

PHYSICS = ("cpu", "gpu")
"""Where a control step is advanced: mujoco's thread pool, or mujoco_warp."""


@dataclass(frozen=True)
class PickPlaceCfg:
  control_hz: float = 10.0
  camera_lag: int = 0
  """How many rows behind the boundary the delivered frame was captured."""
  max_steps: int = 100
  image_size: int = 240
  render_size: int = 480
  reset_attempts: int = 20
  penetration: float = 1e-4
  cameras: dict[str, CameraCfg] = field(
    default_factory=lambda: {slot: CameraCfg() for slot in GRIPPERS}
  )
  scene: SceneCfg = field(default_factory=SceneCfg)
  action: ActionCfg = field(default_factory=ActionCfg)
  task: TaskCfg = field(default_factory=TaskCfg)
  randomise: RandomiseCfg = field(default_factory=RandomiseCfg)


class PickPlaceEnv:
  metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 10}

  def __init__(
    self,
    cfg: PickPlaceCfg | None = None,
    render_mode: str | None = None,
    worlds: int = 1,
    physics: str = "cpu",
  ):
    self.cfg = cfg or PickPlaceCfg()
    if worlds < 1:
      raise ValueError(f"an environment holds one world or more, not {worlds}")
    self.worlds = worlds
    if physics not in PHYSICS:
      raise ValueError(f"{physics} is not one of {list(PHYSICS)}")
    self.physics = physics
    modes = PickPlaceEnv.metadata["render_modes"]
    if render_mode is not None and render_mode not in modes:
      raise ValueError(f"{render_mode} is not one of {modes}")
    if set(self.cfg.cameras) != set(GRIPPERS):
      raise ValueError(
        f"the cameras are {sorted(GRIPPERS)}, not {sorted(self.cfg.cameras)}"
      )
    if (self.cfg.action.damping, self.cfg.action.rotational_damping) != (
      self.cfg.scene.damping,
      self.cfg.scene.rotational_damping,
    ):
      raise ValueError(
        "the feedforward cancels the rig damping, so the two must be equal"
      )
    if not 0 <= self.cfg.camera_lag <= MAX_LAG:
      raise ValueError(
        f"a camera runs 0 to {MAX_LAG} rows behind, not {self.cfg.camera_lag}"
      )
    self.render_mode = render_mode
    self.metadata = {**PickPlaceEnv.metadata, "render_fps": round(self.cfg.control_hz)}
    self._stand()
    self._pace()
    self._furnish()

  def _stand(self) -> None:
    """Build the worlds: the scene, its cameras seated, and one data and one
    task per world.

    Every camera is seated before any world computes kinematics, so the first
    frames of every world already look through a mounted lens.
    """
    self.scene = Scene(self.cfg.scene)
    model = self.scene.model
    for name, camera in {**self.cfg.cameras, "scene_view": SCENE_VIEW}.items():
      camera.mount(model, seat(model, name))
    self.datas = [self.scene.data] + [
      mujoco.MjData(model) for _ in range(self.worlds - 1)
    ]
    self.tasks = [PickPlace(self.scene, data, self.cfg.task) for data in self.datas]
    self.task = self.tasks[0]
    """The first world's task, which is the whole of it when there is one."""

  def _pace(self) -> None:
    """Fit the three clocks together: physics steps, rows, and control steps,
    and hand them to the plant that advances the worlds."""
    control_dt = 1.0 / self.cfg.control_hz
    substeps = round(control_dt / self.cfg.scene.physics_dt)
    if not np.isclose(substeps * self.cfg.scene.physics_dt, control_dt):
      raise ValueError("the control step must be a whole number of physics steps")
    self.hands = Hands(
      self.scene.grippers, substeps, control_dt, self.cfg.action, self.worlds
    )
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
    build = Plant if self.physics == "cpu" else WarpPlant
    self.plant = build(
      self.scene.model,
      self.datas,
      self.scene.grippers,
      self.rows,
      self._per_row,
      self._capture,
      self.hands,
    )

  def _furnish(self) -> None:
    """Allocate what an episode fills: seeds, clocks, samples and frames."""
    self._draws = [np.random.default_rng() for _ in range(self.worlds)]
    """One stream per world, so a world's start answers to its own seed."""
    self.np_random = self._draws[0]
    self._steps = np.zeros(self.worlds, np.int64)
    """How many control steps each world has run since it was last reset."""
    self._at = np.zeros(self.worlds, np.int64)
    """The row of the step timeline each world stands on."""
    self._frames: list[dict[int, dict[str, Any]]] = [{} for _ in range(self.worlds)]
    """What each world's lenses saw, by the row it was taken on."""
    self.renderer: Renderer | None = None
    self.watcher: Renderer | None = None
    self.views: dict[str, Any] = {}

  @property
  def observation_space(self) -> Any:
    return spaces.gym_spaces(self.image_shape, self.rows)[0]

  @property
  def action_space(self) -> Any:
    return spaces.gym_spaces(self.image_shape, self.rows)[1]

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
    self.hands.reset(self.datas, chosen)
    for world in chosen:
      self.tasks[world].reset()
      self._steps[world] = 0
      self._at[world] = 0
    self.plant.gather()
    for hand, grip in enumerate(self.scene.grippers):
      settled = self.plant.qpos[chosen][:, None, grip.qpos : grip.qpos + 7]
      self.plant.poses[chosen, hand] = settled
      self.plant.openings[chosen, hand] = self.plant.qpos[chosen][
        :, None, grip.fingers[0]
      ]
    # The rows before an episode are the state it settled into, and so is every
    # frame the first boundaries ask for.
    seen = self._render(chosen)
    for index, world in enumerate(chosen):
      self._frames[world] = dict.fromkeys(
        range(-self.cfg.camera_lag, 1, self.rows), seen[index]
      )
    return self._observe(), self._stacked([task.info() for task in self.tasks])

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
    """Settle one world onto a start it can hold.

    Retries redraw only what the caller left free, so a fully named start gets
    one attempt and fails loudly rather than quietly landing somewhere else.
    """
    fixed = {"state", "cube"} <= set(start)
    for _ in range(1 if fixed else self.cfg.reset_attempts):
      self._place(world, data, start)
      if not self._penetrating(data):
        return
    if start:
      raise ValueError("the named start puts two bodies through each other")
    randomise.nominal(self.scene, data)
    mujoco.mj_forward(self.scene.model, data)

  def _place(self, world: int, data: mujoco.MjData, options: dict[str, Any]) -> None:
    """Draw a start, honouring whichever half of it the caller named."""
    draw = self._draws[world]
    mujoco.mj_resetData(self.scene.model, data)
    if (state := options.get("state")) is not None:
      randomise.place_hands(self.scene, data, np.asarray(state, np.float64))
    else:
      randomise.sample_hands(self.scene, data, draw, self.cfg.randomise)
    if (cube := options.get("cube")) is not None:
      randomise.place_cube(self.scene, data, np.asarray(cube, np.float64))
    else:
      randomise.sample_cube(self.scene, data, draw, self.cfg.randomise)
    mujoco.mj_forward(self.scene.model, data)

  def step(
    self, action: Float64Array
  ) -> tuple[dict[str, Any], Float64Array, BoolArray, BoolArray, dict[str, Any]]:
    """Track the leading poses of a trajectory, and report where that leaves things.

    Every world is answered for along a leading axis, in reward, in what ended,
    and in everything an observation and its report carry, however many there
    are: a shape does not turn on how the environment was built.
    """
    self.hands.process(action)
    # A world is the truth between steps, so anything moved on one since the
    # last step is what this one runs from.
    self.plant.run(lambda: self._keep(self._capture))
    self._at += self.rows
    self._steps += 1
    judged = [task.evaluate() for task in self.tasks]
    reward = np.array([one[0] for one in judged], np.float64)
    terminated = np.array([one[1] for one in judged], np.bool_)
    truncated = (self._steps >= self.cfg.max_steps) & ~terminated
    return (
      self._observe(),
      reward,
      terminated,
      truncated,
      self._stacked([one[2] for one in judged]),
    )

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
    for hand, slot in enumerate(GRIPPERS):
      out[slot] = {
        "pose": self.plant.poses[:, hand].copy(),
        "opening": self.plant.openings[:, hand].copy(),
        "image": np.stack([frame[slot] for frame in frames]),
        "image_step": taken.copy(),
      }
    return out

  def _render(self, worlds: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """What the named worlds' lenses see, every world's when none are named."""
    if self.renderer is None:
      self.renderer = Renderer(self.scene.model, self.cfg.cameras, self.cfg.image_size)
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
