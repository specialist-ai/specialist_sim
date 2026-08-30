"""A pick-and-place environment for evaluating policies.

Follows the Gymnasium interface without importing it.

The environment is a stack of sensors on one clock. A row is one tick of the
step timeline; a control step spans several and returns at the last of them.
Every value an observation carries is labelled with the row it happened at, and
nothing here reports an age or a delay: a reader subtracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from specialist_sim import randomise, spaces
from specialist_sim.action import ActionCfg, Hands
from specialist_sim.arrays import Float64Array
from specialist_sim.calibration import SCENE_VIEW, CameraCfg
from specialist_sim.observation import read_hands
from specialist_sim.randomise import RandomiseCfg
from specialist_sim.render import Renderer
from specialist_sim.scene import Scene, SceneCfg
from specialist_sim.spaces import F_STEP, GRIPPERS
from specialist_sim.task import PickPlace, TaskCfg

__all__ = ["PickPlaceCfg", "PickPlaceEnv"]


MAX_LAG = 20
"""How many rows a camera may run behind the rest of the stack."""


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

  def __init__(self, cfg: PickPlaceCfg | None = None, render_mode: str | None = None):
    self.cfg = cfg or PickPlaceCfg()
    modes = PickPlaceEnv.metadata["render_modes"]
    if render_mode is not None and render_mode not in modes:
      raise ValueError(f"{render_mode} is not one of {modes}")
    if set(self.cfg.cameras) != set(GRIPPERS):
      raise ValueError(
        f"the cameras are {sorted(GRIPPERS)}, not {sorted(self.cfg.cameras)}"
      )
    self.render_mode = render_mode
    self.metadata = {**PickPlaceEnv.metadata, "render_fps": round(self.cfg.control_hz)}
    self.scene = Scene(self.cfg.scene)
    self.task = PickPlace(self.scene, self.cfg.task)
    control_dt = 1.0 / self.cfg.control_hz
    substeps = round(control_dt / self.cfg.scene.physics_dt)
    if not np.isclose(substeps * self.cfg.scene.physics_dt, control_dt):
      raise ValueError("the control step must be a whole number of physics steps")
    self.hands = Hands(self.scene.grippers, substeps, control_dt, self.cfg.action)
    rows = F_STEP / self.cfg.control_hz
    if not np.isclose(rows, round(rows)):
      raise ValueError("a control step must span a whole number of rows")
    self.rows = round(rows)
    """How many rows of the step timeline one control step spans."""
    if substeps % self.rows:
      raise ValueError("a row must span a whole number of physics steps")
    self._within = substeps // self.rows
    if not 0 <= self.cfg.camera_lag <= MAX_LAG:
      raise ValueError(
        f"a camera runs 0 to {MAX_LAG} rows behind, not {self.cfg.camera_lag}"
      )
    self.np_random = np.random.default_rng()
    self._steps = 0
    self._row = 0
    hands = len(GRIPPERS)
    self._poses = np.zeros((hands, self.rows, 7), np.float32)
    self._openings = np.zeros((hands, self.rows), np.float32)
    self._frames: dict[int, dict[str, Any]] = {}
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
    self, *, seed: int | None = None, options: dict[str, Any] | None = None
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    if seed is not None:
      self.np_random = np.random.default_rng(seed)
    options = options or {}
    if not set(options) <= {"state", "cube"}:
      raise ValueError(f"a start names state and cube, not {sorted(options)}")
    for _ in range(self.cfg.reset_attempts):
      self._place(options)
      if not self._penetrating():
        break
      if options:
        raise ValueError("the named start puts two bodies through each other")
    else:
      randomise.nominal(self.scene)
      mujoco.mj_forward(self.scene.model, self.scene.data)
    self.hands.reset(self.scene.data)
    self.task.reset()
    self._steps = 0
    self._row = 0
    poses, openings = read_hands(self.scene)
    self._poses[:] = poses[:, None, :]
    self._openings[:] = openings[:, None]
    # The rows before an episode are the state it settled into, and so is every
    # frame the first boundaries ask for.
    self._frames = dict.fromkeys(
      range(-self.cfg.camera_lag, 1, self.rows), self._render()
    )
    return self._observe(), self.task.info()

  def _place(self, options: dict[str, Any]) -> None:
    """Draw a start, honouring whichever half of it the caller named."""
    mujoco.mj_resetData(self.scene.model, self.scene.data)
    if (state := options.get("state")) is not None:
      randomise.place_hands(self.scene, np.asarray(state, np.float64))
    else:
      randomise.sample_hands(self.scene, self.np_random, self.cfg.randomise)
    if (cube := options.get("cube")) is not None:
      randomise.place_cube(self.scene, np.asarray(cube, np.float64))
    else:
      randomise.sample_cube(self.scene, self.np_random, self.cfg.randomise)
    mujoco.mj_forward(self.scene.model, self.scene.data)

  def step(
    self, action: Float64Array
  ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    """Track the leading poses of a trajectory, and report where that leaves things."""
    self.hands.process(np.asarray(action, np.float64), self.scene.data)
    boundary = self._row + self.rows
    capture = boundary - (self.cfg.camera_lag % self.rows)
    for substep in range(self.hands.substeps):
      self.hands.apply(self.scene.data, substep)
      mujoco.mj_step(self.scene.model, self.scene.data)
      if (substep + 1) % self._within:
        continue
      row = (substep + 1) // self._within
      self._poses[:, row - 1], self._openings[:, row - 1] = read_hands(self.scene)
      if self._row + row == capture < boundary:
        self._frames[capture] = self._captured()
    mujoco.mj_forward(self.scene.model, self.scene.data)
    if capture == boundary:
      self._frames[capture] = self._render()
    self._row = boundary
    self._steps += 1
    reward, terminated, info = self.task.evaluate()
    truncated = self._steps >= self.cfg.max_steps and not terminated
    return self._observe(), reward, terminated, truncated, info

  def render(self) -> np.ndarray | None:
    """The room, watched from across the table."""
    if self.render_mode is None:
      return None
    if self.watcher is None:
      self.watcher = self._built({"scene_view": SCENE_VIEW}, self.cfg.render_size)
    watched: np.ndarray = self.watcher.render(self.scene.data)["scene_view"].image
    return watched

  def close(self) -> None:
    self.renderer = None
    self.watcher = None

  def _observe(self) -> dict[str, Any]:
    """The rows this step spanned, and the frame that has caught up with it."""
    taken = self._row - self.cfg.camera_lag
    frame = self._frames.pop(taken)
    for older in [row for row in self._frames if row < taken]:
      del self._frames[older]
    out: dict[str, Any] = {
      "step": np.arange(self._row - self.rows + 1, self._row + 1, dtype=np.int64)
    }
    for hand, slot in enumerate(GRIPPERS):
      out[slot] = {
        "pose": self._poses[hand].copy(),
        "opening": self._openings[hand].copy(),
        "image": frame[slot],
        "image_step": np.array(taken, np.int64),
      }
    return out

  def _render(self) -> dict[str, Any]:
    """What the lenses see of the configuration the model currently holds."""
    if self.renderer is None:
      self.renderer = self._built(self.cfg.cameras, self.cfg.image_size)
    self.views = self.renderer.render(self.scene.data)
    return {slot: view.image for slot, view in self.views.items()}

  def _captured(self) -> dict[str, Any]:
    """A frame taken partway through a control step.

    A physics step leaves the frames describing the configuration it started
    from, and the impedance reads them again on the next substep, so they are
    carried forward for the camera and put back exactly as they were.
    """
    data = self.scene.data
    held = (data.xpos.copy(), data.xquat.copy(), data.xmat.copy())
    mujoco.mj_kinematics(self.scene.model, data)
    frame = self._render()
    data.xpos[:], data.xquat[:], data.xmat[:] = held
    return frame

  def _built(self, cameras: dict[str, CameraCfg], size: int) -> Renderer:
    """A renderer with its lenses seated, and the scene carried onto the seats."""
    built = Renderer(self.scene.model, cameras, size)
    mujoco.mj_forward(self.scene.model, self.scene.data)
    return built

  def _penetrating(self) -> bool:
    data = self.scene.data
    return any(
      data.contact[index].dist < -self.cfg.penetration for index in range(data.ncon)
    )
