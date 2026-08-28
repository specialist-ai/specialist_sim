"""What varies between episodes.

A reset writes state before the environment checks it, so a bad placement is
drawn again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from specialist_sim import pose as se3
from specialist_sim.action import gripper_from_opening, turn
from specialist_sim.arrays import Float64Array
from specialist_sim.scene import Scene
from specialist_sim.spaces import GRIPPERS, PER_HAND

__all__ = [
  "RandomiseCfg",
  "home",
  "nominal",
  "place_cube",
  "place_hands",
  "sample_cube",
  "sample_hands",
]

HOME_XY = ((0.47, -0.16), (0.47, 0.16))
HOME_HEIGHT = 0.18
"""How far above the table a hand rests."""
# The fingers point across the table at the other hand, 40 degrees nose down.
_TILT = np.radians(40.0)
_MIRROR = np.diag([1.0, -1.0, 1.0])
_HELD = np.array(
  [
    [0.0, 1.0, 0.0],
    [-np.cos(_TILT), 0.0, -np.sin(_TILT)],
    [-np.sin(_TILT), 0.0, np.cos(_TILT)],
  ]
)
HOME_ROTATION = (_MIRROR @ _HELD @ _MIRROR, _HELD)
"""Each hand's resting orientation, mirrored across the table from each other."""
FINGER_HOME = np.radians(20.0)
CLEARANCE = 1e-4
"""How far above the table a placed cube starts, so it never begins inside it."""


@dataclass(frozen=True)
class RandomiseCfg:
  cube_x: tuple[float, float] = (0.42, 0.62)
  cube_y: tuple[float, float] = (-0.14, 0.14)
  tape_clearance: float = 0.06
  hand_jitter: float = 0.02
  hand_turn: float = np.radians(10.0)


def place_cube(
  scene: Scene, spot: Sequence[float] | Float64Array, yaw: float = 0.0
) -> None:
  """Rest the cube upright at a spot on the table, turned about its own axis."""
  quaternion = np.empty(4)
  mujoco.mju_axisAngle2Quat(quaternion, np.array([0.0, 0.0, 1.0]), yaw)
  address = scene.cube.qpos
  scene.data.qpos[address : address + 3] = (
    spot[0],
    spot[1],
    scene.cube.rest + CLEARANCE,
  )
  scene.data.qpos[address + 3 : address + 7] = quaternion


def place_hand(
  scene: Scene,
  hand: int,
  position: Sequence[float] | Float64Array,
  rotation: Float64Array,
  finger: float,
) -> None:
  """Put one hand at a pose, its two fingers mirrored to an angle."""
  grip = scene.grippers[hand]
  scene.data.qpos[grip.qpos : grip.qpos + 3] = position
  scene.data.qpos[grip.qpos + 3 : grip.qpos + 7] = se3.wxyz(rotation)
  for address, sign in zip(grip.fingers, (1.0, -1.0), strict=True):
    scene.data.qpos[address] = sign * finger


def place_hands(scene: Scene, state: Float64Array) -> None:
  """Put every hand exactly where a state vector says."""
  for hand in range(len(GRIPPERS)):
    one = state[hand * PER_HAND : (hand + 1) * PER_HAND]
    place_hand(
      scene,
      hand,
      one[:3],
      se3.rotation(turn(one[3:7])),
      gripper_from_opening(float(one[7])),
    )


def sample_cube(scene: Scene, rng: np.random.Generator, cfg: RandomiseCfg) -> None:
  """Anywhere on the table that is clear of the tape, at any quarter turn."""
  while True:
    spot = (rng.uniform(*cfg.cube_x), rng.uniform(*cfg.cube_y))
    if np.hypot(spot[0] - scene.tape[0], spot[1] - scene.tape[1]) >= cfg.tape_clearance:
      break
  place_cube(scene, spot, rng.uniform(0.0, np.pi / 2.0))


def sample_hands(scene: Scene, rng: np.random.Generator, cfg: RandomiseCfg) -> None:
  """Both hands near home, held part open."""
  for hand in range(len(GRIPPERS)):
    rotation = _turn(rng, cfg.hand_turn) @ HOME_ROTATION[hand]
    jitter = rng.uniform(-cfg.hand_jitter, cfg.hand_jitter, size=3)
    place_hand(scene, hand, home(scene, hand) + jitter, rotation, FINGER_HOME)


def nominal(scene: Scene) -> None:
  """A fixed valid state, for when sampling keeps failing."""
  mujoco.mj_resetData(scene.model, scene.data)
  place_cube(scene, (0.55, -0.10))
  for hand in range(len(GRIPPERS)):
    place_hand(scene, hand, home(scene, hand), HOME_ROTATION[hand], FINGER_HOME)


def home(scene: Scene, hand: int) -> Float64Array:
  """A hand's resting position, standing off above the table."""
  return np.array([*HOME_XY[hand], scene.table_top + HOME_HEIGHT])


def _turn(rng: np.random.Generator, limit: float) -> Float64Array:
  """A rotation of up to ``limit`` radians about a uniformly random axis."""
  axis = rng.normal(size=3)
  angle = rng.uniform(0.0, limit)
  quaternion, matrix = np.empty(4), np.empty(9)
  mujoco.mju_axisAngle2Quat(quaternion, axis / np.linalg.norm(axis), angle)
  mujoco.mju_quat2Mat(matrix, quaternion)
  return matrix.reshape(3, 3)
