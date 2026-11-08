"""The pick-and-place task."""

from dataclasses import dataclass
from typing import ClassVar

import mujoco
import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array
from specialist_sim.env import Objects, Task, Verdict
from specialist_sim.gripper import ACROSS, DRAWN, HANDS
from specialist_sim.tasks import desk

TAPE_HALF = 0.013
TAPE_HALF_THICK = 0.0002
# The cube is on the tape while its middle is this near the tape's, along x and y.
ON_TAPE = TAPE_HALF + 0.010
CUBE_HALF = 0.015
# The height of the cube's middle when it sits on the desk.
REST = desk.TOP + CUBE_HALF
# How far to its side of the tape each hand rests, level with it along x.
HOME_ACROSS = 0.16
# How far above the desk a hand rests.
HOME_OVER_DESK = 0.18
# How far the cube's start spreads about the tape, along x and across y.
REGION = (0.32, 0.24)
UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class PickPlace(Task):
  """Carry a cube onto a tape target on the desk's middle. It is won once the
  cube rests there, let go of, for ``dwell`` seconds, and lost once it falls
  off the desk."""

  dwell: ClassVar[float] = 0.5
  latched: ClassVar[tuple[str, ...]] = ("picked",)
  focus: ClassVar[Float64Array] = np.array([0.0, 0.0, desk.TOP])

  def spec(self) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    room = spec.worldbody.add_body(name="room")
    desk.stand(room)
    room.add_geom(
      name="tape",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=[0.0, 0.0, desk.TOP + TAPE_HALF_THICK],
      size=[TAPE_HALF, TAPE_HALF, TAPE_HALF_THICK],
      rgba=[0.15, 0.32, 0.70, 1.0],
      group=DRAWN,
      contype=0,
      conaffinity=0,
    )
    cube = spec.worldbody.add_body(name="cube", pos=[0.0, 0.0, REST + desk.GAP])
    cube.add_joint(name="cube", type=mujoco.mjtJoint.mjJNT_FREE)
    cube.add_geom(
      name="cube",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=[CUBE_HALF] * 3,
      rgba=[0.75, 0.12, 0.1, 1.0],
      mass=0.012,
      friction=list(desk.FRICTION),
      group=DRAWN,
    )
    return spec

  def draw(
    self, rng: np.random.Generator, driven: tuple[str, ...]
  ) -> tuple[Float64Array, Float64Array]:
    """Every hand near its home, turned a little and aimed at the tape, and
    the cube upright in ``REGION`` at a random yaw, off the tape."""
    hands = np.zeros((len(HANDS), 8))
    for index, hand in enumerate(HANDS):
      axis = rng.normal(size=3)
      tilt = rng.uniform(0.0, 0.1745)
      turned = rotation.exp(axis / np.linalg.norm(axis) * tilt)
      home = np.array([0.0, ACROSS[hand] * HOME_ACROSS, desk.TOP + HOME_OVER_DESK])
      hands[index, :3] = home + rng.uniform(-0.02, 0.02, size=3)
      aim = rotation.pointing(self.focus - hands[index, :3])
      hands[index, 3:7] = rotation.multiply(turned, aim)
    half = np.array(REGION) / 2.0
    spot = rng.uniform(-half, half)
    while (np.abs(spot) <= ON_TAPE).all():
      spot = rng.uniform(-half, half)
    yaw = rng.uniform(0.0, np.pi / 2.0)  # a cube repeats every quarter turn
    return hands, np.array([[*spot, REST + desk.GAP, *rotation.exp(yaw * UP)]])

  def judge(self, objects: Objects) -> Verdict:
    position, quaternion = objects.poses[:, 0, :3], objects.poses[:, 0, 3:]
    velocity = objects.velocity[:, 0]
    upright = np.abs(rotation.rotate(rotation.conjugate(quaternion), UP)).max(-1)
    resting = (
      (np.abs(position[:, 2] - REST) < 0.005)
      & (np.linalg.norm(velocity[:, 3:], axis=-1) < 0.02)
      & (np.linalg.norm(velocity[:, :3], axis=-1) < 0.5)
      & (upright > 0.966)  # a face within 15 degrees of up
    )
    on_tape = (np.abs(position[:, :2]) <= ON_TAPE).all(-1)
    held = objects.held[:, 0]
    placed = on_tape & resting & ~held
    return Verdict(
      progress=placed.astype(np.float64),
      failed=desk.fallen(position[:, 2]),
      labels={"picked": held & (position[:, 2] > REST + 0.010)},
    )
