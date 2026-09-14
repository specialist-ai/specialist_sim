"""The pick-and-place task: where a cube starts, and what it must do for an
episode to be won."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import mujoco
import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.env import Task, Verdict
from specialist_sim.gripper import SLOTS
from specialist_sim.pickplace import room as rooms
from specialist_sim.pickplace.room import Room
from specialist_sim.rig import Rig

__all__ = ["REGION", "PickPlace", "draw", "home", "on_tape"]

HOME_ACROSS = 0.16
"""How far to its side of the tape each hand rests, level with it along the
table; the near side's left is the room's +y."""
HOME_HEIGHT = 0.18
"""How far above the table a hand rests."""
JITTER = 0.02
TURN = np.radians(10.0)
"""How far from home, and how far turned, a hand starts."""
QUARTER = np.pi / 2.0
"""A square cube shows every face it has within a quarter turn."""
REGION = (0.32, 0.24)
"""How far a cube's start spreads about the tape, along x and across y, meters."""


def home(room: Room, side: int) -> Float64Array:
  """Where the hand on one side of the rig rests; ``side`` indexes ``SLOTS``."""
  across = HOME_ACROSS if side == 0 else -HOME_ACROSS
  return np.array([room.tape[0], room.tape[1] + across, room.table_top + HOME_HEIGHT])


def draw(
  room: Room, rng: np.random.Generator, region: tuple[float, float]
) -> tuple[Float64Array, Float64Array]:
  """Every hand near its home, turned a little, and the cube upright in
  ``region`` at a random yaw, clear of the tape."""
  hands = np.empty((len(room.slots), 8))
  for index, slot in enumerate(room.slots):
    side = SLOTS.index(slot)
    axis = rng.normal(size=3)
    turned = rotation.exp(axis / np.linalg.norm(axis) * rng.uniform(0.0, TURN))
    hands[index, :3] = home(room, side) + rng.uniform(-JITTER, JITTER, size=3)
    aim = np.array([*room.tape, room.table_top]) - hands[index, :3]
    hands[index, 3:7] = rotation.multiply(turned, rotation.pointing(aim))
    hands[index, 7] = 0.0
  half = np.array(region) / 2.0
  while True:
    spot = rng.uniform(room.tape - half, room.tape + half)
    if not on_tape(room, np.array([*spot, 0.0])):
      return hands, np.array([*spot, rng.uniform(0.0, QUARTER)])


PLACE_TOLERANCE = 0.010
"""How far the cube's center may stand beyond the tape's half-width."""
REST_HEIGHT = 0.005
REST_SPEED = 0.02
REST_SPIN = 0.5
UPRIGHT = 0.966
"""What a resting cube is: seated on the table, still, and with a face up."""
DROP_DEPTH = 0.10
"""How far below the table top a cube has fallen off it."""
PICK_LIFT = 0.010
"""How far off the table the cube must be carried for an episode to count as
a pick rather than a nudge."""


def on_tape(room: Room, position: Float64Array) -> BoolArray:
  """Whether a cube at each position, ``(..., 3)``, stands on the tape."""
  out: BoolArray = (
    np.abs(position[..., :2] - room.tape) <= room.tape_half + PLACE_TOLERANCE
  ).all(axis=-1)
  return out


def _evaluate(room: Room, data: mujoco.MjData) -> tuple[Float64Array, bool, bool, bool]:
  """One env's cube: its position, and whether it is resting, held and
  dropped."""
  cube = room.cube
  position = data.xpos[cube.body]
  velocity = data.cvel[cube.body]
  upright = float(np.abs(data.xmat[cube.body].reshape(3, 3)[2]).max())
  resting = bool(
    abs(position[2] - cube.rest) < REST_HEIGHT
    and np.linalg.norm(velocity[3:]) < REST_SPEED
    and np.linalg.norm(velocity[:3]) < REST_SPIN
    and upright > UPRIGHT
  )
  pairs = data.contact.geom[: data.ncon]
  touching = pairs[(pairs == cube.geom).any(axis=1)]
  held = bool(np.isin(touching, room.pads).any())
  dropped = bool(position[2] < room.table_top - DROP_DEPTH)
  return position.copy(), resting, held, dropped


@dataclass
class PickPlace(Task):
  """Carry the cube to the tape and let it go. It is won once the cube rests
  there, untouched, for ``dwell`` steps, and lost once it falls off the table."""

  region: tuple[float, float] = REGION
  """How far the cube's start spreads about the tape, along x and across y."""

  name: ClassVar[str] = "pickplace"
  dwell: ClassVar[int] = 5
  latched: ClassVar[tuple[str, ...]] = ("picked",)

  def spec(self) -> mujoco.MjSpec:
    return rooms.spec()

  @property
  def focus(self) -> Float64Array:
    return np.array([*self.room.tape, self.room.table_top])

  def bind(self, rig: Rig) -> None:
    super().bind(rig)
    self.room = Room(rig.model, rig.hands)

  def open(self, env_idx: int, seed: int, start: Any) -> Float64Array:
    """Both hands near their homes and the cube somewhere in the region, or
    where ``start`` says: every hand's state and the cube's ``(x, y, yaw)``."""
    rig, rng = self.rig, np.random.default_rng(seed)
    hands, cube = draw(self.room, rng, self.region) if start is None else start
    rig.place(env_idx, hands)
    self.room.place_cube(rig.datas[env_idx], cube)
    drawn = rng.uniform(size=len(rig.slots))
    return np.array([drawn[rig.slots.index(slot)] for slot in rig.lenses])

  def cube(self) -> Float64Array:
    """Every env's cube position, ``(num_envs, 3)``."""
    return np.stack([data.xpos[self.room.cube.body] for data in self.rig.datas])

  def judge(self) -> Verdict:
    seen = [_evaluate(self.room, data) for data in self.rig.datas]
    cube = np.stack([one[0] for one in seen])
    resting = np.array([one[1] for one in seen])
    held = np.array([one[2] for one in seen])
    dropped = np.array([one[3] for one in seen])
    on_target = on_tape(self.room, cube)
    placed = on_target & resting & ~held
    return Verdict(
      progress=placed.astype(np.float64),
      solved=placed,
      failed=dropped,
      labels={
        "dropped": dropped,
        "held": held,
        "on_tape": on_target,
        "picked": held & (cube[:, 2] > self.room.cube.rest + PICK_LIFT),
      },
    )
