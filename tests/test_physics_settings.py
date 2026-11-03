"""What the solver and contact settings have to deliver. Each threshold sits
between what ``Physics()`` measures and what a wrong setting measures, so a
regression trips it and ordinary noise does not."""

from dataclasses import replace

import mujoco
import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.env import Cfg, Env
from specialist_sim.gripper import GRIPPING, SLOTS
from specialist_sim.rig import Clock
from specialist_sim.sort_lego import SortLego, initial

OPEN, SHUT = 0.3, -0.15
PITCH = np.radians(48.0)
"""How far the fingers tip down to reach a brick lying flat."""


GRASP = (0.123, 0.0, -0.0177)
"""A held brick's middle in the base tag's frame, meters: x fit over the
grasps against their reconstructed layout (123.4 mm, std 20), z where the pads'
centers sit, y the jaws' middle."""


def _slerp(a: Float64Array, b: Float64Array, fraction: float) -> Float64Array:
  """A ``fraction`` of the way from ``a`` to ``b`` along the shorter arc."""
  out: Float64Array = rotation.multiply(
    a,
    rotation.exp(fraction * rotation.log(rotation.multiply(rotation.conjugate(a), b))),
  )
  return out


BLIND = Cfg(
  clock=replace(Clock(), per_action=1, per_step=1), render=False, image_size=16
)
"""No lens, and one commanded pose a row: these measure the plant along a path,
not what a policy or a camera sees."""
PILE = SortLego(mode="pile", bricks=1)

ROW_HZ = 1.0 / BLIND.clock.row_dt


def _env() -> Env[SortLego]:
  return Env(PILE, BLIND)


def _geoms(env: Env[SortLego]) -> tuple[BoolArray, BoolArray]:
  model = env.rig.model
  bricks = np.isin(
    model.geom_bodyid,
    [model.body(f"brick_{i}").id for i in range(len(env.task.colors))],
  )
  return bricks, model.geom_condim == GRIPPING


def _deepest(data: mujoco.MjData, first: BoolArray, second: BoolArray) -> float:
  """The deepest any geom of ``first`` sinks into one of ``second``, metres."""
  deepest = 0.0
  for contact in data.contact[: data.ncon]:
    one, other = contact.geom
    if (first[one] and second[other]) or (first[other] and second[one]):
      deepest = max(deepest, -contact.dist)
  return deepest


def _grasp(brick: Float64Array, slot: str) -> Float64Array:
  """The hand state that holds ``brick`` between its pads, fingers along its
  long side and pointing away from the hand's home."""
  long_side = rotation.to_matrix(brick[3:7])[:, 0]
  along = np.array([long_side[0], long_side[1], 0.0]) / np.linalg.norm(long_side[:2])
  if np.dot(along[:2], brick[:2] - initial.home(slot)[:2]) < 0.0:
    along = -along
  turn = rotation.pointing(np.cos(PITCH) * along - np.array([0.0, 0.0, np.sin(PITCH)]))
  at = brick[:3] - rotation.rotate(turn, np.asarray(GRASP))
  return np.array([*at, *turn, OPEN])


def _eased(plan: list[tuple[float, Float64Array]], at: float) -> Float64Array:
  """A hand state along ``plan``, smoothstepped between its arrival times."""
  if at <= plan[0][0]:
    return plan[0][1]
  for (first, a), (second, b) in zip(plan, plan[1:], strict=False):
    if at <= second:
      part = (at - first) / (second - first)
      eased = part * part * (3.0 - 2.0 * part)
      out = a + eased * (b - a)
      out[3:7] = _slerp(a[3:7], b[3:7], eased)
      return out
  return plan[-1][1]


def _in_hand(state: Float64Array, brick: Float64Array) -> Float64Array:
  """A brick's pose in a hand's frame, ``(7,)``."""
  inverse = rotation.conjugate(state[3:7])
  return np.concatenate(
    [
      rotation.rotate(inverse, brick[:3] - state[:3]),
      rotation.multiply(inverse, brick[3:7]),
    ]
  )


def test_a_shaken_grasp_neither_slips_nor_sinks() -> None:
  """The elliptic cone with an impedance ratio of 100 holds a brick through a
  shake, and the pads' 2 ms time constant keeps them off its surface. A ratio
  of 10 slips 2.0 mm; softer pads sink 7.0 mm."""
  env = _env()
  bricks, pads = _geoms(env)
  observation, _ = env.reset([0])
  layout = env.task.layouts[0]
  assert layout is not None
  which = int(np.flatnonzero(layout.active)[0])
  homes = observation.state[0, -1].copy()
  grasp = _grasp(env.task.poses(0)[which], SLOTS[1])
  above, shut = grasp.copy(), grasp.copy()
  above[2] += 0.06
  shut[7] = SHUT
  lifted = shut.copy()
  lifted[2] += 0.10
  plan = [(0.0, homes[1]), (0.8, above), (1.3, grasp), (1.7, shut), (2.6, lifted)]
  data = env.rig.datas[0]
  sinks: list[float] = []
  held: Float64Array | None = None
  for row in range(round(4.8 * ROW_HZ)):
    at = row / ROW_HZ
    target = homes.copy()
    target[1] = _eased(plan, at)
    if at >= 2.8:
      phase = 2.0 * np.pi * 1.5 * (at - 2.8)
      ramp = min(1.0, (at - 2.8) / 0.3)
      target[1, :3] += 0.05 * np.sin(phase + np.array([0.0, 2.1, 4.2])) * ramp
    observation, _ = env.step(target[None, None])
    if at >= 2.0:
      sinks.append(_deepest(data, pads, bricks))
    if held is None and at >= 2.8:
      held = _in_hand(observation.state[0, -1, 1], env.task.poses(0)[which])
  assert held is not None
  end = _in_hand(observation.state[0, -1, 1], env.task.poses(0)[which])
  slip = float(np.linalg.norm(end[:3] - held[:3]))
  assert slip < 1.5e-3, f"the brick slipped {1e3 * slip:.1f} mm in the grasp"
  sank = float(np.median(sinks))
  assert sank < 2e-3, f"the pads sank {1e3 * sank:.1f} mm into the brick"


def test_resting_bricks_stop_moving() -> None:
  """A 2 ms contact time constant and near-zero object damping let a pile come
  down and stay down. Damping of 3e-4 leaves it creeping at 18.8 mm/s."""
  env = _env()
  bricks, _ = _geoms(env)
  layout = initial.draw("pile", 0, env.task.colors, env.task.bins, bricks=8)
  env.task.place(0, layout.poses, initial.starts(SLOTS))
  env.rig.settle([0], 1.5)
  observation = env.rig.begin([0])
  for _ in range(round(ROW_HZ)):
    observation, _ = env.step(observation.state[:, -1:])
  before = env.task.poses(0)
  for _ in range(round(ROW_HZ)):
    observation, _ = env.step(observation.state[:, -1:])
  crept = float(np.linalg.norm(env.task.poses(0)[:, :3] - before[:, :3], axis=1).max())
  assert crept < 2e-3, f"a resting brick crept {1e3 * crept:.1f} mm in a second"
  sank = _deepest(env.rig.datas[0], bricks, bricks)
  assert sank < 2e-3, f"bricks sank {1e3 * sank:.1f} mm into each other"


def test_a_brick_in_flight_keeps_its_spin() -> None:
  """Object damping is near zero, so flight takes almost nothing out of a
  tossed brick. At 3e-4 it stops turning; the footage turns at about 25 rad/s."""
  env = _env()
  poses = initial.parked(env.task.colors, env.task.bins)
  poses[0, :3] = [0.3, 0.3, 0.5]
  env.task.place(0, poses, initial.starts(SLOTS))
  observation = env.rig.begin([0])
  data = env.rig.datas[0]
  at = env.rig.model.joint("brick_0_joint").dofadr[0]
  spinning = np.array([25.0, 5.0, 3.0])
  data.qvel[at + 3 : at + 6] = spinning
  for _ in range(round(0.05 * ROW_HZ)):
    observation, _ = env.step(observation.state[:, -1:])
  left = float(np.linalg.norm(data.qvel[at + 3 : at + 6]) / np.linalg.norm(spinning))
  assert left > 0.5, f"only {100 * left:.0f}% of the spin survived 50 ms of flight"
